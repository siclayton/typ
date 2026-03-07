#include "MicroBit.h"
#define ARM_MATH_CM4
#include "arm_math.h"
#include "SpeakerClassification.h"
#include "DecisionTree.h"

#include <cstdio>

#define NUM_SAMPLES 150

class MicSink : public DataSink {
    public:
        MicSink(DataSource &source);
        int pullRequest() override;
        ManagedBuffer getBuffer();
    private:
        DataSource &upstream;
        ManagedBuffer buffer;
        volatile bool newDataAvailable;
};

MicSink::MicSink(DataSource &source) : upstream(source) {
    upstream.connect(*this);
    upstream.dataWanted(DATASTREAM_WANTED);
    buffer = ManagedBuffer();
    newDataAvailable = false;
}

int MicSink::pullRequest() {
    buffer = upstream.pull();

    if (buffer.length() == 0) {
        return DEVICE_NO_DATA;
    }

    newDataAvailable = true;
    return DEVICE_OK;
}

ManagedBuffer MicSink::getBuffer() {
    while (!newDataAvailable) {
        fiber_sleep(1);
    }

    newDataAvailable = false;
    return buffer;
}

// Global variables
MicroBit uBit;
arm_rfft_fast_instance_f32 fftInstance;
auto sink = MicSink(*uBit.audio.splitter->createChannel());
DecisionTree model;

int currentClass = 0; //The ID for the class that the user is currently providing samples of
int currentSample = 0; //The position in the samples array to add the next sample to
TrainingSample samples[NUM_SAMPLES];
bool training = true;

int melBins[NUM_MEL + 2];
float melWeights[NUM_MEL][FFT_SIZE / 2];
float window[FFT_SIZE];
float fftOutput[FFT_SIZE];
float magnitudeSpectrum[FFT_SIZE / 2];
float melOutput[NUM_MEL];

void applyMelFilters(float* fft, float* mel) {
    // Loop over the filters
    for (int i = 0; i < NUM_MEL; i++) {
        float sum = 0;

        // Apply filter
        for (int j = melBins[i]; j < melBins[i + 2]; j++) {
            sum += fft[j] * melWeights[i][j];
        }
        mel[i] = sum;
    }
}

void applyfftAndMel() {
    // Calculate FFT and magnitude spectrum
    arm_rfft_fast_f32(&fftInstance, window, fftOutput, 0);
    arm_cmplx_mag_f32(fftOutput, magnitudeSpectrum, FFT_SIZE / 2);

    // Apply Mel Filterbank to the magnitude spectrum
    applyMelFilters(magnitudeSpectrum, melOutput);
}

SpeechSample takeSample() {
    int windowIndex = 0;
    int count = 0;
    uint64_t start = system_timer_current_time();

    float means[NUM_MEL] = {0};
    float m2s[NUM_MEL] = {0};

    while (system_timer_current_time() - start < 1000) {
        ManagedBuffer buf = sink.getBuffer();
        uint8_t* bytes = buf.getBytes();

        // Get the data back into 16 bit format
        auto micSamples = reinterpret_cast<int16_t*>(bytes);

        int numSamples = buf.length() / 2;

        // Store samples in a window until we have enough to perform a fft
        for (int i = 0; i < numSamples; i++) {
            window[windowIndex++] = micSamples[i]  * (1.0f / 32768.0f); // Normalises the data to floats in the range (-1, 1)

            if (windowIndex == FFT_SIZE) {
                applyfftAndMel();
                windowIndex = 0;
                count++;

                // Calculate mean and m2 (used in variance calculation) to allow aggregation of the melOutput for different windows
                for (int j = 0; j < NUM_MEL; j++) {
                    float diff = melOutput[j] - means[j];

                    means[j] += diff / count;
                    m2s[j] += diff * (melOutput[j] - means[j]);
                }
            }
        }
    }

    count--;
    float variances[NUM_MEL] = {0};
    for (int i = 0; i < NUM_MEL; i++) {
        variances[i] = m2s[i] / count;
    }

    // Pack aggregated melOutputs into a SpeechSample
    SpeechSample sample;
    for (int i = 0; i < NUM_MEL; i++) {
        sample.features[i] = log10f(means[i] + 1e-6f);
    }
    for (int i = 0; i < NUM_MEL; i++) {
        sample.features[NUM_MEL + i] = variances[i];
    }

    //Print the sample for debugging
    //Build the line to print in a buffer and print once (to put everything on one line in the output)
    char buf[128];
    int offset = 0;

    for (int i = 0; i < NUM_FEATURES; i++) {
        int value = static_cast<int>(sample.features[i] * 1000);
        offset += snprintf(
            buf + offset,
            sizeof(buf) - offset,
            "%d%s",
            value,
            i < NUM_FEATURES - 1 ? "," : ""
        );
    }

    DMESG("%s", buf);

    return sample;
}

float hzToMel(float hz) {
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

float melToHz(float mel) {
    return 700.0f * (powf(10.0f, mel / 2959.0f) - 1.0f);
}

void computeMelFilterbank() {
    // Initialise all weights to 0
    memset(melWeights, 0, sizeof(melWeights));

    float low = hzToMel(0);
    float high = hzToMel(SAMPLE_RATE / 2);
    float step = (high - low) / (NUM_MEL + 1);

    // Calculate what mel values go into each fft bin
    // Need NUM_MEL_FILTERS + 2 as each of the triangles need 3 points
    for (int i = 0; i < NUM_MEL + 2; i++) {
        float hz = melToHz(low + i * step);

        melBins[i] = static_cast<int>((FFT_SIZE + 1) * hz / SAMPLE_RATE);
    }

    // Build mel triangle
    for (int i = 0; i < NUM_MEL; i++) {
        int left = melBins[i];
        int centre = melBins[i + 1];
        int right = melBins[i + 2];

        // Rising slope
        for (int j = left; j < centre; j++) {
            melWeights[i][j] = static_cast<float>(j - left) / (centre - left);
        }
        // Falling slope
        for (int j = centre; j < right; j++) {
            melWeights[i][j] = static_cast<float>(right - j) / (right - centre);
        }
    }
}

void onButtonA(MicroBitEvent e){
    SpeechSample speechSample = takeSample();

    if (training) {
        TrainingSample sample = {currentClass, speechSample};
        samples[currentSample++] = sample;
    } else {
        int prediction = model.predict(speechSample);
         uBit.display.print(prediction);
    }
}
void onButtonB(MicroBitEvent e) {
    currentClass++;
}
void onButtonAB(MicroBitEvent e) {
    model = DecisionTree(NUM_FEATURES, currentClass + 1, currentSample, samples);
    uBit.serial.printf("Model trained\r\n");
    training = false;
}

int main() {
    uBit.init();
    uBit.audio.enable();

    // Initialise the fft instance
    arm_rfft_fast_init_f32(&fftInstance, FFT_SIZE);

    // Compute Mel filterbank
    computeMelFilterbank();

    uBit.messageBus.listen(MICROBIT_ID_BUTTON_A, MICROBIT_BUTTON_EVT_CLICK, onButtonA);
    uBit.messageBus.listen(MICROBIT_ID_BUTTON_B, MICROBIT_BUTTON_EVT_CLICK, onButtonB);
    uBit.messageBus.listen(MICROBIT_ID_BUTTON_AB, MICROBIT_BUTTON_EVT_CLICK, onButtonAB);

    release_fiber();
}