#include "MicroBit.h"
#include "arm_math.h"

#define SAMPLE_RATE (11 * 1024)
#define AUDIO_SAMPLES_NUMBER 1024
#define NUM_MEL_FILTERS 10
#define FFT_SIZE 254

MicroBit uBit;
arm_rfft_fast_instance_f32 fftInstance;

int melBins[NUM_MEL_FILTERS + 2];
float melWeights[NUM_MEL_FILTERS][FFT_SIZE/2];

typedef struct {
    float features[NUM_MEL_FILTERS];
} SpeechSample;

typedef struct {
    int sampleClass;
    SpeechSample sample;
} TrainingSample;

void applyMelFilters(float* fft, float* mel) {
    // Loop over the filters
    for (int i = 0; i < NUM_MEL_FILTERS; i++) {
        float sum = 0;

        for (int j = melBins[i]; j < melBins[i + 2]; j++) {
            sum += fft[j] * melWeights[i][j];
        }

        mel[i] = logf(sum);
    }
}

SpeechSample takeSample() {

    // Capture a seconds worth of audio
        // Will output the buf used in fft function call

    // Calculate FFT and magnitude spectrum
    auto* fftOutput = static_cast<float *>(malloc(sizeof(float) * AUDIO_SAMPLES_NUMBER));
    auto* magnitudeSpectrum = static_cast<float *>(malloc(sizeof(float) * AUDIO_SAMPLES_NUMBER / 2));

    arm_rfft_fast_f32(&fftInstance, buf + offset, fftOutput, 0);
    arm_cmplx_mag_f32(fftOutput, magnitudeSpectrum, AUDIO_SAMPLES_NUMBER / 2);

    // Apply Mel Filterbank to the magnitude spectrum
    auto* melOutput = static_cast<float *>(malloc(sizeof(float) * NUM_MEL_FILTERS));
    applyMelFilters(fftOutput, melOutput);

    // Pack feature vector returned by mel filter bank calculations into a SpeechSample
    SpeechSample sample;
    for (int i = 0; i < NUM_MEL_FILTERS; i++) {
        sample.features[i] = melOutput[i];
    }

    free(fftOutput);
    free(magnitudeSpectrum);
    free(melOutput);

    return sample;
}

float hzToMel(float hz) {
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

float melToHz(float mel) {
    return 700.0f * (powf(10.0f, mel / 2959.0f) - 1.0f);
}

void computeMelFilterbank() {
    float low = hzToMel(0);
    float high = hzToMel(SAMPLE_RATE / 2);
    float step = (high - low) / NUM_MEL_FILTERS + 1;

    // Calculate what mel values go into each fft bin
    // Need NUM_MEL_FILTERS + 2 as each of the triangles need 3 points (
    for (int i = 0; i < NUM_MEL_FILTERS + 2; i++) {
        float hz = melToHz(low + i * step);

        melBins[i] = static_cast<int>((FFT_SIZE + 1) * hz / SAMPLE_RATE);
    }

    // Build mel triangle
    for (int i = 0; i < NUM_MEL_FILTERS; i++) {
        int left = melBins[i];
        int centre = melBins[i + 1];
        int right = melBins[i + 2];

        // Rising slope
        for (int j = left; j < centre; j++) {
            melWeights[i][j] = static_cast<float>((j - left) / (centre - left));
        }
        // Falling slope
        for (int j = centre; j < right; j++) {
            melWeights[i][j] = static_cast<float>((right - j) / (right - centre));
        }
    }
}

void onButtonA(MicroBitEvent e){
    //Take sample
}
void onButtonB(MicroBitEvent e);
void onButtonAB(MicroBitEvent e);

int main() {
    uBit.init();
    uBit.audio.enable();

    // Initialise the fft instance
    arm_rfft_fast_init_f32(&fftInstance, AUDIO_SAMPLES_NUMBER);

    // Compute Mel filterbank

    uBit.messageBus.listen(MICROBIT_ID_BUTTON_A, MICROBIT_BUTTON_EVT_CLICK, onButtonA);
    uBit.messageBus.listen(MICROBIT_ID_BUTTON_B, MICROBIT_BUTTON_EVT_CLICK, onButtonB);
    uBit.messageBus.listen(MICROBIT_ID_BUTTON_AB, MICROBIT_BUTTON_EVT_CLICK, onButtonAB);

    release_fiber();
}