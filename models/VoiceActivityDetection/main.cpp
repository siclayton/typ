#include "MicroBit.h"
#include "VoiceActivityDetection.h"
#include "LogisticRegression.h"

#define NUM_SAMPLES 100

MicroBit uBit;

int currentClass = 0; //The ID for the class that the user is currently providing samples of
int currentSample = 0; //The position in the samples array to add the next sample to
bool training = true;
TrainingSample samples[NUM_SAMPLES];

//Initialise the model variable to nothing (until the model is trained)
//Acts as a placeholder until a trained model is created
LogisticRegressionModel* model = nullptr;

/**
 * Scale features to prevent dot product from exploding
 * @param variance a pointer to the variance
 * @param max a pointer to the max
 * @param zcr a pointer to the zero-crossing rate
 */
void scaleFeatures(float* variance, float* max, float* zcr) {
    //Very naive scaling based on seen values when training
    *variance = (*variance - 50.0f) / 50.0f;
    *max = (*max - 425.0f) / 50.0f;
    *zcr = (*zcr - 0.16f) / 0.04f;
}
/**
 * Collect a sample which represents whether there is voice activity or not
 * @return the collected sample
 */
MicrophoneSample takeSample() {
    //Discard the first 1/4 of a second of measurements to ensure samples don't include noise
    uint64_t discardStart = system_timer_current_time();
    while (system_timer_current_time() - discardStart < 250){
        uBit.io.microphone.getAnalogValue();
    }

    uint64_t start = system_timer_current_time();

    int prev = uBit.io.microphone.getAnalogValue();

    //The mean and variance of the samples is calculated using Welford's algorithm
    //This allows for calculations are the stream of inputs is coming in
    float count = 0;
    auto mean = static_cast<float>(prev);
    float m2 = 0; //Running sum of squared differences from the mean

    float max = 0;
    float zeroCrossings = 0;

    while (system_timer_current_time() - start < 1000) {
        const int threshold = 2;
        int value = uBit.io.microphone.getAnalogValue();
        count++;

        //Update max value
        if (value > max) max = value;

        //Check if there was a "zeroCrossing"
        //As the range of values from the microphone are 0-1024, zerocrossings are calculated over the mean
        float diff1 = prev - mean;
        float diff2 = value - mean;
        if ((diff1 > threshold && diff2 < -threshold) || (diff1 < -threshold && diff2 > threshold)) {
            zeroCrossings++;
        }

        //Update values needed to calculate variance
        float diff = static_cast<float>(value) - mean;
        mean += diff / count;
        m2 += diff * (static_cast<float>(value) - mean);

        prev = value;
        uBit.sleep(2);
    }

    float variance = m2 / (count - 1);
    float zcr = zeroCrossings / count; //Divide by the count to get the zero crossing rate

    scaleFeatures(&variance, &max, &zcr);

    MicrophoneSample sample = {variance, max, zcr};

    uBit.serial.printf("Sample %d: %d, %d, %d\r\n",
        currentSample, static_cast<int>(variance * 1000), max, static_cast<int>(zcr * 1000)
    );

    return sample;
}

void onButtonA(MicroBitEvent e) {
    MicrophoneSample micSample = takeSample();

    //If in training mode, label sample and add it to list of samples used to train the model
    //Otherwise, predict the class of the sample collected
    if (training) {
        TrainingSample sample = {currentClass, micSample};
        samples[currentSample++] = sample;
    } else {
        int prediction = model->predictClass(micSample);
        uBit.serial.printf("Prediction %d\r\n", prediction);
        uBit.display.print(prediction);
    }
}

void onButtonB(MicroBitEvent e) {
    currentClass++;
}

void onButtonAB(MicroBitEvent e) {
    //Update model variable to a trained LogisticRegressionModel instance
    delete model;
    model = new LogisticRegressionModel(NUM_FEATURES, currentSample, samples);

    uBit.serial.printf("Model trained\r\n");
    training = false;
}

int main() {
    uBit.init();

    //Turn on the microphone
    uBit.audio.enable();
    uBit.audio.activateMic();

    uBit.messageBus.listen(MICROBIT_ID_BUTTON_A, MICROBIT_BUTTON_EVT_CLICK, onButtonA);
    uBit.messageBus.listen(MICROBIT_ID_BUTTON_B, MICROBIT_BUTTON_EVT_CLICK, onButtonB);
    uBit.messageBus.listen(MICROBIT_ID_BUTTON_AB, MICROBIT_BUTTON_EVT_CLICK, onButtonAB);

    release_fiber();
}
