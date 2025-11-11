//
// Created by simon on 11/11/2025.
//
#include "MicroBit.h"

MicroBit uBit;
int currentClass = 0; //The ID for the class that the user is currently providing samples of
int currentSample = 0; //The position in the samples array to add the next sample to

typedef struct {
    int sampleClass;

    float mean;
    float variance;

    int max;

    int zeroCrossings;
} MicrophoneSample;

MicrophoneSample samples[20];

MicrophoneSample takeSample(int sampleClass) {
    uint64_t start = system_timer_current_time();

    //The mean and variance of the samples is calculated using Welford's algorithm
    //This allows for calculations are the stream of inputs is coming in
    int count = 0;
    float mean = 0;
    float m2 = 0; //Running sum of squared differences from the mean

    int max = 0;
    int zeroCrossings = 0;

    int lastValueWasNegative = 0;

    while (system_timer_current_time() - start < 1000) {
        int value = uBit.io.microphone.getAnalogValue();
        count++;

        //Update max value
        if (value > max) max = value;

        if (lastValueWasNegative == 0 && value < 0) {
            zeroCrossings++;
            lastValueWasNegative = 1;
        } else if (lastValueWasNegative == 1 && value > 0) {
            zeroCrossings++;
            lastValueWasNegative = 0;
        }

        //Calculate mean and variance for each axis
        float diff = (float) value - mean;
        mean += diff / count;
        m2 += diff * (value - mean);

        uBit.sleep(2);
    }

    float variance = m2 / (count - 1);

    MicrophoneSample sample = {sampleClass, mean, variance, max, zeroCrossings};

    return sample;
}

void onButtonA(MicroBitEvent e) {
    MicrophoneSample sample = takeSample(currentClass);
    samples[currentSample++] = sample;
}

void onButtonB(MicroBitEvent e) {
    currentClass++;
}

void onButtonAB(MicroBitEvent e) {
    uBit.serial.printf("Samples array:\r\n");

    for (int i = 0; i < currentSample; i++) {
        uBit.serial.printf("%d, %d, %d, %d\r\n",
        samples[i].sampleClass,
        (int) (samples[i].mean * 1000), (int) (samples[i].variance * 1000), (int) samples[i].max);

        uBit.sleep(20);
    }
}

int main() {
    uBit.init();
    uBit.serial.setBaud(115200);

    uBit.audio.enable();
    uBit.audio.activateMic();

    uBit.messageBus.listen(MICROBIT_ID_BUTTON_A, MICROBIT_BUTTON_EVT_CLICK, onButtonA);
    uBit.messageBus.listen(MICROBIT_ID_BUTTON_B, MICROBIT_BUTTON_EVT_CLICK, onButtonB);
    uBit.messageBus.listen(MICROBIT_ID_BUTTON_AB, MICROBIT_BUTTON_EVT_CLICK, onButtonAB);

    release_fiber();
}