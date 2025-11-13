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

    int lastValueWasBelowMid = 0;

    while (system_timer_current_time() - start < 1000) {
        int value = uBit.io.microphone.getAnalogValue();
        count++;

        //Update max value
        if (value > max) max = value;

        /*
            The microphone outputs values from 0 to 1024, so to detect how much the noise oscillates
            I am detecting how often it goes above and below 512, as this is the mid-point, but I am
            having a threshold of 10, so the value must be either 10 or more over or under 512 to count
        */
        if (lastValueWasBelowMid == 0 && value < 522) {
            zeroCrossings++;
            lastValueWasBelowMid = 1;
        } else if (lastValueWasBelowMid == 1 && value > 532) {
            zeroCrossings++;
            lastValueWasBelowMid = 0;
        }

        //Calculate mean and variance for each axis
        float diff = (float) value - mean;
        mean += diff / count;
        m2 += diff * (value - mean);

        uBit.sleep(2);
    }

    float variance = m2 / (count - 1);

    MicrophoneSample sample = {sampleClass, mean, variance, max, zeroCrossings};

    uBit.serial.printf("%d, %d, %d, %d, %d\r\n",
        sampleClass, (int) (mean * 1000), (int) (variance * 1000), max, zeroCrossings
    );

    return sample;
}

void onButtonA(MicroBitEvent e) {
    MicrophoneSample sample = takeSample(currentClass);
    samples[currentSample++] = sample;
}

void onButtonB(MicroBitEvent e) {
    currentClass++;
}

int main() {
    uBit.init();

    uBit.audio.enable();
    uBit.audio.activateMic();

    uBit.messageBus.listen(MICROBIT_ID_BUTTON_A, MICROBIT_BUTTON_EVT_CLICK, onButtonA);
    uBit.messageBus.listen(MICROBIT_ID_BUTTON_B, MICROBIT_BUTTON_EVT_CLICK, onButtonB);

    release_fiber();
}