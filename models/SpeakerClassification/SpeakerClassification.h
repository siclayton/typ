//
// Created by simon on 06/03/2026.
//

#ifndef CODAL_SPEAKERCLASSIFICATION_H
#define CODAL_SPEAKERCLASSIFICATION_H

#define SAMPLE_RATE (11 * 1024)
#define FFT_SIZE 256
#define NUM_MEL 10
#define NUM_FEATURES (NUM_MEL * 2)

/**
 * A struct which represents a 1-second sample taken from the microphone
 */
typedef struct {
    float features[NUM_FEATURES];
} SpeechSample;

typedef struct {
    int sampleClass;
    SpeechSample sample;
} TrainingSample;

#endif // CODAL_SPEAKERCLASSIFICATION_H