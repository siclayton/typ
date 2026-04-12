#ifndef SPEAKERCLASSIFICATION_H
#define SPEAKERCLASSIFICATION_H

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

/**
 * A struct which represents a labelled SpeechSample
 */
typedef struct {
    int sampleClass;
    SpeechSample sample;
} TrainingSample;

#endif