#ifndef VOICEACTIVITYDETECTION_H
#define VOICEACTIVITYDETECTION_H

#define NUM_FEATURES 3

/**
 * Struct containing the features of a microphone sample
 */
typedef struct {
    float features[NUM_FEATURES];
} MicrophoneSample;

/**
 * Struct representing a labelled MicrophoneSample
 */
typedef struct {
    int sampleClass;

    MicrophoneSample sample;
}TrainingSample;

#endif