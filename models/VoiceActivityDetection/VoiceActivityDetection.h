//
// Created by simon on 06/03/2026.
//

#ifndef CODAL_VOICEACTIVITYDETECTION_H
#define CODAL_VOICEACTIVITYDETECTION_H

#define NUM_FEATURES 3

//Data type containing the features of a microphone sample
typedef struct {
    float features[NUM_FEATURES];
} MicrophoneSample;

//Data type representing a labelled MicrophoneSample
//Used for training the model
typedef struct {
    int sampleClass;

    MicrophoneSample sample;
}TrainingSample;

#endif // CODAL_VOICEACTIVITYDETECTION_H
