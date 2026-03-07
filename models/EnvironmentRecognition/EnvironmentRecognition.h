//
// Created by simon on 07/03/2026.
//

#ifndef ENVIRONMENTRECOGNITION_H
#define ENVIRONMENTRECOGNITION_H

#define NUM_FEATURES 4

// A sample to predict
typedef struct {
    float features[NUM_FEATURES];
} EnvironmentSample;

// A sample used to train the model (contains the class label)
typedef struct {
    int sampleClass;
    EnvironmentSample sample;
} TrainingSample;

#endif // ENVIRONMENTRECOGNITION_H