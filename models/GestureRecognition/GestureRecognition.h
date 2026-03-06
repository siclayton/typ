//
// Created by simon on 06/03/2026.
//

#ifndef CODAL_GESTURERECOGNITION_H
#define CODAL_GESTURERECOGNITION_H

#define NUM_SAMPLES 100
#define FEATURE_COUNT 24

/**
 * A struct which represents a gesture
 * Contains data from the accelerometer and magnetometer
 */
typedef struct {
    float features[FEATURE_COUNT];
} GestureSample;

/**
 * A struct which represents a labelled GestureSample
 */
typedef struct {
    int sampleClass;
    GestureSample sample;
} TrainingSample;

#endif // CODAL_GESTURERECOGNITION_H
