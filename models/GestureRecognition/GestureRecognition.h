//
// Created by simon on 06/03/2026.
//

#ifndef GESTURERECOGNITION_H
#define GESTURERECOGNITION_H

#define NUM_FEATURES 24

/**
 * A struct which represents a gesture
 * Contains data from the accelerometer and magnetometer
 */
typedef struct {
    float features[NUM_FEATURES];
} GestureSample;

/**
 * A struct which represents a labelled GestureSample
 */
typedef struct {
    int sampleClass;
    GestureSample sample;
} TrainingSample;

#endif