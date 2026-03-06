//
// Created by simon on 06/03/2026.
//
#ifndef CODAL_KNN_H
#define CODAL_KNN_H

#include "GestureRecognition.h"

/**
 * A class which defines the functionality for a KNN model
 */
class KNN {
public:
    KNN() {}
    KNN(int, int, int, int, TrainingSample[]);
    int predict(GestureSample);

private:
    int num_features;
    int num_classes;
    int k;
    int lenXTrain;
    TrainingSample* xTrain;
    TrainingSample* kNearest;

    float *nearestDistances;
    void calcKNearestNeighbours(GestureSample);
    float squared_euclidean_distance(GestureSample, int);
    void sortKNearestNeighbours();
    int majorityClass();
};

#endif // CODAL_KNN_H
