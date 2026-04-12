#ifndef KNN_H
#define KNN_H

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

#endif