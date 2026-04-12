//
// Created by simon on 07/03/2026.
//

#ifndef NAIVEBAYES_H
#define NAIVEBAYES_H

#include "EnvironmentRecognition.h"
#include <cmath>

#define NUM_BINS 4

class NaiveBayesModel {
public:
    NaiveBayesModel(int numFeatures, int numClasses, int lenXTrain, TrainingSample* xTrain);
    int predict(EnvironmentSample sample);

private:
    int numFeatures;
    int numClasses;
    int lenXTrain;
    TrainingSample* xTrain;
    int* classCounts;
    float* classProbabilities;
    float* conditionalProbabilities;

    //Set mins initially to a large value and set maxes initially to 0
    float* minFeatureValues;
    float* maxFeatureValues;

    void trainModel();
    void calcClassProbabilities();
    void calcConditionalProbabilities();
    int getCProbIndex(int c, int feature, int bin);
    int getBin(int feature, float value);
};

#endif