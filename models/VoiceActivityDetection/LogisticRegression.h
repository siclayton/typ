#ifndef LOGISTICREGRESSION_H
#define LOGISTICREGRESSION_H

#include "VoiceActivityDetection.h"

class LogisticRegressionModel {
public:
    LogisticRegressionModel(int numFeatures, int lenXTrain, TrainingSample xTrain[]);
    LogisticRegressionModel(int numFeatures, int lenXTrain, int maxIter, float lr, float threshold, TrainingSample[]);
    int predictClass(MicrophoneSample s);
private:
    int numFeatures{};
    int lenXTrain{};
    int maxIter{};
    float lr{};
    float threshold{}; //The minimum amount the loss must decrease by to continue training
    TrainingSample* xTrain{};
    //Values to train using gradient descent
    float weights[NUM_FEATURES];
    float bias;
    float epsilon; //Used to clamp predictions to ensure the log of 0 or 1 is never computed

    void trainModel();
    float predict(MicrophoneSample);
    double sigmoid(double);
    float dotProduct(MicrophoneSample);
    float crossEntropyLoss(float[]);
    float calcBiasUpdate(float[]);
    float* calcWeightsUpdates(float[]);
};

#endif