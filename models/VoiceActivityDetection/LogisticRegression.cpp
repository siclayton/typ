//
// Created by simon on 06/03/2026.
//

#include "LogisticRegression.h"
#include <cmath>

// Constructor that uses default parameters for the maximum number of iterations, the learning rate and the loss increase threshold
LogisticRegressionModel::LogisticRegressionModel(int numFeatures, int lenXTrain, TrainingSample xTrain[]) :
    LogisticRegressionModel(numFeatures, lenXTrain, 5000, 0.01, 1e-5, xTrain) {}

LogisticRegressionModel::LogisticRegressionModel(int numFeatures, int lenXTrain, int maxIter, float lr, float threshold, TrainingSample xTrain[]) {
    this->numFeatures = numFeatures;
    this->lenXTrain = lenXTrain;
    this->maxIter = maxIter;
    this->lr = lr;
    this->threshold = threshold;
    this->xTrain = xTrain;

    //Initialise both the weights and bias to 0
    this->weights = new float[numFeatures];
    for (int i = 0; i < numFeatures; i++){
        this->weights[i] = 0;
    }
    this->bias = 0;
    this->epsilon = 1e-7f; //Used to clamp predictions to avoid calculating log(0) or log(1)

    trainModel();
}

void LogisticRegressionModel::trainModel() {
    float previousLoss = 1000; //Set to a large value (any actual loss will be smaller than this)
    auto* predictions = new float[lenXTrain];

    //Training loop
    for (int i = 0; i < maxIter; i++) {
        for (int j = 0; j < lenXTrain; j++) {
            float pred = predict(xTrain[j].sample);
            predictions[j] = pred;
        }

        float loss = crossEntropyLoss(predictions);

        //Calculate the updates needed for the weights and bias
        float biasChange = calcBiasUpdate(predictions);
        float* weightsChanges = calcWeightsUpdates(predictions);

        //Update the weights and bias
        bias -= lr * biasChange;
        for (int j = 0; j < numFeatures; j++) {
            weights[j] -= lr * weightsChanges[j];
        }

        delete[] weightsChanges;

        //Stop iterating if the loss hasn't improved by more than the threshold value
        if (previousLoss - loss < threshold)
            break;

        previousLoss = loss;
    }

    delete[] predictions;
}

double LogisticRegressionModel::sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

float LogisticRegressionModel::dotProduct(MicrophoneSample s) {
    float logit = 0;

    //Compute the dot product of the sample features and the weights
    for (int i = 0; i < numFeatures; i++) {
        logit += s.features[i] * weights[i];
    }

    return logit;
}

float LogisticRegressionModel::predict(MicrophoneSample sample) {
    float logit = dotProduct(sample) + bias;

    float prediction = sigmoid(logit);

    return prediction;
}

float LogisticRegressionModel::crossEntropyLoss(float* preds) {
    double loss = 0;

    for (int i = 0; i < lenXTrain; i++) {
        float p = preds[i];

        //Clamp p into the range: epsilon >= p >= 1 - epsilon
        if (p < epsilon)
            p = epsilon;
        else if (p > 1 - epsilon)
            p = 1 - epsilon;

        //Compute the cross-entropy loss for this sample and add it to the total
        loss += static_cast<float>(xTrain[i].sampleClass) * log(p) + static_cast<float>(1 - xTrain[i].sampleClass) * log(1 - p);
    }

    return -loss / static_cast<float>(lenXTrain);
}

float LogisticRegressionModel::calcBiasUpdate(float* preds) {
    float updateAmount = 0;

    for (int i = 0; i < lenXTrain; i++) {
        updateAmount += preds[i] - static_cast<float>(xTrain[i].sampleClass);
    }

    return updateAmount / static_cast<float>(lenXTrain);
}

float *LogisticRegressionModel::calcWeightsUpdates(float* preds) {
    auto* updateAmounts = new float[numFeatures]();

    for (int i = 0; i < lenXTrain; i++) {
        TrainingSample curr = xTrain[i];
        float error = preds[i] - static_cast<float>(curr.sampleClass);

        for (int j = 0; i < numFeatures; j++) {
            updateAmounts[j] += error * curr.sample.features[j];
        }
    }

    for (int i = 0; i < numFeatures; i++) {
        updateAmounts[i] /= static_cast<float>(lenXTrain);
    }

    return updateAmounts;
}

//Return a prediction of the class of the given sample
//  1 means the model predicts speech, 0 means the model predicts no speech
int LogisticRegressionModel::predictClass(MicrophoneSample s) {
    float sigmoidOutput = predict(s);

    return sigmoidOutput > 0.50 ? 1 : 0;
}