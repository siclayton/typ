#include "LogisticRegression.h"
#include <cmath>

/**
 * A constructor for a LogisticRegressionModel instance that uses default parameters for the maximum number of iterations, the learning rate and the loss increase threshold
 */
LogisticRegressionModel::LogisticRegressionModel(int numFeatures, int lenXTrain, TrainingSample xTrain[]) :
    LogisticRegressionModel(numFeatures, lenXTrain, 5000, 0.01, 1e-5, xTrain) {
}
/**
 * A constructor for a LogisticRegression instance
 * @param numFeatures the number of features that each sample in the training data has
 * @param lenXTrain the number of samples in the training data
 * @param maxIter the maximum number of iterations that the training loop should be run for
 * @param lr the learning rate of the model
 * @param threshold the amount that the loss needs to decrease buy each iteration for the training
 * loop to continue
 * @param xTrain the trainng data array
 */
LogisticRegressionModel::LogisticRegressionModel(int numFeatures, int lenXTrain, int maxIter, float lr, float threshold, TrainingSample xTrain[]) {
    this->numFeatures = numFeatures;
    this->lenXTrain = lenXTrain;
    this->maxIter = maxIter;
    this->lr = lr;
    this->threshold = threshold;
    this->xTrain = xTrain;

    //Initialise both the weights and bias to 0
    for (int i = 0; i < NUM_FEATURES; i++){
        this->weights[i] = 0;
    }
    this->bias = 0;
    this->epsilon = 1e-7f; //Used to clamp predictions to avoid calculating log(0) or log(1)

    trainModel();
}
/**
 * Runs the training loop for the model
 */
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
/**
 * Computes the result of the sigmoid function on the given value
 * @param x the given value
 * @return the results if the sigmoid function
 */
double LogisticRegressionModel::sigmoid(double x) {
    return 1 / (1 + exp(-x));
}
/**
 * Computes the dot product of the given sample and the model's weights
 * @param s the given sample
 * @return the result of the dot product calculation
 */
float LogisticRegressionModel::dotProduct(MicrophoneSample s) {
    float logit = 0;

    //Compute the dot product of the sample features and the weights
    for (int i = 0; i < numFeatures; i++) {
        logit += s.features[i] * weights[i];
    }

    return logit;
}
/**
 * Calculates the dot product and then passes the output of that to the sigmoid function
 * This is the prediction for a sample without thresholding it
 * @param sample the sample to predict
 * @return the prediction
 */
float LogisticRegressionModel::predict(MicrophoneSample sample) {
    float logit = dotProduct(sample) + bias;

    float prediction = sigmoid(logit);

    return prediction;
}
/**
 * Calculates the cross entropy loss of the predicted classes and the actual classes of the sample
 * @param preds the predicted classes of the samples
 * @return the loss
 */
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
/**
 * Calculate how much the bias needs to be updated by
 * @param preds the predicted classes of the samples
 * @return the amount the bias should be updated by
 */
float LogisticRegressionModel::calcBiasUpdate(float* preds) {
    float updateAmount = 0;

    for (int i = 0; i < lenXTrain; i++) {
        updateAmount += preds[i] - static_cast<float>(xTrain[i].sampleClass);
    }

    return updateAmount / static_cast<float>(lenXTrain);
}
/**
 * Calculate how much the weights needs to be updated by
 * @param preds the predicted classes of the samples
 * @return the amount each weight should be updated by
 */
float *LogisticRegressionModel::calcWeightsUpdates(float* preds) {
    auto* updateAmounts = new float[numFeatures]();

    for (int i = 0; i < lenXTrain; i++) {
        TrainingSample curr = xTrain[i];
        float error = preds[i] - static_cast<float>(curr.sampleClass);

        for (int j = 0; j < numFeatures; j++) {
            updateAmounts[j] += error * curr.sample.features[j];
        }
    }

    for (int i = 0; i < numFeatures; i++) {
        updateAmounts[i] /= static_cast<float>(lenXTrain);
    }

    return updateAmounts;
}

/**
* Return a prediction of the class of the given sample
 * @param s the sample to predict the class of
 * @return 1 if the model predicts speech, 0 if the model predicts no speech
 */
int LogisticRegressionModel::predictClass(MicrophoneSample s) {
    float sigmoidOutput = predict(s);

    return sigmoidOutput > 0.50 ? 1 : 0;
}