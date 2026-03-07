//
// Created by simon on 07/03/2026.
//

#include "NaiveBayes.h"

NaiveBayesModel::NaiveBayesModel(int numFeatures, int numClasses, int lenXTrain, TrainingSample* xTrain) {
    this->numFeatures = numFeatures;
    this->numClasses = numClasses;
    this->lenXTrain = lenXTrain;
    this->xTrain = xTrain;

    this->classCounts = new int[numClasses]();
    this->classProbabilities = new float[numClasses];
    this->conditionalProbabilities = new float[numClasses * numFeatures * NUM_BINS]();
    this->minFeatureValues = new float[numFeatures];
    this->maxFeatureValues = new float[numFeatures];

    // Initialise the max and min values to the feature values from the first training sample
    for (int i = 0; i < numFeatures; i++) {
        float value = xTrain[0].sample.features[i];
        maxFeatureValues[i] = value;
        minFeatureValues[i] = value;
    }

    trainModel();
}

void NaiveBayesModel::trainModel() {
    for (int i = 0; i < lenXTrain; i++) {
        //Count the number of occurrences of each class in the training data
        classCounts[xTrain[i].sampleClass]++;

        //Find min and max values for each feature (used to determine the bin sizes)
        for (int j = 0; j < numFeatures; j++) {
            float value = xTrain[i].sample.features[j];

            if (value < minFeatureValues[j]) minFeatureValues[j] = value;
            if (value > maxFeatureValues[j]) maxFeatureValues[j] = value;
        }
    }

    calcClassProbabilities();
    calcConditionalProbabilities();

    for (int i = 0; i < numClasses * numFeatures * NUM_BINS; i++) {
        uBit.serial.printf("\r\n");
        uBit.serial.printf("%d, ", static_cast<int>(conditionalProbabilities[i] * 1000));
        if ((i + 1) % 4 == 0) uBit.serial.printf("\r\n");
    }
}

void NaiveBayesModel::calcClassProbabilities() {
    for (int i = 0; i < numClasses; i++) {
        classProbabilities[i] = static_cast<float>(classCounts[i]) / static_cast<float>(lenXTrain);
    }
}

void NaiveBayesModel::calcConditionalProbabilities() {
    //Count how many occurrences of each feature "value" there are
    //The values have been split into 4 bins
    for (int i = 0; i < lenXTrain; i++) {
        for (int j = 0; j < numFeatures; j++) {
            TrainingSample sample = xTrain[i];
            float value = sample.sample.features[j];

            int bin = getBin(j, value);
            conditionalProbabilities[getCProbIndex(sample.sampleClass, j, bin)]++;
        }
    }

    //Convert to probabilities
    for (int i = 0; i < numClasses; i ++) {
        for (int j = 0; j < numFeatures; j++) {
            for (int k = 0; k < NUM_BINS; k++)
            {
                int index = getCProbIndex(i,j,k);
                conditionalProbabilities[index] /= static_cast<float>(classCounts[i]);

                //If the combination of class, feature and bin didn't appear in the dataset,
                //set the probability to a very low number (so the combination isn't completely ignored during predictions)
                if (conditionalProbabilities[index] == 0) conditionalProbabilities[index] = 1e-7;
            }
        }
    }
}

// Get the index of the array for the given class, feature and bin for the conditionalProbabilites array
// This is needed as the conditionalProbabilties array is 1D
int NaiveBayesModel::getCProbIndex(int c, int feature, int bin) {
    return c * numFeatures * NUM_BINS + feature * NUM_BINS + bin;
}

// Return the bin that the given value belongs in for the given feature
int NaiveBayesModel::getBin(int feature, float value) {
    float range = maxFeatureValues[feature] - minFeatureValues[feature];

    //If the range for that feature is too small, assign all values to bin 0
    if (range < 1e-6) return 0;

    float binSize = range / NUM_BINS;
    int bin = static_cast<int>((value - minFeatureValues[feature]) / binSize);

    //Ensure bin isn't out of range
    if (bin >= NUM_BINS) bin = NUM_BINS - 1;
    if (bin < 0) bin = 0;

    return bin;
}

int NaiveBayesModel::predict(EnvironmentSample sample) {
    int prediction = 0;
    float highestProbability = -1e30f; //Set highestProbability to a low negative number

    for (int i = 0; i < numClasses; i++) {
        // Use log probabilities to prevent underflow
        float classProbability = log(classProbabilities[i]);

        for (int j = 0; j < numFeatures; j++) {
            float value = sample.features[j];
            int bin = getBin(j, value);
            int cProbIndex = getCProbIndex(i, j, bin);

            // Probabilities are calculated using Bayes theorem
            classProbability += log(conditionalProbabilities[cProbIndex]);
        }

        // If this is the highest probability seen so far, update the prediction
        if (classProbability > highestProbability) {
            highestProbability = classProbability;
            prediction = i;
        }
    }

    return prediction;
}