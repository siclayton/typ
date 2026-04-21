#include "KNN.h"

/**
 * The constructor for a KNN model
 * @param num_features the number of features per sample in the training dataset
 * @param num_classes the number of classes present in the training dataset
 * @param k the number of neighbours to consider during prediction
 * @param lenXTrain the number of samples in the training dataset
 * @param xTrain the training dataset
 */
KNN::KNN(int num_features, int num_classes, int k, int lenXTrain, TrainingSample xTrain[]) {
    this->num_features = num_features;
    this->num_classes = num_classes;
    this->k = k;
    this->lenXTrain = lenXTrain;
    this->xTrain = xTrain;
    //Create two arrays to hold the nearest samples and nearest distances (used for predictions)
    this->kNearest = new TrainingSample[k];
    this->nearestDistances = new float[k];

    //Initialise the nearestDistances array to very large values
    //All the values calculated will be smaller than these so they will be placed into the array
    for (int i = 0; i < k; i++) {
        nearestDistances[i] = 1e30f;
    }
}
/**
 * Predict the class of a given sample
 * @param sample the sample to predict the class of
 * @return the models prediction
 */
int KNN::predict(GestureSample sample) {
    calcKNearestNeighbours(sample); //Update the kNearest and nearestDistances arrays
    int prediction = majorityClass(); //Use the modal class of the kNearest samples to predict the class of the sample

    return prediction;
}

/**
 * Calculates the k-nearest neighbours for a given sample
 * @param s the sample
 */
void KNN::calcKNearestNeighbours(GestureSample s) {
    for (int i = 0; i < this->lenXTrain; i++) {
        //Calculate the distance between the sample s and this element in the xTrain array
        float dist = squared_euclidean_distance(s, i);

        //If the kNearest array isn't full or the distance to this element is less than the lowest distance in the nearestDistances array,
        //Add the sample to the kNearest samples and the distance to the nearestDistances array
        if (i < k) {
            nearestDistances[i] = dist;
            kNearest[i] = xTrain[i];
        } else {
            //Sort the arrays, so the lowest distance is at index k-1
            sortKNearestNeighbours();
            if (dist < nearestDistances[k-1]) {
                nearestDistances[k-1] = dist;
                kNearest[k-1] = xTrain[i];
            }
        }
    }
}
/**
 * Calculates the Squared Euclidean distance between the given sample and the sample at the given
 * index in the xTrain array
 * @param sample1 the sample
 * @param index the index of the xTrain array
 * @return the distance between the sample and the sample of the xTrain array
 */
float KNN::squared_euclidean_distance(GestureSample sample1, int index) {
    GestureSample sample2 = xTrain[index].sample;

    //Calculate differences in variables
    float dist = 0;

    for (int i = 0; i < NUM_FEATURES; i++) {
        float featureDiff = sample1.features[i] - sample2.features[i];
        dist += featureDiff * featureDiff;
    }

    return dist;
}
/**
 * Sorts the kNearest and nearestDistances arrays, using bubble sort
 */
void KNN::sortKNearestNeighbours(){
    for (int i = 0; i < k - 1; i++) {
        for (int j = 0; j < k - i - 1; j++) {
            if (nearestDistances[j] > nearestDistances[j + 1]) {
                float tempDist = nearestDistances[j];
                nearestDistances[j] = nearestDistances[j+1];
                nearestDistances[j+1] = tempDist;

                TrainingSample tempSample = kNearest[j];
                kNearest[j] = kNearest[j+1];
                kNearest[j+1] = tempSample;
            }
        }
    }
}
/**
 * Determine the modal class of the kNearest array
 * @return the majority class of the k-nearest neighbours
 */
int KNN::majorityClass() {
    int majorityClass = 0, maxCount = 0;

    for (int i = 0; i < num_classes; i++) {
        int classCount = 0;

        for (int j = 0; j < k; j++) {
            if (kNearest[j].sampleClass == i) {
                classCount++;
            }
        }

        if (classCount > maxCount) {
            maxCount = classCount;
            majorityClass = i;
        }
    }
    return majorityClass;
}
