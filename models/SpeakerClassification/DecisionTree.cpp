//
// Created by simon on 06/03/2026.
//

#include "MicroBit.h"
#include "DecisionTree.h"

DecisionTree::DecisionTree(int numFeatures, int numClasses, int lenXTrain, TrainingSample xTrain[]) {
    this->numFeatures = numFeatures;
    this->numClasses = numClasses;
    this->lenXTrain = lenXTrain;
    this->xTrain = xTrain;

    this->numNodes = 0;
    this->indices = new int[lenXTrain];
    for (int i = 0; i < lenXTrain; i++) {
        indices[i] = i;
    }

    trainModel();
}
void DecisionTree::trainModel() {
    // A queue to hold the nodes we still need to calculate the feature and threshold for
    // Stores indexes of the nodes array
    int queue[MAX_NODES];
    int start = 0, end = 0;

    // Add the root node to the nodes array and the queue
    nodes[numNodes++] = {0, lenXTrain - 1, -1, -1, -1, -1, 0, false, -1};
    queue[end++] = 0;

    // Use the CART algorithm to create the tree
    // Loop while there are still nodes left in the queue
    while (start < end) {
        int currentIndex = queue[start++];
        TreeNode &current = nodes[currentIndex];

        // Stopping conditions
        if (current.depth >= MAX_DEPTH || current.end - current.start < MIN_SAMPLES_TO_SPLIT || nodeIsPure(current)) {
            current.isLeaf = true;
            current.prediction = findMajorityClass(current.start, current.end);
            continue;
        }

        Split bestSplit = findBestSplit(current.start, current.end);

        // Reorder indices array and find the index at which the values are split
        int midIndex = reorderIndices(current.start, current.end, bestSplit.feature, bestSplit.value);

        // Store the feature, value pair for the split at this node
        current.feature = bestSplit.feature;
        current.threshold = bestSplit.value;

        // Create the children for this node and add them to the queue
        int left = numNodes++;
        int right = numNodes++;
        current.left = left;
        current.right = right;
        nodes[left] = {current.start, midIndex, -1, -1, -1, -1, current.depth + 1, false, -1};
        nodes[right] = {midIndex + 1, current.end, -1, -1, -1, -1, current.depth + 1, false, -1};

        queue[end++] = left;
        queue[end++] = right;
    }

    for (int i = 0; i < numNodes; i++) {
        TreeNode &n = nodes[i];
        if (n.isLeaf) {
            DMESG("Node %d: LEAF prediction=%d (start=%d end=%d)", i, n.prediction, n.start, n.end);
        } else {
            DMESG("Node %d: SPLIT feature=%d threshold=%d left=%d right=%d", i, n.feature, (int)(n.threshold * 1000), n.left, n.right);
        }
    }
}
/**
 * Checks whether the part of the dataset that a given node considers is made up of only one class
 * @param node the node to check
 * @return true if the node is pure, false if not
 */
bool DecisionTree::nodeIsPure(TreeNode node) {
    int sampleClass = xTrain[indices[node.start]].sampleClass;

    for (int i = node.start + 1; i <= node.end; i++) {
        if (xTrain[indices[i]].sampleClass != sampleClass) {
            return false;
        }
    }

    return true;
}
/**
 * Find the majority class for a given part of the dataset
 * @param startIndex the first index in the indices array to consider
 * @param endIndex the last index in the indices array to consider
 * @return the majority class
 */
int DecisionTree::findMajorityClass(int startIndex, int endIndex) {
    int highestClassCount = 0;
    int majorityClass = 0;

    for (int i = 0; i < numClasses; i++) {
        int count = 0;
        for (int j = startIndex; j <= endIndex; j++) {
            if (xTrain[indices[j]].sampleClass == i) {
                count++;
            }
        }

        if (count > highestClassCount) {
            highestClassCount = count;
            majorityClass = i;
        }
    }

    return majorityClass;
}
/**
 * Find the best feature, value pair to split the data on to reduce the Gini impurity
 * @param startIndex the first index in the indices array to consider
 * @param endIndex the last index of the indices array to consider
 * @return a Split object containing {impurity, value, feature} of the best split found
 */
DecisionTree::Split DecisionTree::findBestSplit(int startIndex, int endIndex) {
    Split best = {1e7, 0, 0};

    // Brute-force each feature,value pair to find the one with the lowest impurity
    for (int i = 0; i < numFeatures; i++) {
        for (int j = startIndex; j < endIndex; j++) {
            float value = xTrain[indices[j]].sample.features[i];

            float splitGini = calcGiniImpurity(startIndex, endIndex, i, value);

            if (splitGini < best.impurity) {
                best = {splitGini, value, i};
            }
        }
    }

    return best;
}
/**
 * Calculate gini impurity for one side of a split, based on given class counts
 * @param classCounts an array of the counts for each class, of size int[numClasses]
 * @param total the total number of samples on this side of the split
 * @return the gini impurity of this side
 */
float DecisionTree::calcGiniFromClassCounts(int* classCounts, int total) {
    if (total <= 0) {
        return 0;
    }

    float sum = 0;
    for (int i = 0; i < numClasses; i++) {
        float classProb = static_cast<float>(classCounts[i]) / total;
        sum += classProb * classProb;
    }

    return 1 - sum;
}
/**
 * Calculate gini impurity for a given split
 * @param startIndex the first index of the indices array to consider
 * @param endIndex the last index of the indices array to consider
 * @param feature the feature to split on
 * @param threshold the value to split the given feature on
 * @return the weighted gini impurity of the split
 */
float DecisionTree::calcGiniImpurity(int startIndex, int endIndex, int feature, float threshold) {
    int leftClassCounts[numClasses];
    int rightClassCounts[numClasses];
    int numLeft = 0;
    int numRight = 0;

    for (int i = 0; i < numClasses; i++) {
        leftClassCounts[i] = rightClassCounts[i] = 0;
    }

    // Count the number of samples in each class to the left and right of the threshold
    for (int i = startIndex; i <= endIndex; i++) {
        int label = xTrain[indices[i]].sampleClass;
        float featureValue = xTrain[indices[i]].sample.features[feature];

        if (featureValue < threshold) {
            leftClassCounts[label]++;
            numLeft++;
        } else {
            rightClassCounts[label]++;
            numRight++;
        }
    }

    float giniLeft = calcGiniFromClassCounts(leftClassCounts, numLeft);
    float giniRight = calcGiniFromClassCounts(rightClassCounts, numRight);

    float weightedGini = (numLeft * giniLeft + numRight * giniRight) / (numLeft + numRight);
    return weightedGini;
}
/**
 * Reorders a part of the indices array so that all the indices whose corresponding sample has a feature value < threshold are on the left,
 * and samples with a feature value > threshold are on the right
 *
 * @param startIndex the first index to consider
 * @param endIndex the last index to consider
 * @param feature the feature whose value we are interested in (as an index of the features array in a SpeechSample struct)
 * @param threshold the threshold value to compare feature values to
 * @return the index of the last index in the indices array whose corresponding sample in xTrain has a feature value < threshold
 */
int DecisionTree::reorderIndices(int startIndex, int endIndex, int feature, float threshold) {
    // Index of the first index in the indices array, whose corresponding sample in xTrain has a feature value >= threshold
    int greaterThan = startIndex;

    for (int i = startIndex; i <= endIndex; i ++) {
        TrainingSample &trainingSample = xTrain[indices[i]];

        if (trainingSample.sample.features[feature] < threshold) {
            int temp = indices[i];
            indices[i] = indices[greaterThan];
            indices[greaterThan++] = temp;
        }
    }

    // Return the index of the last item < threshold
    return greaterThan - 1;
}
/**
 * Predict the class of the given sample
 *
 * @param sample the sample to predict the class of
 * @return the predicted class
 */
int DecisionTree::predict(SpeechSample sample) {
    TreeNode current = nodes[0]; // Start traversal at the root node

    while (!current.isLeaf) {
        if (sample.features[current.feature] < current.threshold) {
            current = nodes[current.left];
        } else {
            current = nodes[current.right];
        }
    }

    return current.prediction;
}