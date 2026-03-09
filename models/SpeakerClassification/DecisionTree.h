//
// Created by simon on 06/03/2026.
//

#ifndef DECISIONTREE_H
#define DECISIONTREE_H

#include "SpeakerClassification.h"

#define MAX_DEPTH 6
#define MIN_SAMPLES_TO_SPLIT 5
#define MAX_NODES (((MAX_DEPTH + 1) * (MAX_DEPTH + 1)) - 1)

typedef struct {
    int start, end; // Indexes of the indices array (in the DecisionTree class) that represent the part of the dataset passed to this node
    int right, left; // Indexes of the nodes array (in the DecisionTree class) that represent this node's children
    int feature; // Index of the features array (in SpeechSample struct) that this node splits on
    float threshold; // The threshold value that this node splits on
    int depth;
    bool isLeaf;
    int prediction; // The class this node predicts (only for leaf nodes)
} TreeNode;

class DecisionTree {
public:
    DecisionTree() = default;
    DecisionTree(int numFeatures, int numClasses, int lenXTrain, TrainingSample xTrain[]);
    int predict(SpeechSample sample);
private:
    typedef struct {
        float impurity;
        float value;
        int feature;
    } Split;

    int numFeatures{};
    int numClasses{};
    int lenXTrain{};
    int numNodes{};
    TrainingSample* xTrain{};
    TreeNode nodes[MAX_NODES]{};
    int* indices{};

    void trainModel();
    void createChildren(TreeNode &current, int* queue, int* end, int midIndex);
    void printTree();
    bool nodeIsPure(TreeNode node);
    Split findBestSplit(int startIndex, int endIndex);
    float calcGiniFromClassCounts(int* classCounts, int total);
    float calcGiniImpurity(int startIndex, int endIndex, int feature, float threshold);
    int reorderIndices(int startIndex, int endIndex, int feature, float threshold);
    int findMajorityClass(int startIndex, int endIndex);
};

#endif