#include "MicroBit.h"
#define ARM_MATH_CM4
#include "arm_math.h"

#include <cstdio>

#define SAMPLE_RATE (11 * 1024)
#define FFT_SIZE 512
#define NUM_MEL 10
#define NUM_FEATURES (NUM_MEL * 2)

#define NUM_SAMPLES 150
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

typedef struct {
    float features[NUM_FEATURES];
} SpeechSample;

typedef struct {
    int sampleClass;
    SpeechSample sample;
} TrainingSample;

class DecisionTree {
    public:
        DecisionTree() = default;
        DecisionTree(int numFeatures, int numClasses, int lenXTrain, TrainingSample xTrain[]);
        //int predict(SpeechSample sample);
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
        Split findBestSplit(int startIndex, int endIndex);
        float calcGiniFromClassCounts(int* classCounts, int total);
        float calcGiniImpurity(int startIndex, int endIndex, int feature, float threshold);
        int reorderIndices(int startIndex, int endIndex, int feature, float threshold);
        int findMajorityClass(int startIndex, int endIndex);
};

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

        // Stopping conditions to prevent overfitting
        if (current.depth >= MAX_DEPTH || current.end - current.start < MIN_SAMPLES_TO_SPLIT) {
            current.isLeaf = true;
            current.prediction = findMajorityClass(current.start, current.end);
            continue;
        }

        Split bestSplit = findBestSplit(current.start, current.end);

        // Stop splitting as this node is pure (all samples in the data it considers are the same class)
        if (bestSplit.impurity <= 0) {
            current.isLeaf = true;
            current.prediction = findMajorityClass(current.start, current.end);
            continue;
        }

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
    Split best = {0, 0, 0};

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
    if (total < 0) {
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

// Global variables
MicroBit uBit;
arm_rfft_fast_instance_f32 fftInstance;
DataSource& source = *uBit.audio.processor;
DecisionTree model;

int currentClass = 0; //The ID for the class that the user is currently providing samples of
int currentSample = 0; //The position in the samples array to add the next sample to
TrainingSample samples[NUM_SAMPLES];
bool training = true;

int melBins[NUM_MEL + 2];
float melWeights[NUM_MEL][FFT_SIZE / 2];
float window[FFT_SIZE];
float fftOutput[FFT_SIZE];
float magnitudeSpectrum[FFT_SIZE / 2];
float melOutput[NUM_MEL];

void applyMelFilters(float* fft, float* mel) {
    // Loop over the filters
    for (int i = 0; i < NUM_MEL; i++) {
        float sum = 0;

        // Apply filter
        for (int j = melBins[i]; j < melBins[i + 2]; j++) {
            sum += fft[j] * melWeights[i][j];
        }
        mel[i] = logf(sum);
    }
}

void applyfftAndMel() {
    // Calculate FFT and magnitude spectrum
    arm_rfft_fast_f32(&fftInstance, window, fftOutput, 0);
    arm_cmplx_mag_f32(fftOutput, magnitudeSpectrum, FFT_SIZE / 2);

    // Apply Mel Filterbank to the magnitude spectrum
    applyMelFilters(magnitudeSpectrum, melOutput);
}

SpeechSample takeSample() {
    int windowIndex = 0;
    int count = 0;
    uint64_t start = system_timer_current_time();

    auto means = static_cast<float *>(calloc(NUM_MEL, sizeof(float)));
    auto m2s = static_cast<float *>(calloc(NUM_MEL, sizeof(float)));

    while (system_timer_current_time() - start < 1000) {
        // Pull a sample from the StreamNormaliser for the mic
        ManagedBuffer buf = source.pull();
        uint8_t* bytes = buf.getBytes();

        // Get the data back into 16 bit format
        auto samples = reinterpret_cast<int16_t*>(bytes);
        int numSamples = buf.length() / 2;

        // Store samples in a window until we have enough to perform a fft
        for (int i = 0; i < numSamples; i++) {
            window[windowIndex++] = samples[i]  * (1.0f / 32768.0f); // Normalises the data to floats in the range (-1, 1)

            if (windowIndex == FFT_SIZE) {
                applyfftAndMel();
                windowIndex = 0;
                count++;

                // Calculate mean and m2 (used in variance calculation) to allow aggregation of the melOutput for different windows
                for (int j = 0; j < NUM_MEL; j++) {
                    float diff = melOutput[j] - means[j];

                    means[j] += diff / count;
                    m2s[j] += diff * (melOutput[j] - means[j]);
                }
            }
        }
    }

    count--;
    auto variances = static_cast<float *>(calloc(NUM_MEL, sizeof(float)));
    for (int i = 0; i < NUM_MEL; i++) {
        variances[i] = m2s[i] / count;
    }

    // Pack aggregated melOutputs into a SpeechSample
    SpeechSample sample;
    for (int i = 0; i < NUM_MEL; i++) {
        sample.features[i] = means[i];
    }
    for (int i = 0; i < NUM_MEL; i++) {
        sample.features[NUM_MEL + i] = variances[i];
    }

    free(means);
    free(m2s);
    free(variances);

    //Print the sample for debugging
    //Build the line to print in a buffer and print once (to put everything on one line in the output)
    char buf[128];
    int offset = 0;

    for (int i = 0; i < NUM_FEATURES; i++) {
        int value = static_cast<int>(sample.features[i] * 1000);
        offset += snprintf(
            buf + offset,
            sizeof(buf) - offset,
            "%d%s",
            value,
            i < NUM_FEATURES - 1 ? "," : ""
        );
    }

    DMESG("%s", buf);

    return sample;

    return sample;
}

float hzToMel(float hz) {
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

float melToHz(float mel) {
    return 700.0f * (powf(10.0f, mel / 2959.0f) - 1.0f);
}

void computeMelFilterbank() {
    float low = hzToMel(0);
    float high = hzToMel(SAMPLE_RATE / 2);
    float step = (high - low) / NUM_MEL + 1;

    // Calculate what mel values go into each fft bin
    // Need NUM_MEL_FILTERS + 2 as each of the triangles need 3 points
    for (int i = 0; i < NUM_MEL + 2; i++) {
        float hz = melToHz(low + i * step);

        melBins[i] = static_cast<int>((FFT_SIZE + 1) * hz / SAMPLE_RATE);
    }

    // Build mel triangle
    for (int i = 0; i < NUM_MEL; i++) {
        int left = melBins[i];
        int centre = melBins[i + 1];
        int right = melBins[i + 2];

        // Rising slope
        for (int j = left; j < centre; j++) {
            melWeights[i][j] = static_cast<float>((j - left) / (centre - left));
        }
        // Falling slope
        for (int j = centre; j < right; j++) {
            melWeights[i][j] = static_cast<float>((right - j) / (right - centre));
        }
    }
}

void onButtonA(MicroBitEvent e){
    SpeechSample speechSample = takeSample();

    if (training) {
        TrainingSample sample = {currentClass, speechSample};
        samples[currentSample++] = sample;
    } else {
        //int prediction = model.predict(speechSample);
        // uBit.serial.printf("Prediction %d\r\n", prediction);
        // uBit.display.print(prediction);
    }
}
void onButtonB(MicroBitEvent e) {
    currentClass++;
}
void onButtonAB(MicroBitEvent e) {
    model = DecisionTree(NUM_FEATURES, currentClass + 1, currentSample, samples);
    uBit.serial.printf("Model trained\r\n");
    training = false;
}

int main() {
    uBit.init();
    uBit.audio.enable();

    // Initialise the fft instance
    arm_rfft_fast_init_f32(&fftInstance, FFT_SIZE);

    // Compute Mel filterbank
    computeMelFilterbank();

    uBit.messageBus.listen(MICROBIT_ID_BUTTON_A, MICROBIT_BUTTON_EVT_CLICK, onButtonA);
    uBit.messageBus.listen(MICROBIT_ID_BUTTON_B, MICROBIT_BUTTON_EVT_CLICK, onButtonB);
    uBit.messageBus.listen(MICROBIT_ID_BUTTON_AB, MICROBIT_BUTTON_EVT_CLICK, onButtonAB);

    release_fiber();
}