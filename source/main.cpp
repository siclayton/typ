#include "MicroBit.h"
#include "arm_math.h"

#define SAMPLE_RATE (11 * 1024)
#define FFT_SIZE 512
#define NUM_MEL 10
#define NUM_FEATURES (NUM_MEL * 2)

#define NUM_SAMPLES 150
#define MAX_DEPTH 5
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
        int predict(SpeechSample sample);
    private:
        int numFeatures{};
        int numClasses{};
        int lenXTrain{};
        int numNodes{};
        TrainingSample* xTrain{};
        TreeNode* nodes{};
        int* indices{};

        void trainModel();
};

DecisionTree::DecisionTree(int numFeatures, int numClasses, int lenXTrain, TrainingSample xTrain[]) {
    this->numFeatures = numFeatures;
    this->numClasses = numClasses;
    this->lenXTrain = lenXTrain;
    this->xTrain = xTrain;

    this->numNodes = 0;
    this->nodes = new TreeNode[MAX_NODES];
    this->indices = new int[lenXTrain];

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

    // Use the CART algorithm to populate the nodes array
    // Loop while there are still nodes left in the queue
    while (start < end) {
        int currentIndex = queue[start++];
        TreeNode &current = nodes[currentIndex];

        // Stopping conditions
        // TODO: Add any other stopping conditions (e.g. min num samples or if node is pure)
        if (current.depth >= MAX_DEPTH) {
            current.isLeaf = true;
            // TODO: Calculate the prediction for this node
            // current.prediction = majority class of samples it considers
            continue;
        }

        // TODO: Calculate the feature and threshold for that node
        // findBestSplit();

        // TODO: Check split usefulness stopping condition
        // If split gain <=0:
        // node.isLeaf = true
        // node.prediction = majority class of samples it considers
        // continue

        // node.feature = best feature found in findBestSplit()
        // node.threshold = best threshold found in findBestSplit()

        int left = numNodes++;
        int right = numNodes++;

        current.left = left;
        current.right = right;

        // midIndex is the index of the indices array where for this nodes split, everything to the left < threshold, everything to the right >= threshold
        // nodes[left] = {current.start, midIndex, -1, -1, -1, -1, current.depth + 1, false, -1};

        queue[end++] = left;
        queue[end++] = right;
    }
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
        ManagedBuffer buf = source.pull();
        uint8_t* bytes = buf.getBytes();

        // Get the data back into 16 bit format
        auto samples = reinterpret_cast<int16_t*>(bytes);
        int numSamples = buf.length() / 2;

        for (int i = 0; i < numSamples; i++) {
            window[windowIndex++] = samples[i]  * (1.0f / 32768.0f); // Normalises the data to floats in the range (-1, 1)

            if (windowIndex == FFT_SIZE) {
                applyfftAndMel();
                windowIndex = 0;
                count++;

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

    // Pack feature vector returned by mel filter bank calculations into a SpeechSample
    SpeechSample sample;
    for (int i = 0; i < NUM_MEL; i++) {
        sample.features[i] = means[i];
    }
    for (int i = NUM_MEL; i < NUM_FEATURES; i++) {
        sample.features[i] = variances[i];
    }

    free(means);
    free(m2s);
    free(variances);

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
    // Need NUM_MEL_FILTERS + 2 as each of the triangles need 3 points (
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
        int prediction = model.predict(speechSample);
        uBit.serial.printf("Prediction %d\r\n", prediction);
        uBit.display.print(prediction);
    }
}
void onButtonB(MicroBitEvent e);
void onButtonAB(MicroBitEvent e);

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