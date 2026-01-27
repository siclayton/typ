#include "MicroBit.h"

#include <vector>

#define NUM_SAMPLES 20
#define NUM_BINS 4
#define TRUE 1
#define FALSE 0

MicroBit uBit;
int currentClass = 0; //The ID for the class that the user is currently providing samples of
int currentSample = 0; //The position in the samples array to add the next sample to
int training = TRUE;

typedef struct {
    int sampleClass;

    float meanTemp;
    float meanLightLevel;

    float tempVariance;
    float lightLevelVariance;

    float getFeature(int index) {
        switch (index) {
            case 0: return meanTemp;
            case 1: return meanLightLevel;
            case 2: return tempVariance;
            case 3: return lightLevelVariance;
            default: return 0;
        }
    }

} EnvironmentSample;

class NaiveBayesModel {
    public:
        NaiveBayesModel(int numFeatures, int numClasses, int lenXTrain, EnvironmentSample* xTrain);
        int predict(EnvironmentSample sample);

    private:
        int numFeatures;
        int numClasses;
        int lenXTrain;
        EnvironmentSample* xTrain;
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

NaiveBayesModel::NaiveBayesModel(int numFeatures, int numClasses, int lenXTrain, EnvironmentSample* xTrain) {
    this->numFeatures = numFeatures;
    this->numClasses = numClasses;
    this->lenXTrain = lenXTrain;
    this->xTrain = xTrain;

    this->classCounts = new int[numClasses]();
    this->classProbabilities = new float[numClasses];
    this->conditionalProbabilities = new float[numClasses * numFeatures * NUM_BINS]();
    this->minFeatureValues = new float[numFeatures];
    this->maxFeatureValues = new float[numFeatures];

    for (int i = 0; i < numFeatures; i++) {
        float value = xTrain[0].getFeature(i);
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
            float value = xTrain[i].getFeature(j);

            if (value < minFeatureValues[j]) minFeatureValues[j] = value;
            if (value > maxFeatureValues[j]) maxFeatureValues[j] = value;
        }
    }

    calcClassProbabilities();
    calcConditionalProbabilities();

    for (int i = 0; i < numClasses * numFeatures * NUM_BINS; i++) {
        uBit.serial.printf("%d, ", (int)(conditionalProbabilities[i] * 1000));
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
            EnvironmentSample sample = xTrain[i];
            float value = sample.getFeature(j);

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
                if (conditionalProbabilities[index] == 0) conditionalProbabilities[index] = 1e-4;
            }
        }
    }
}

int NaiveBayesModel::getCProbIndex(int c, int feature, int bin) {
    return c * numFeatures * NUM_BINS + feature * NUM_BINS + bin;
}

int NaiveBayesModel::getBin(int feature, float value) {
    float range = maxFeatureValues[feature] - minFeatureValues[feature];
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
        //Use log probabilities to prevent underflow
        float classProbability = log(classProbabilities[i]);

        for (int j = 0; j < numFeatures; j++) {
            float value = sample.getFeature(j);
            int bin = getBin(j, value);
            int cProbIndex = getCProbIndex(i, j, bin);

            classProbability += log(conditionalProbabilities[cProbIndex]);
        }

        if (classProbability > highestProbability) {
            highestProbability = classProbability;
            prediction = i;
        }
    }

    return prediction;
}

EnvironmentSample samples[NUM_SAMPLES];
//Initialise the model variable to nothing (until the model is trained)
//Acts as a placeholder until a trained model is created
NaiveBayesModel* model = nullptr;


EnvironmentSample takeSample(int sampleClass) {
    uBit.display.clear();
    uBit.sleep(50);
    uint64_t start = system_timer_current_time();

    //The mean and variance of the samples is calculated using Welford's algorithm
    //This allows for calculations are the stream of inputs is coming in
    float count = 0;
    float meanTemp = 0;
    float m2Temp = 0; //Running sum of squared differences from the mean

    float meanLightLevel = 0;
    float m2LightLevel = 0;

    while (system_timer_current_time() - start < 1000) {
        int tempValue = uBit.thermometer.getTemperature();
        int lightLevelValue = uBit.display.readLightLevel();
        count++;

        //Calculate mean and variance for each axis
        float tempDiff = static_cast<float>(tempValue) - meanTemp;
        meanTemp += tempDiff / count;
        m2Temp += tempDiff * (tempValue - meanTemp);

        float lightLevelDiff = static_cast<float>(lightLevelValue) - meanLightLevel;
        meanLightLevel += lightLevelDiff / count;
        m2LightLevel += lightLevelDiff * (lightLevelValue - meanLightLevel);

        uBit.sleep(20);
    }

    float tempVariance = m2Temp / (count - 1);
    float lightLevelVariance = m2LightLevel / (count - 1);

    EnvironmentSample sample = {sampleClass, meanTemp, meanLightLevel, tempVariance, lightLevelVariance};

    uBit.serial.printf("%d, %d, %d, %d, %d\r\n",
        sampleClass, static_cast<int>(meanTemp * 1000), static_cast<int>(meanLightLevel * 1000),
        static_cast<int>(tempVariance * 1000), static_cast<int>(lightLevelVariance * 1000)
    );

    return sample;
}

void onButtonA(MicroBitEvent e) {
    EnvironmentSample sample = takeSample(currentClass);

    //If in training mode, label sample and add it to list of samples used to train the model
    //Otherwise, predict the class of the sample collected
    if (training == TRUE) {
        samples[currentSample++] = sample;
    } else {
        int prediction = model->predict(sample);
        uBit.serial.printf("Prediction %d\r\n", prediction);
        uBit.display.print(prediction);
    }
}

void onButtonB(MicroBitEvent e) {
    currentClass++;
}

void onButtonAB(MicroBitEvent e) {
    delete model;
    model = new NaiveBayesModel(4, currentClass + 1, currentSample, samples);

    uBit.serial.printf("Model trained\r\n");
    training = FALSE;
}

int main() {
    uBit.init();
    uBit.serial.setBaud(115200);

    uBit.display.setDisplayMode(DISPLAY_MODE_BLACK_AND_WHITE_LIGHT_SENSE);

    uBit.messageBus.listen(MICROBIT_ID_BUTTON_A, MICROBIT_BUTTON_EVT_CLICK, onButtonA);
    uBit.messageBus.listen(MICROBIT_ID_BUTTON_B, MICROBIT_BUTTON_EVT_CLICK, onButtonB);
    uBit.messageBus.listen(MICROBIT_ID_BUTTON_AB, MICROBIT_BUTTON_EVT_CLICK, onButtonAB);

    release_fiber();
}