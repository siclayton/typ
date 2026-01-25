#include "MicroBit.h"

#include <vector>

#define NUM_SAMPLES 20
#define NUM_BINS 4

MicroBit uBit;
int currentClass = 0; //The ID for the class that the user is currently providing samples of
int currentSample = 0; //The position in the samples array to add the next sample to

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
        float getConditionalProbability(int c, int feature, int bin);
        int getBin(int feature, float value);
};

NaiveBayesModel::NaiveBayesModel(int numFeatures, int numClasses, int lenXTrain, EnvironmentSample* xTrain) {
    this->numFeatures = numFeatures;
    this->numClasses = numClasses;
    this->lenXTrain = lenXTrain;
    this->xTrain = xTrain;

    this->classCounts = new int[numClasses];
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
}

void NaiveBayesModel::calcClassProbabilities() {
    for (int i = 0; i < numClasses; i++) {
        classProbabilities[i] = static_cast<float>(classCounts[i]) / static_cast<float>(lenXTrain);
    }
}

void NaiveBayesModel::calcConditionalProbabilities() {

}

float NaiveBayesModel::getConditionalProbability(int c, int feature, int bin) {
    return conditionalProbabilities[c * numFeatures * NUM_BINS +
                                    feature * NUM_BINS + bin ];
}

EnvironmentSample samples[NUM_SAMPLES];

EnvironmentSample takeSample(int sampleClass) {
    uint64_t start = system_timer_current_time();

    //The mean and variance of the samples is calculated using Welford's algorithm
    //This allows for calculations are the stream of inputs is coming in
    int count = 0;
    float meanTemp = 0;
    float m2Temp = 0; //Running sum of squared differences from the mean

    float meanLightLevel = 0;
    float m2LightLevel = 0;

    while (system_timer_current_time() - start < 1000) {
        int tempValue = uBit.thermometer.getTemperature();
        int lightLevelValue = uBit.display.readLightLevel();
        count++;

        //Calculate mean and variance for each axis
        float tempDiff = (float) tempValue - meanTemp;
        meanTemp += tempDiff / count;
        m2Temp += tempDiff * (tempValue - meanTemp);

        float lightLevelDiff = (float) lightLevelValue - meanLightLevel;
        meanLightLevel += lightLevelDiff / count;
        m2LightLevel += lightLevelDiff * (lightLevelValue - meanLightLevel);

        uBit.sleep(2);
    }

    float tempVariance = m2Temp / (count - 1);
    float lightLevelVariance = m2LightLevel / (count - 1);

    EnvironmentSample sample = {sampleClass, meanTemp, meanLightLevel, tempVariance, lightLevelVariance};

    uBit.serial.printf("%d, %d, %d, %d, %d\r\n",
        sampleClass, (int) (meanTemp * 1000), (int) (meanLightLevel * 1000),
        (int) (tempVariance * 1000), (int) (lightLevelVariance * 1000)
    );

    return sample;
}

void onButtonA(MicroBitEvent e) {
    EnvironmentSample sample = takeSample(currentClass);
    samples[currentSample++] = sample;
}

void onButtonB(MicroBitEvent e) {
    currentClass++;
}

int main() {
    uBit.init();
    uBit.serial.setBaud(115200);

    uBit.display.setDisplayMode(DISPLAY_MODE_BLACK_AND_WHITE_LIGHT_SENSE);

    uBit.messageBus.listen(MICROBIT_ID_BUTTON_A, MICROBIT_BUTTON_EVT_CLICK, onButtonA);
    uBit.messageBus.listen(MICROBIT_ID_BUTTON_B, MICROBIT_BUTTON_EVT_CLICK, onButtonB);

    release_fiber();
}