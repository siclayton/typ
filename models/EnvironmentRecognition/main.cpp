#include "MicroBit.h"
#include "EnvironmentRecognition.h"
#include "NaiveBayes.h"

#define NUM_SAMPLES 20

MicroBit uBit;
int currentClass = 0; //The ID for the class that the user is currently providing samples of
int currentSample = 0; //The position in the samples array to add the next sample to
bool training = true;

TrainingSample samples[NUM_SAMPLES];
//Initialise the model variable to nothing (until the model is trained)
//Acts as a placeholder until a trained model is created
NaiveBayesModel* model = nullptr;

/**
 * Collect a sample that represents the environment that the micro:bit is currently in
 * @return the sample collected
 */
EnvironmentSample takeSample() {
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

    EnvironmentSample sample = {meanTemp, meanLightLevel, tempVariance, lightLevelVariance};

    for (float feature : sample.features) {
        uBit.serial.printf("%d, ", static_cast<int>(feature * 1000));
    }
    uBit.serial.printf("\r\n");

    return sample;
}

void onButtonA(MicroBitEvent e) {
    EnvironmentSample envSample = takeSample();

    // If in training mode, label sample and add it to list of samples used to train the model
    // Otherwise, predict the class of the sample collected
    if (training) {
        TrainingSample sample = {currentClass, envSample};
        samples[currentSample++] = sample;
    } else {
        int prediction = model->predict(envSample);
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
    training = false;
}

int main() {
    uBit.init();

    uBit.display.setDisplayMode(DISPLAY_MODE_BLACK_AND_WHITE_LIGHT_SENSE);

    uBit.messageBus.listen(MICROBIT_ID_BUTTON_A, MICROBIT_BUTTON_EVT_CLICK, onButtonA);
    uBit.messageBus.listen(MICROBIT_ID_BUTTON_B, MICROBIT_BUTTON_EVT_CLICK, onButtonB);
    uBit.messageBus.listen(MICROBIT_ID_BUTTON_AB, MICROBIT_BUTTON_EVT_CLICK, onButtonAB);

    release_fiber();
}