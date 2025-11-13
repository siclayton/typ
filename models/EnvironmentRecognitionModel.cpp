#include "MicroBit.h"

MicroBit uBit;
int currentClass = 0; //The ID for the class that the user is currently providing samples of
int currentSample = 0; //The position in the samples array to add the next sample to

typedef struct {
    int sampleClass;

    float meanTemp;
    float meanLightLevel;

    float tempVariance;
    float lightLevelVariance;

} EnvironmentSample;

EnvironmentSample samples[20];

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