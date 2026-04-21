#include "MicroBit.h"
#include "GestureRecognition.h"
#include "KNN.h"

#include <cstdio>
#include <algorithm>

#define K_VALUE 5
#define NUM_SAMPLES 100

//Global variables
MicroBit uBit;
int currentClass = 0; //The ID for the class that the user is currently providing samples of
int currentSample = 0; //The position in the samples array to add the next sample to
bool training = true;

TrainingSample samples[NUM_SAMPLES]; //The training data for the model
KNN model; //The KNN model instance

//Scale features to stop distances becoming huge
void scaleFeatures(float* meanAccX, float* meanAccY, float* meanAccZ, float* varAccX, float* varAccY, float* varAccZ, float* minAccX, float* minAccY, float* minAccZ, float* maxAccX, float* maxAccY, float* maxAccZ,
                    float* meanMagX, float* meanMagY, float* meanMagZ, float* varMagX, float* varMagY, float* varMagZ, float* minMagX, float* minMagY, float* minMagZ, float* maxMagX, float* maxMagY, float* maxMagZ) {
    *meanAccX /= 2048.0f;
    *meanAccY /= 2048.0f;
    *meanAccZ /= 2048.0f;
    *varAccX /= 1e6f;
    *varAccY /= 1e6f;
    *varAccZ /= 1e6f;
    *minAccX /= 2048.0f;
    *maxAccX /= 2048.0f;
    *minAccY /= 2048.0f;
    *maxAccY /= 2048.0f;
    *minAccZ /= 2048.0f;
    *maxAccZ /= 2048.0f;

    *meanMagX /= 30000.0f;
    *meanMagY /= 30000.0f;
    *meanMagZ /= 30000.0f;
    *varMagX /= 1e9f;
    *varMagY /= 1e9f;
    *varMagZ /= 1e9f;
    *minMagX /= 30000.0f;
    *maxMagX /= 30000.0f;
    *minMagY /= 30000.0f;
    *maxMagY /= 30000.0f;
    *minMagZ /= 30000.0f;
    *maxMagZ /= 30000.0f;
}
void printSample(GestureSample &sample) {
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
            i < FEATURE_COUNT - 1 ? "," : ""
        );
    }

    DMESG("%s", buf);
}
/**
 * Collects data from the accelerometer and the magnetometer and creates a GestureSample object from
 * the data
 * @return the GestureSample created
 */
GestureSample takeSample() {
    uint64_t start = system_timer_current_time();

    //The mean and variance of the samples is calculated using Welford's algorithm
    //This allows for calculations are the stream of inputs is coming in
    float count = 0;
    float meanAccX = 0, meanAccY = 0, meanAccZ = 0;
    float m2AccX = 0, m2AccY = 0, m2AccZ = 0; //Running sum of squared differences from the mean
    float minAccX = 2500, minAccY = 2500, minAccZ = 2500; //Set all the min values to something higher than a real sample
    float maxAccX = -2500, maxAccY = -2500, maxAccZ = -2500; //Set all the max values to something lower than a real sample

    //As mag values can be large, the mean and m2 values are stored as doubles to prevent overflow
    double dMeanMagX = 0, dMeanMagY = 0, dMeanMagZ = 0;
    double m2MagX = 0, m2MagY = 0, m2MagZ = 0;
    float minMagX = 1e9f, minMagY = 1e9f, minMagZ = 1e9f;
    float maxMagX = -1e9f, maxMagY = -1e9f, maxMagZ = -1e9f;

    while (system_timer_current_time() - start < 1000) {
        Sample3D accSample = uBit.accelerometer.getSample();
        int x = uBit.compass.getX();
        int y = uBit.compass.getY();
        int z = uBit.compass.getZ();
        count++;

        auto accX = static_cast<float>(accSample.x), accY = static_cast<float>(accSample.y), accZ = static_cast<float>(accSample.z);
        auto magX = static_cast<float>(x), magY = static_cast<float>(y), magZ = static_cast<float>(z);

        //Update min and max values seen
        minAccX = std::min(minAccX, accX);
        maxAccX = std::max(maxAccX, accX);
        minAccY = std::min(minAccY, accY);
        maxAccY = std::max(maxAccY, accY);
        minAccZ = std::min(minAccZ, accZ);
        maxAccZ = std::max(maxAccZ, accZ);

        minMagX = std::min(minMagX, magX);
        maxMagX = std::max(maxMagX, magX);
        minMagY = std::min(minMagY, magY);
        maxMagY = std::max(maxMagY, magY);
        minMagZ = std::min(minMagZ, magZ);
        maxMagZ = std::max(maxMagZ, magZ);

        //Calculate mean and variance
        float diffAccX = accX - meanAccX;
        float diffAccY = accY - meanAccY;
        float diffAccZ = accZ - meanAccZ;
        meanAccX += diffAccX / count;
        meanAccY += diffAccY / count;
        meanAccZ += diffAccZ / count;
        m2AccX += diffAccX * (accX - meanAccX);
        m2AccY += diffAccY * (accY - meanAccY);
        m2AccZ += diffAccZ * (accZ - meanAccZ);

        double diffMagX = magX - dMeanMagX;
        double diffMagY = magY - dMeanMagY;
        double diffMagZ = magZ - dMeanMagZ;
        dMeanMagX += diffMagX / count;
        dMeanMagY += diffMagY / count;
        dMeanMagZ += diffMagZ / count;
        m2MagX += diffMagX * (magX - dMeanMagX);
        m2MagY += diffMagY * (magY - dMeanMagY);
        m2MagZ += diffMagZ * (magZ - dMeanMagZ);

        uBit.sleep(2);
    }

    //Calculate variance
    float varAccX = m2AccX / (count - 1);
    float varAccY = m2AccY / (count - 1);
    float varAccZ = m2AccZ / (count - 1);

    float varMagX = static_cast<float>(m2MagX) / (count - 1);
    float varMagY = static_cast<float>(m2MagY) / (count - 1);
    float varMagZ = static_cast<float>(m2MagZ) / (count - 1);

    //Cast mag mean values down to floats
    auto meanMagX = static_cast<float>(dMeanMagX);
    auto meanMagY = static_cast<float>(dMeanMagY);
    auto meanMagZ = static_cast<float>(dMeanMagZ);

    scaleFeatures(&meanAccX, &meanAccY, &meanAccZ, &varAccX, &varAccY, &varAccZ, &minAccX, &minAccY, &minAccZ, &maxAccX, &maxAccY, &maxAccZ,
                    &meanMagX, &meanMagY, &meanMagZ, &varMagX, &varMagY, &varMagZ, &minMagX, &minMagY, &minMagZ, &maxMagX, &maxMagY, &maxMagZ);

    GestureSample sample = {
        {meanAccX, meanAccY, meanAccZ, varAccX, varAccY, varAccZ,
        minAccX, minAccY, minAccZ, maxAccX, maxAccY, maxAccZ,
        meanMagX, meanMagY, meanMagZ, varMagX, varMagY, varMagZ,
        minMagX, minMagY, minMagZ, maxMagX, maxMagY, maxMagZ}
    };

    //Print the sample for debugging
    printSample(sample);

    return sample;
}
void onButtonA(MicroBitEvent e) {
    GestureSample accSample = takeSample();

    if (training) {
        TrainingSample sample = {currentClass, accSample};
        samples[currentSample++] = sample;
    } else {
        int prediction = model.predict(accSample);
        uBit.serial.printf("Prediction %d\r\n", prediction);
        uBit.display.print(prediction);
    }
}

void onButtonB(MicroBitEvent e) {
    currentClass++;
}

void onButtonAB(MicroBitEvent e) {
    model = KNN(NUM_FEATURES, currentClass + 1, K_VALUE, currentSample, samples);
    uBit.serial.printf("Model trained\r\n");
    training = false;
}

int main() {
    uBit.init();
    uBit.compass.calibrate();

    uBit.messageBus.listen(MICROBIT_ID_BUTTON_A, MICROBIT_BUTTON_EVT_CLICK, onButtonA);
    uBit.messageBus.listen(MICROBIT_ID_BUTTON_B, MICROBIT_BUTTON_EVT_CLICK, onButtonB);
    uBit.messageBus.listen(MICROBIT_ID_BUTTON_AB, MICROBIT_BUTTON_EVT_CLICK, onButtonAB);

    release_fiber();
}
