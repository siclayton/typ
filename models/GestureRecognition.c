#include "MicroBit.h"
#include "samples/Tests.h"

MicroBit uBit;
int currentClass = 0; //The ID for the class that the user is currently providing samples of
int currentSample = 0; //The position in the samples array to add the next sample to

typedef struct {
    int sampleClass;

    //Mean movement in x,y and z axis across the time the sample was recorded for
    float meanX, meanY, meanZ;

    //Variance of movement in x,y and z, across the time the sample was recorded for
    float varX, varY, varZ;

    int minX, minY, minZ;
    int maxX, maxY, maxZ;
} AccelerometerSample;

AccelerometerSample samples[20];

AccelerometerSample takeSample(int sampleClass) {
    int start = system_timer_current_time();

    //The mean and variance of the samples is calculated using Welford's algorithm
    //This allows for calculations are the stream of inputs is coming in
    int count = 0;
    float meanX = 0, meanY = 0, meanZ = 0;
    float m2X = 0, m2Y = 0, m2Z = 0; //Running sum of squared differences from the mean

    int minX = 2500, minY = 2500, minZ = 2500; //Set all the min values to something higher than a real sample
    int maxX = -2500, maxY = -2500, maxZ = -2500; //Set all the max values to something lower than a real sample

    while (system_timer_current_time() - start < 1000) {
        Sample3D sample = uBit.accelerometer.getSample();
        count++;

        //Update min and max values seen for each axis
        if (sample.x < minX) minX = sample.x;
        if (sample.y < minY) minY = sample.y;
        if (sample.z < minZ) minZ = sample.z;
        if (sample.x > maxX) maxX = sample.x;
        if (sample.y > maxY) maxY = sample.y;
        if (sample.z > maxZ) maxZ = sample.z;

        //Calculate mean and variance for each axis
        float diffX = (float) sample.x - meanX;
        float diffY = (float) sample.y - meanY;
        float diffZ = (float) sample.z - meanZ;

        meanX += diffX / count;
        meanY += diffY / count;
        meanZ += diffZ / count;

        m2X += diffX * (sample.x - meanX);
        m2Y += diffY * (sample.y - meanY);
        m2Z += diffZ * (sample.z - meanZ);

        uBit.sleep(2);
    }

    float varX = m2X / (count - 1);
    float varY = m2Y / (count - 1);
    float varZ = m2Z / (count - 1);

    AccelerometerSample sample = {sampleClass, meanX, meanY, meanZ, varX, varY, varZ, minX, minY, minZ, maxX, maxY, maxZ};

    uBit.serial.printf("%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d\r\n",
        sampleClass,
        (int) (meanX * 1000), (int) (meanY * 1000), (int) (meanZ * 1000),
        (int) (varX * 1000), (int) (varY * 1000), (int) (varZ * 1000),
        minX, minY, minZ, maxX, maxY, maxZ
    );

    return sample;
}

void onButtonA(MicroBitEvent e) {
    AccelerometerSample sample = takeSample(currentClass);
    samples[currentSample++] = sample;
}

void onButtonB(MicroBitEvent e) {
    currentClass++;
}

int main() {
    uBit.init();

    uBit.messageBus.listen(MICROBIT_ID_BUTTON_A, MICROBIT_BUTTON_EVT_CLICK, onButtonA);
    uBit.messageBus.listen(MICROBIT_ID_BUTTON_B, MICROBIT_BUTTON_EVT_CLICK, onButtonB);

    release_fiber();
}