#include "MicroBit.h"
#include "samples/Tests.h"

#define NUM_SAMPLES 25
#define K_VALUE 5
#define TRUE 1
#define FALSE 0

MicroBit uBit;
int currentClass = 0; //The ID for the class that the user is currently providing samples of
int currentSample = 0; //The position in the samples array to add the next sample to

int training = TRUE;

typedef struct {
    int sampleClass;

    //Mean movement in x,y and z axis across the time the sample was recorded for
    float meanX, meanY, meanZ;

    //Variance of movement in x,y and z, across the time the sample was recorded for
    float varX, varY, varZ;

    float minX, minY, minZ;
    float maxX, maxY, maxZ;
} AccelerometerSample;

AccelerometerSample samples[NUM_SAMPLES];

AccelerometerSample takeSample(int sampleClass) {
    int start = system_timer_current_time();

    //The mean and variance of the samples is calculated using Welford's algorithm
    //This allows for calculations are the stream of inputs is coming in
    float count = 0;
    float meanX = 0, meanY = 0, meanZ = 0;
    float m2X = 0, m2Y = 0, m2Z = 0; //Running sum of squared differences from the mean

    float minX = 2500, minY = 2500, minZ = 2500; //Set all the min values to something higher than a real sample
    float maxX = -2500, maxY = -2500, maxZ = -2500; //Set all the max values to something lower than a real sample

    while (system_timer_current_time() - start < 1000) {
        Sample3D sample = uBit.accelerometer.getSample();
        count++;

        float x = (float) sample.x, y = (float) sample.y, z = (float) sample.z;

        //Update min and max values seen for each axis
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (z < minZ) minZ = z;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
        if (z > maxZ) maxZ = z;

        //Calculate mean and variance for each axis
        float diffX = x - meanX;
        float diffY = y - meanY;
        float diffZ = z - meanZ;

        meanX += diffX / count;
        meanY += diffY / count;
        meanZ += diffZ / count;

        m2X += diffX * (x- meanX);
        m2Y += diffY * (y - meanY);
        m2Z += diffZ * (z - meanZ);

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
        (int) minX, (int) minY, (int) minZ, (int) maxX, (int) maxY, (int) maxZ
    );

    return sample;
}

class KNN {
    public:
        int num_features;
        int num_classes;
        int k;
        int lenxTrain;
        AccelerometerSample* xTrain;
        KNN() {}
        KNN(int, int, int, int, AccelerometerSample[]);
        int predict(AccelerometerSample);

    private:
        AccelerometerSample* kNearest;
        float *nearestDistances;
        void calcKNearestNeighbours(AccelerometerSample);
        float squared_euclidean_distance(AccelerometerSample, int);
        void sortKNearestNeighbours();
        int majorityClass();
};

KNN::KNN(int num_features, int num_classes, int k, int lenxTrain, AccelerometerSample xTrain[]) {
    this->num_features = num_features;
    this->num_classes = num_classes;
    this->k = k;
    this->lenxTrain = lenxTrain;
    this->xTrain = xTrain;
    this->kNearest = new AccelerometerSample[k];
    this->nearestDistances = new float[k];

    for (int i = 0; i < k; i++) {
        nearestDistances[i] = 1e30f;
    }
}

int KNN::predict(AccelerometerSample sample) {
    calcKNearestNeighbours(sample);
    int prediction = majorityClass();

    return prediction;
}

void KNN::calcKNearestNeighbours(AccelerometerSample s) {
    for (int i = 0; i < this->lenxTrain; i++) {
        float dist = squared_euclidean_distance(s, i);

        if (i < k) {
            nearestDistances[i] = dist;
            kNearest[i] = xTrain[i];
            uBit.serial.printf("Inserted dist %d at index %d\r\n", (int) (dist * 1000), i);
        } else {
            sortKNearestNeighbours();
            if (dist < nearestDistances[k-1]) {
                nearestDistances[k-1] = dist;
                kNearest[k-1] = xTrain[i];
                uBit.serial.printf("Inserted dist %d at index %d\r\n", (int) (dist * 1000), k-1);
            }
        }
        uBit.sleep(20);
    }

    for (int i = 0; i < k; i++) {
        uBit.serial.printf("%d\r\n", i);
        uBit.serial.printf("%d, ",(int) (nearestDistances[i] * 1000));
        uBit.serial.printf("%d\r\n", kNearest[i].sampleClass);
    }
}

float KNN::squared_euclidean_distance(AccelerometerSample sample1, int index) {
    AccelerometerSample sample2 = xTrain[index];

    //Scale features to stop distances becoming huge (aim is to try to scale features into range (-1, 1)
    float scaledMeanX1 = sample1.meanX / 2048.0f, scaledMeanX2 = sample2.meanX / 2048.0f;
    float scaledMeanY1 = sample1.meanY / 2048.0f, scaledMeanY2 = sample2.meanY / 2048.0f;
    float scaledMeanZ1 = sample1.meanZ / 2048.0f, scaledMeanZ2 = sample2.meanZ / 2048.0f;
    float scaledVarX1 = sample1.varX / 1e6f, scaledVarX2 = sample2.varX / 1e6f;
    float scaledVarY1 = sample1.varY / 1e6f, scaledVarY2 = sample2.varY / 1e6f;
    float scaledVarZ1 = sample1.varZ / 1e6f, scaledVarZ2 = sample2.varZ / 1e6f;
    float scaledMinX1 = sample1.minX / 2048.0f, scaledMinX2 = sample2.minX / 2048.0f;
    float scaledMaxX1 = sample1.maxX / 2048.0f, scaledMaxX2 = sample2.maxX / 2048.0f;
    float scaledMinY1 = sample1.minY / 2048.0f, scaledMinY2 = sample2.minY / 2048.0f;
    float scaledMaxY1 = sample1.maxY / 2048.0f, scaledMaxY2 = sample2.maxY / 2048.0f;
    float scaledMinZ1 = sample1.minZ / 2048.0f, scaledMinZ2 = sample2.minZ / 2048.0f;
    float scaledMaxZ1 = sample1.maxZ / 2048.0f, scaledMaxZ2 = sample2.maxZ / 2048.0f;

    //Calculate differences in variables
    float diffMeanX = scaledMeanX1 - scaledMeanX2;
    float diffMeanY = scaledMeanY1 - scaledMeanY2;
    float diffMeanZ = scaledMeanZ1 - scaledMeanZ2;
    float diffVarX = scaledVarX1 - scaledVarX2;
    float diffVarY = scaledVarY1 - scaledVarY2;
    float diffVarZ = scaledVarZ1 - scaledVarZ2;
    float diffMinX = scaledMinX1 - scaledMinX2;
    float diffMaxX = scaledMaxX1 - scaledMaxX2;
    float diffMinY = scaledMinY1 - scaledMinY2;
    float diffMaxY = scaledMaxY1 - scaledMaxY2;
    float diffMinZ = scaledMinZ1 - scaledMinZ2;
    float diffMaxZ = scaledMaxZ1 - scaledMaxZ2;

    float dist = diffMeanX * diffMeanX +
                diffMeanY * diffMeanY +
                diffMeanZ * diffMeanZ +
                diffVarX * diffVarX +
                diffVarY * diffVarY +
                diffVarZ * diffVarZ +
                diffMinX * diffMinX +
                diffMaxX * diffMaxX +
                diffMinY * diffMinY +
                diffMaxY * diffMaxY +
                diffMinZ * diffMinZ +
                diffMaxZ * diffMaxZ;

    return dist;
}

void KNN::sortKNearestNeighbours(){
    for (int i = 0; i < k - 1; i++) {
        for (int j = 0; j < k - i - 1; j++) {
            if (nearestDistances[j] > nearestDistances[j + 1]) {
                float tempDist = nearestDistances[j];
                nearestDistances[j] = nearestDistances[j+1];
                nearestDistances[j+1] = tempDist;

                AccelerometerSample tempSample = kNearest[j];
                kNearest[j] = kNearest[j+1];
                kNearest[j+1] = tempSample;
            }
        }
    }
}

int KNN::majorityClass() {
    int majorityClass = 0, maxCount = 0;

    for (int i = 0; i < num_classes; i++) {
        int classCount = 0;

        for (int j = 0; j < k; j++) {
            if (kNearest[j].sampleClass == i) {
                classCount++;
            }
        }

        if (classCount > maxCount) {
            maxCount = classCount;
            majorityClass = i;
        }
    }
    return majorityClass;
}

KNN model;

void onButtonA(MicroBitEvent e) {
    AccelerometerSample sample = takeSample(currentClass);

    if (training == TRUE) {
        samples[currentSample++] = sample;
    } else {
        int prediction = model.predict(sample);
        uBit.serial.printf("Prediction %d\r\n", prediction);
    }
}

void onButtonB(MicroBitEvent e) {
    currentClass++;
}

void onButtonAB(MicroBitEvent e) {
    model = KNN(12, currentClass + 1, K_VALUE, currentSample, samples);
    uBit.serial.printf("Model trained\r\n");
    training = FALSE;
}

int main() {
    uBit.init();

    uBit.messageBus.listen(MICROBIT_ID_BUTTON_A, MICROBIT_BUTTON_EVT_CLICK, onButtonA);
    uBit.messageBus.listen(MICROBIT_ID_BUTTON_B, MICROBIT_BUTTON_EVT_CLICK, onButtonB);
    uBit.messageBus.listen(MICROBIT_ID_BUTTON_AB, MICROBIT_BUTTON_EVT_CLICK, onButtonAB);

    release_fiber();
}