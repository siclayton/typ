#include "MicroBit.h"
#include "samples/Tests.h"

#define NUM_SAMPLES 100
#define K_VALUE 5
#define FEATURE_COUNT 24
#define TRUE 1
#define FALSE 0

MicroBit uBit;
int currentClass = 0; //The ID for the class that the user is currently providing samples of
int currentSample = 0; //The position in the samples array to add the next sample to

int training = TRUE;

typedef struct {
    float features[FEATURE_COUNT];
} GestureSample;

typedef struct {
    int sampleClass;
    GestureSample sample;
} TrainingSample;

//The KNN model
class KNN {
    public:
        KNN() {}
        KNN(int, int, int, int, TrainingSample[]);
        int predict(GestureSample);

    private:
        int num_features;
        int num_classes;
        int k;
        int lenXTrain;
        TrainingSample* xTrain;
        TrainingSample* kNearest;

        float *nearestDistances;
        void calcKNearestNeighbours(GestureSample);
        float squared_euclidean_distance(GestureSample, int);
        void sortKNearestNeighbours();
        int majorityClass();
};

KNN::KNN(int num_features, int num_classes, int k, int lenXTrain, TrainingSample xTrain[]) {
    this->num_features = num_features;
    this->num_classes = num_classes;
    this->k = k;
    this->lenXTrain = lenXTrain;
    this->xTrain = xTrain;
    //Create two arrays to hold the nearest samples and nearest distances (used for predictions)
    this->kNearest = new TrainingSample[k];
    this->nearestDistances = new float[k];

    //Initialise the nearestDistances array to very large values
    //All the values calculated will be smaller than these so they will be placed into the array
    for (int i = 0; i < k; i++) {
        nearestDistances[i] = 1e30f;
    }
}

//Predict the class of a given sample
int KNN::predict(GestureSample sample) {
    calcKNearestNeighbours(sample); //Update the kNearest and nearestDistances arrays
    int prediction = majorityClass(); //Use the modal class of the kNearest samples to predict the class of the sample

    return prediction;
}

//Update the kNearest and nearestDistances arrays
void KNN::calcKNearestNeighbours(GestureSample s) {
    for (int i = 0; i < this->lenXTrain; i++) {
        //Calculate the distance between the sample s and this element in the xTrain array
        float dist = squared_euclidean_distance(s, i);

        //If the kNearest array isn't full or the distance to this element is less than the lowest distance in the nearestDistances array,
        //Add the sample to the kNearest samples and the distance to the nearestDistances array
        if (i < k) {
            nearestDistances[i] = dist;
            kNearest[i] = xTrain[i];
        } else {
            //Sort the arrays, so the lowest distance is at index k-1
            sortKNearestNeighbours();
            if (dist < nearestDistances[k-1]) {
                nearestDistances[k-1] = dist;
                kNearest[k-1] = xTrain[i];
            }
        }
    }
}

//Calculate the Squared Euclidean distance between the given sample and the sample at the given index in the xTrain array
float KNN::squared_euclidean_distance(GestureSample sample1, int index) {
    GestureSample sample2 = xTrain[index].sample;

    //Calculate differences in variables
    float dist = 0;

    for (int i = 0; i < FEATURE_COUNT; i++) {
        float featureDiff = sample1.features[i] - sample2.features[i];
        dist += featureDiff * featureDiff;
    }

    return dist;
}

//Sort the kNearest and nearestDistances arrays, using bubble sort
void KNN::sortKNearestNeighbours(){
    for (int i = 0; i < k - 1; i++) {
        for (int j = 0; j < k - i - 1; j++) {
            if (nearestDistances[j] > nearestDistances[j + 1]) {
                float tempDist = nearestDistances[j];
                nearestDistances[j] = nearestDistances[j+1];
                nearestDistances[j+1] = tempDist;

                TrainingSample tempSample = kNearest[j];
                kNearest[j] = kNearest[j+1];
                kNearest[j+1] = tempSample;
            }
        }
    }
}

//Determine the modal class of the kNearest array
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

GestureSample takeSample() {
    uint64_t start = system_timer_current_time();

    //The mean and variance of the samples is calculated using Welford's algorithm
    //This allows for calculations are the stream of inputs is coming in
    float count = 0;
    float meanAccX = 0, meanAccY = 0, meanAccZ = 0;
    float m2AccX = 0, m2AccY = 0, m2AccZ = 0; //Running sum of squared differences from the mean
    float minAccX = 2500, minAccY = 2500, minAccZ = 2500; //Set all the min values to something higher than a real sample
    float maxAccX = -2500, maxAccY = -2500, maxAccZ = -2500; //Set all the max values to something lower than a real sample

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
        if (accX < minAccX) minAccX = accX;
        if (accY < minAccY) minAccY = accY;
        if (accZ < minAccZ) minAccZ = accZ;
        if (accX > maxAccX) maxAccX = accX;
        if (accY > maxAccY) maxAccY = accY;
        if (accZ > maxAccZ) maxAccZ = accZ;

        if (magX < minMagX) minMagX = magX;
        if (magY < minMagY) minMagY = magY;
        if (magZ < minMagZ) minMagZ = magZ;
        if (magX > maxMagX) maxMagX = magX;
        if (magY > maxMagY) maxMagY = magY;
        if (magZ > maxMagZ) maxMagZ = magZ;

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

    GestureSample sample = {meanAccX, meanAccY, meanAccZ, varAccX, varAccY, varAccZ,
                            minAccX, minAccY, minAccZ, maxAccX, maxAccY, maxAccZ,
                            meanMagX, meanMagY, meanMagZ, varMagX, varMagY, varMagZ,
                            minMagX, minMagY, minMagZ, maxMagX, maxMagY, maxMagZ};

    //Print the sample for debugging
    for (float feature : sample.features) {
        uBit.serial.printf("%d, ", static_cast<int>(feature * 1000));
        uBit.sleep(2);
    }
    uBit.serial.printf("\r\n");

    return sample;
}

void onButtonA(MicroBitEvent e) {
    GestureSample accSample = takeSample();

    if (training == TRUE) {
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
    model = KNN(FEATURE_COUNT, currentClass + 1, K_VALUE, currentSample, samples);
    uBit.serial.printf("Model trained\r\n");
    training = FALSE;
}

int main() {
    uBit.init();
    uBit.compass.calibrate();

    uBit.messageBus.listen(MICROBIT_ID_BUTTON_A, MICROBIT_BUTTON_EVT_CLICK, onButtonA);
    uBit.messageBus.listen(MICROBIT_ID_BUTTON_B, MICROBIT_BUTTON_EVT_CLICK, onButtonB);
    uBit.messageBus.listen(MICROBIT_ID_BUTTON_AB, MICROBIT_BUTTON_EVT_CLICK, onButtonAB);

    release_fiber();
}