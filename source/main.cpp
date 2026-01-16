#include <cmath>
#include "MicroBit.h"

#define NUM_SAMPLES 100
#define TRUE 1
#define FALSE 0

MicroBit uBit;
int currentClass = 0; //The ID for the class that the user is currently providing samples of
int currentSample = 0; //The position in the samples array to add the next sample to
int training = TRUE;

//Data type containing the features of a microphone sample
typedef struct {
    float variance;
    int max;
    float zcr;
} MicrophoneSample;

//Data type representing a labelled MicrophoneSample
//Used for training the model
typedef struct {
    int sampleClass;

    MicrophoneSample features;
}TrainingSample;

TrainingSample samples[NUM_SAMPLES];

class LogisticRegressionModel {
    public:
        ~LogisticRegressionModel() {delete[] weights;} //A destructor that ensures the memory used for the weights isn't leaked
        LogisticRegressionModel(int numFeatures, int lenXTrain, TrainingSample xTrain[]);
        LogisticRegressionModel(int numFeatures, int lenXTrain, int maxIter, float lr, float threshold, TrainingSample[]);
        int predictClass(MicrophoneSample s);
    private:
        int numFeatures{};
        int lenXTrain{};
        int maxIter{};
        float lr{};
        float threshold{}; //The minimum amount the loss must decrease by to continue training
        TrainingSample* xTrain{};

        //Values to train using gradient descent
        float* weights;
        float bias;

        float epsilon; //Used to clamp predictions to ensure the log of 0 or 1 is never computed

        void trainModel();
        float predict(MicrophoneSample);
        double sigmoid(double);
        float dotProduct(MicrophoneSample);
        float crossEntropyLoss(float[]);
        float calcBiasUpdate(float[]);
        float* calcWeightsUpdates(float[]);
};

// Constructor that uses default parameters for the maximum number of iterations, the learning rate and the loss increase threshold
LogisticRegressionModel::LogisticRegressionModel(
    int numFeatures, int lenXTrain, TrainingSample xTrain[]
    ) : LogisticRegressionModel(numFeatures, lenXTrain, 5000, 0.01, 1e-5, xTrain) {}

LogisticRegressionModel::LogisticRegressionModel(int numFeatures, int lenXTrain, int maxIter, float lr, float threshold, TrainingSample xTrain[]) {
    this->numFeatures = numFeatures;
    this->lenXTrain = lenXTrain;
    this->maxIter = maxIter;
    this->lr = lr;
    this->threshold = threshold;
    this->xTrain = xTrain;

    //Initialise both the weights and bias to 0
    this->weights = new float[numFeatures];
    for (int i = 0; i < numFeatures; i++){
        this->weights[i] = 0;
    }
    this->bias = 0;
    this->epsilon = 1e-7f; //Used to clamp predictions to avoid calculating log(0) or log(1)

    trainModel();
}

void LogisticRegressionModel::trainModel() {
    float previousLoss = 1000; //Set to a large value (any actual loss will be smaller than this)
    auto* predictions = new float[lenXTrain];

    //Training loop
    for (int i = 0; i < maxIter; i++) {
        for (int j = 0; j < lenXTrain; j++) {
            float pred = predict(xTrain[j].features);
            predictions[j] = pred;
        }

        float loss = crossEntropyLoss(predictions);

        //Calculate the updates needed for the weights and bias
        float biasChange = calcBiasUpdate(predictions);
        float* weightsChanges = calcWeightsUpdates(predictions);

        //Update the weights and bias
        bias -= lr * biasChange;
        for (int j = 0; j < numFeatures; j++) {
            weights[j] -= lr * weightsChanges[j];
        }

        delete[] weightsChanges;

        //Stop iterating if the loss hasn't improved by more than the threshold value
        if (previousLoss - loss < threshold)
            break;

        previousLoss = loss;
    }

    delete[] predictions;

    uBit.serial.printf("Bias=%d\r\n Weights =", (int) (bias *1000));
    for (int i = 0; i < numFeatures; i++){
        uBit.serial.printf("%d, ", (int) (weights[i] * 1000));
    }
    uBit.serial.printf("\r\n");
}

double LogisticRegressionModel::sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

float LogisticRegressionModel::dotProduct(MicrophoneSample s) {
    //Scale features to prevent dot product from exploding
    //Very naive scaling based on seen values when training
    float variance = (s.variance - 50.0f) / 50.0f;
    float max = (static_cast<float>(s.max) - 425.0f) / 50.0f;
    float zcr = (s.zcr - 0.16f) / 0.04f;

    float logit = 0;

    //Compute the dot product of the sample features and the weights
    logit += variance * weights[0];
    logit += max * weights[1];
    logit += zcr * weights[2];

    return logit;
}

float LogisticRegressionModel::predict(MicrophoneSample sample) {
    float logit = dotProduct(sample) + bias;

    float prediction = sigmoid(logit);

    return prediction;
}

float LogisticRegressionModel::crossEntropyLoss(float* preds) {
    double loss = 0;

    for (int i = 0; i < lenXTrain; i++) {
        float p = preds[i];

        //Clamp p into the range: epsilon >= p >= 1 - epsilon
        if (p < epsilon)
            p = epsilon;
        else if (p > 1 - epsilon)
            p = 1 - epsilon;

        //Compute the cross-entropy loss for this sample and add it to the total
        loss += static_cast<float>(xTrain[i].sampleClass) * log(p) + static_cast<float>(1 - xTrain[i].sampleClass) * log(1 - p);
    }

    return -loss / static_cast<float>(lenXTrain);
}

float LogisticRegressionModel::calcBiasUpdate(float* preds) {
    float updateAmount = 0;

    for (int i = 0; i < lenXTrain; i++) {
        updateAmount += preds[i] - static_cast<float>(xTrain[i].sampleClass);
    }

    return updateAmount / static_cast<float>(lenXTrain);
}

float *LogisticRegressionModel::calcWeightsUpdates(float* preds) {
    auto* updateAmounts = new float[numFeatures]();

    for (int i = 0; i < lenXTrain; i++) {
        //Apply feature scaling
        float variance = (xTrain[i].features.variance - 50.0f) / 50.0f;
        float max = (static_cast<float>(xTrain[i].features.max) - 425.0f) / 50.0f;
        float zcr = (xTrain[i].features.zcr - 0.16f) / 0.04f;

        float error = preds[i] - static_cast<float>(xTrain[i].sampleClass);
        updateAmounts[0] += error * variance;
        updateAmounts[1] += error * max;
        updateAmounts[2] += error * zcr;
    }

    for (int i = 0; i < numFeatures; i++) {
        updateAmounts[i] /= static_cast<float>(lenXTrain);
    }

    return updateAmounts;
}

//Return a prediction of the class of the given sample
//  1 means the model predicts speech, 0 means the model predicts no speech
int LogisticRegressionModel::predictClass(MicrophoneSample s) {
    float sigmoidOutput = predict(s);

    uBit.serial.printf("Sigmoid output = %d\r\n", static_cast<int>(sigmoidOutput * 100));

    return sigmoidOutput > 0.50 ? 1 : 0;
}

//Initialise the model variable to nothing (until the model is trained)
//Acts as a placeholder until a trained model is created
LogisticRegressionModel* model = nullptr;

MicrophoneSample takeSample(int sampleClass) {
    //Discard the first 1/4 of a second of measurements to ensure samples don't include noise
    uint64_t discardStart = system_timer_current_time();
    while (system_timer_current_time() - discardStart < 250){
        uBit.io.microphone.getAnalogValue();
    }

    uint64_t start = system_timer_current_time();

    int prev = uBit.io.microphone.getAnalogValue();

    //The mean and variance of the samples is calculated using Welford's algorithm
    //This allows for calculations are the stream of inputs is coming in
    float count = 0;
    float mean = static_cast<float>(prev);
    float m2 = 0; //Running sum of squared differences from the mean

    int max = 0;
    float zeroCrossings = 0;

    const int threshold = 2;


    while (system_timer_current_time() - start < 1000) {
        int value = uBit.io.microphone.getAnalogValue();
        count++;

        //Update max value
        if (value > max) max = value;

        float d1 = prev - mean;
        float d2 = value - mean;

        if ((d1 > threshold && d2 < -threshold) || (d1 < -threshold && d2 > threshold)) {
            zeroCrossings++;
        }
        // //Check if there was a zerocrossing
        // if ((prev - mean) * (value - mean) > 0) zeroCrossings++;

        //Update values needed to calculate variance
        float diff = static_cast<float>(value) - mean;
        mean += diff / count;
        m2 += diff * (static_cast<float>(value) - mean);

        prev = value;
        uBit.sleep(2);
    }

    float variance = m2 / (count - 1);
    float zcr = zeroCrossings / count; //Divide by the count to get the zero crossing rate

    MicrophoneSample sample = {variance, max, zcr};

    uBit.serial.printf("Sample %d: %d, %d, %d, %d\r\n",
        currentSample, sampleClass, static_cast<int>(variance * 1000), max, static_cast<int>(zcr * 1000)
    );

    return sample;
}

void onButtonA(MicroBitEvent e) {
    MicrophoneSample micSample = takeSample(currentClass);

    //If in training mode, label sample and add it to list of samples used to train the model
    //Otherwise, predict the class of the sample collected
    if (training == TRUE) {
        TrainingSample sample = {currentClass, micSample};
        samples[currentSample++] = sample;
    } else {
        int prediction = model->predictClass(micSample);
        uBit.serial.printf("Prediction %d\r\n", prediction);
        uBit.display.print(prediction);
    }
}

void onButtonB(MicroBitEvent e) {
    currentClass++;
}

void onButtonAB(MicroBitEvent e) {
    //Update model variable to a trained LogisticRegressionModel instance
    delete model;
    model = new LogisticRegressionModel(3, currentSample, samples);

    uBit.serial.printf("Model trained\r\n");
    training = FALSE;
}

int main() {
    uBit.init();

    //Turn on the microphone
    uBit.audio.enable();
    uBit.audio.activateMic();

    uBit.messageBus.listen(MICROBIT_ID_BUTTON_A, MICROBIT_BUTTON_EVT_CLICK, onButtonA);
    uBit.messageBus.listen(MICROBIT_ID_BUTTON_B, MICROBIT_BUTTON_EVT_CLICK, onButtonB);
    uBit.messageBus.listen(MICROBIT_ID_BUTTON_AB, MICROBIT_BUTTON_EVT_CLICK, onButtonAB);

    release_fiber();
}