#include <math.h>
#include "MicroBit.h"

#define NUM_SAMPLES 25
#define TRUE 1
#define FALSE 0

MicroBit uBit;
int currentClass = 0; //The ID for the class that the user is currently providing samples of
int currentSample = 0; //The position in the samples array to add the next sample to
int training = TRUE;

typedef struct {
    int sampleClass;

    float mean;
    float variance;

    int max;

    int zeroCrossings;
} MicrophoneSample;

MicrophoneSample samples[NUM_SAMPLES];

class LogisticRegressionModel {
    public:
        ~LogisticRegressionModel() {delete[] weights;} //A destructor that ensures the memory used for the weights isn't leaked
        LogisticRegressionModel() = default;
        LogisticRegressionModel(int numFeatures, int lenXTrain, MicrophoneSample xTrain[]);
        LogisticRegressionModel(int numFeatures, int lenXTrain, int maxIter, float lr, float threshold, MicrophoneSample[]);
        int predictClass(MicrophoneSample s);
    private:
        int numFeatures{};
        int lenxTrain{};
        int maxIter{};
        float lr{};
        float threshold{}; //The minimum amount the loss must decrease by to continue training
        MicrophoneSample* xTrain{};

        //Values to train using gradient descent
        float* weights{};
        float bias{};

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
    int numFeatures, int lenxTrain, MicrophoneSample xTrain[]
    ) : LogisticRegressionModel(numFeatures, lenxTrain, 1000, 0.01, 0.0001, xTrain) {}

LogisticRegressionModel::LogisticRegressionModel(int numFeatures, int lenxTrain, int maxIter, float lr, float threshold, MicrophoneSample xTrain[]) {
    this->numFeatures = numFeatures;
    this->lenxTrain = lenxTrain;
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

    trainModel();
}

void LogisticRegressionModel::trainModel() {
    float previousLoss = 1000; //Set to a large value
    float* predictions = new float[lenxTrain];

    for (int i = 0; i < maxIter; i++) {
        for (int j = 0; j < lenxTrain; j++) {
            float pred = predict(xTrain[j]);
            predictions[j] = pred;
        }

        float loss = crossEntropyLoss(predictions);

        float biasChange = calcBiasUpdate(predictions);
        float* weightsChanges = calcWeightsUpdates(predictions);

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
}

double LogisticRegressionModel::sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

float LogisticRegressionModel::dotProduct(MicrophoneSample s) {
    float logit = 0;

    //Compute the dot product of the sample features and the weights
    logit += s.mean * weights[0];
    logit += s.variance * weights[1];
    logit += (float) s.max * weights[2];
    logit += (float) s.zeroCrossings * weights[3];

    return logit;
}

//Return a prediction of the class of the given sample
//  1 means the model predicts speech, 0 means the model predicts no speech
float LogisticRegressionModel::predict(MicrophoneSample sample) {
    float logit = dotProduct(sample) + bias;

    float prediction = sigmoid(logit);

    return prediction;
}

float LogisticRegressionModel::crossEntropyLoss(float* preds) {
    double loss = 0;

    for (int i = 0; i < lenxTrain; i++) {
        loss += xTrain[i].sampleClass * log(preds[i]) + (1 - xTrain[i].sampleClass) * log(1 - preds[i]);
    }

    return -loss / (float) lenxTrain;
}

float LogisticRegressionModel::calcBiasUpdate(float* preds) {
    float updateAmount = 0;

    for (int i = 0; i < lenxTrain; i++) {
        updateAmount += preds[i] - (float) xTrain[i].sampleClass;
    }

    return updateAmount / (float) lenxTrain;
}

float *LogisticRegressionModel::calcWeightsUpdates(float* preds) {
    float* updateAmounts = new float[numFeatures]();

    for (int i = 0; i < lenxTrain; i++) {
        updateAmounts[0] += (preds[i] - xTrain[i].sampleClass) * xTrain[i].mean;
        updateAmounts[1] += (preds[i] - xTrain[i].sampleClass) * xTrain[i].variance;
        updateAmounts[2] += (preds[i] - xTrain[i].sampleClass) * (float) xTrain[i].max;
        updateAmounts[3] += (preds[i] - xTrain[i].sampleClass) * (float) xTrain[i].zeroCrossings;
    }

    for (int i = 0; i < numFeatures; i++) {
        updateAmounts[i] /= (float) lenxTrain;
    }

    return updateAmounts;
}

int LogisticRegressionModel::predictClass(MicrophoneSample s) {
    float sigmoidOutput = predict(s);

    return sigmoidOutput > 0.5 ? 1 : 0;
}

LogisticRegressionModel model; //The logistic regression model instance

MicrophoneSample takeSample(int sampleClass) {
    uint64_t start = system_timer_current_time();

    //The mean and variance of the samples is calculated using Welford's algorithm
    //This allows for calculations are the stream of inputs is coming in
    int count = 0;
    float mean = 0;
    float m2 = 0; //Running sum of squared differences from the mean

    int max = 0;
    int zeroCrossings = 0;

    int lastValueWasBelowMid = 0;

    while (system_timer_current_time() - start < 1000) {
        int value = uBit.io.microphone.getAnalogValue();
        count++;

        //Update max value
        if (value > max) max = value;

        /*
            The microphone outputs values from 0 to 1024, so to detect how much the noise oscillates
            I am detecting how often it goes above and below 512, as this is the mid-point, but I am
            having a threshold of 10, so the value must be either 10 or more over or under 512 to count
        */
        if (lastValueWasBelowMid == 0 && value < 522) {
            zeroCrossings++;
            lastValueWasBelowMid = 1;
        } else if (lastValueWasBelowMid == 1 && value > 532) {
            zeroCrossings++;
            lastValueWasBelowMid = 0;
        }

        //Calculate mean and variance for each axis
        float diff = (float) value - mean;
        mean += diff / count;
        m2 += diff * (value - mean);

        uBit.sleep(2);
    }

    float variance = m2 / (count - 1);

    MicrophoneSample sample = {sampleClass, mean, variance, max, zeroCrossings};

    uBit.serial.printf("%d, %d, %d, %d, %d\r\n",
        sampleClass, (int) (mean * 1000), (int) (variance * 1000), max, zeroCrossings
    );

    return sample;
}

void onButtonA(MicroBitEvent e) {
    MicrophoneSample sample = takeSample(currentClass);

    if (training == TRUE) {
        samples[currentSample++] = sample;
    } else {
        int prediction = model.predictClass(sample);
        uBit.serial.printf("Prediction %d\r\n", prediction);
        uBit.display.print(prediction);
    }
}

void onButtonB(MicroBitEvent e) {
    currentClass++;
}

void onButtonAB(MicroBitEvent e) {
    model = LogisticRegressionModel(4, NUM_SAMPLES, samples);
    uBit.serial.printf("Model trained\r\n");
    training = FALSE;
}

int main() {
    uBit.init();

    uBit.audio.enable();
    uBit.audio.activateMic();

    uBit.messageBus.listen(MICROBIT_ID_BUTTON_A, MICROBIT_BUTTON_EVT_CLICK, onButtonA);
    uBit.messageBus.listen(MICROBIT_ID_BUTTON_B, MICROBIT_BUTTON_EVT_CLICK, onButtonB);

    release_fiber();
}