#include "MicroBit.h"
#define ARM_MATH_CM4
#include "arm_math.h"

#define MIC_SAMPLE_RATE         (11 * 1024)
#define AUDIO_SAMPLES_NUMBER    1024

MicroBit uBit;
arm_rfft_fast_instance_f32 fftInstance;

typedef struct {

} SpeechSample;

typedef struct {
    int sampleClass;
    SpeechSample sample;
} TrainingSample;

SpeechSample takeSample() {

    // Capture a seconds worth of audio
        // Will output the buf used in fft function call

    // Calculate FFT and mag
    auto* fftOutput = static_cast<float *>(malloc(sizeof(float) * AUDIO_SAMPLES_NUMBER));
    auto* magnitudeSpectrum = static_cast<float *>(malloc(sizeof(float) * AUDIO_SAMPLES_NUMBER / 2));

    arm_rfft_fast_f32(&fftInstance, buf + offset, fftOutput, 0);
    arm_cmplx_mag_f32(fftOutput, magnitudeSpectrum, AUDIO_SAMPLES_NUMBER / 2);

    // Apply Mel filter bank
        // Do not need to compute the triangles here (that should be done on start up)

    // Pack feature vector returned by mel filter bank calculations into a SpeechSample
    return SpeechSample();
}

void onButtonA(MicroBitEvent e){
    //Take sample
}
void onButtonB(MicroBitEvent e);
void onButtonAB(MicroBitEvent e);

int main() {
    uBit.init();
    uBit.audio.enable();

    // Initialise the fft instance
    arm_rfft_fast_init_f32(&fftInstance, AUDIO_SAMPLES_NUMBER);

    // Compute Mel filterbank

    uBit.messageBus.listen(MICROBIT_ID_BUTTON_A, MICROBIT_BUTTON_EVT_CLICK, onButtonA);
    uBit.messageBus.listen(MICROBIT_ID_BUTTON_B, MICROBIT_BUTTON_EVT_CLICK, onButtonB);
    uBit.messageBus.listen(MICROBIT_ID_BUTTON_AB, MICROBIT_BUTTON_EVT_CLICK, onButtonAB);

    release_fiber();
}