import sounddevice 
import numpy as np
import scipy.signal
import timeit
import python_speech_features

import RPi.GPIO as GPIO
import time

from tflite_runtime.interpreter import Interpreter

# Inference Parameters
debug_time = 1
debug_acc = 1
led_pin = 8
word_threshold = 0.5
rec_duration = 0.5
window_stride = 0.5
sample_rate = 48000
resample_rate = 8000
num_channels = 1
num_mfcc = 16
model_path = 'SpeechCommandRecognition_model.tflite'

# Sliding window
window = np.zeros(int(rec_duration * resample_rate) * 2)

# Load model (interpreter)
interpreter = Interpreter(model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)

# Decimate (filter and downsample)
def decimate(signal, old_fs, new_fs):
    
    # Check to make sure we're downsampling
    if new_fs > old_fs:
        print("Error: target sample rate higher than original")
        return signal, old_fs
    
    # We can only downsample by an integer factor
    dec_factor = old_fs / new_fs
    if not dec_factor.is_integer():
        print("Error: can only decimate by integer factor")
        return signal, old_fs

    # Do decimation
    resampled_signal = scipy.signal.decimate(signal, int(dec_factor))

    return resampled_signal, new_fs

# This gets called every 0.5 seconds
def sd_callback(rec, frames, time, status):

    # Start timing for testing
    start = timeit.default_timer()
    
    # Notify if errors
    if status:
        print('Error:', status)
    
    # Remove 2nd dimension from recording sample
    rec = np.squeeze(rec)
    
    # Resample
    rec, new_fs = decimate(rec, sample_rate, resample_rate)
    
    # Save recording onto sliding window
    window[:len(window)//2] = window[len(window)//2:]
    window[len(window)//2:] = rec

    # Compute features
    mfccs = python_speech_features.base.mfcc(window, 
                                        samplerate=new_fs,
                                        winlen=0.256,
                                        winstep=0.050,
                                        numcep=num_mfcc,
                                        nfilt=26,
                                        nfft=4096,
                                        preemph=0.0,
                                        ceplifter=0,
                                        appendEnergy=False,
                                        winfunc=np.hanning)
    mfccs = mfccs.transpose()

    # Make prediction from model
    in_tensor = np.float32(mfccs.reshape(1, mfccs.shape[0], mfccs.shape[1], 1))
    interpreter.set_tensor(input_details[0]['index'], in_tensor)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    val = output_data[0]

    if debug_acc:
        print(val)
        
    if debug_time:
        print(timeit.default_timer() - start)
    
    # Choose the max score and check word_threshold
    word_threshold = 5
    perdict_index = np.argmax(val)
    print ('perdict index:',perdict_index)
    train_commands = ['on','stop','unknown','slience']
    if val[perdict_index] > word_threshold:
        print ('dectect voice:',train_commands[perdict_index])
    else :
        print ('dectect voice:',train_commands[-1])
    print('----------------------------------------------------------------------------')
   
    # Control the GPIO
    # Gpio parameters
    LED_PIN = 16
    FAN_PIN = 18
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(LED_PIN, GPIO.OUT)
    if perdict_index == 0:
        GPIO.output(LED_PIN, GPIO.HIGH)
        print("on!")
    elif perdict_index == 1:  
        GPIO.output(LED_PIN, GPIO.LOW)
        print("off!")

    
    
    '''
    # Choose the max score
    perdict_index = np.argmax(val)
    print ('perdict index:',perdict_index)
    train_commands = ['on','stop','unknown','slience']
    print ('dectect voice:',train_commands[perdict_index])
    print('----------------------------------------------------------------------------')
    '''
    
# Start streaming from microphone
with sounddevice.InputStream(channels=num_channels,
                    samplerate=sample_rate,
                    blocksize=int(sample_rate * rec_duration),
                    callback=sd_callback):
    while True:
        pass
