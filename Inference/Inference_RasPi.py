import sounddevice 
import numpy as np
import scipy.signal
import timeit
import librosa

import RPi.GPIO as GPIO
import time

from tflite_runtime.interpreter import Interpreter
'''
# GPIO parameters
LED_PIN = 16
FAN_PIN = 18
GPIO.setmode(GPIO.BOARD)
# Led
GPIO.setup(LED_PIN, GPIO.OUT)
GPIO.output(LED_PIN, GPIO.LOW)
# Fan
GPIO.setup(FAN_PIN, GPIO.OUT)
p = GPIO.PWM(FAN_PIN, 25000)
p.start(0)
dc = 0
'''


# Inference Parameters
debug_time = 1
debug_acc = 1
led_pin = 8
word_threshold = 0.5

rec_duration = 0.5
window_stride = 0.01
microphone_sample_rate = 48000
resample_rate = 16000
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
    rec, new_fs = decimate(rec, microphone_sample_rate, resample_rate)
    
    # Save recording onto sliding window
    window[:len(window)//2] = window[len(window)//2:]
    window[len(window)//2:] = rec

    # Compute features
    sr = 16000
    n_mfcc = 64
    n_mels = 64
    n_fft = 512 
    window_stride = 0.5
    hop_length = int(sr*window_stride)
    fmin = 0
    fmax = None
    
    
    mfcc_librosa = librosa.feature.mfcc(y=window, sr=sr, n_fft=n_fft,
                                        n_mfcc=n_mfcc, n_mels=n_mels,
                                        hop_length=hop_length,
                                        fmin=fmin, fmax=fmax, htk=False)
    
    print (mfcc_librosa.shape)
    
    # Make prediction from model
    in_tensor = np.float32(mfcc_librosa.reshape(1, mfcc_librosa.shape[0], mfcc_librosa.shape[1], 1))
    interpreter.set_tensor(input_details[0]['index'], in_tensor)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    val = output_data[0]
    val = list(np.around(np.array(val),2))
    
    # debug
    print('mfccs.shape:', mfcc_librosa.shape)
    print('in_tensor.shape:', in_tensor.shape)
    print('rec:', rec)
    print('frames:', frames)
    print('new_fs:', new_fs)
    
    
    
    train_commands = ["yes","no","up","down","left","right","on","off","stop","go","unknown","slience"]
    if debug_acc:
      print(train_commands)
      print(val)
        
    if debug_time:
        print('Latency:', round(timeit.default_timer() - start , 4) ,' ms')
    
    '''
    # global parameters
    global dc
    global LED_PIN
    
    # Choose the max score and check word_threshold
    word_threshold = 1
    perdict_index = np.argmax(val)
    print ('perdict index:',perdict_index)
    
    if val[perdict_index] > word_threshold:
        print ('dectect command:',train_commands[perdict_index])
        # Control the GPIO
        if perdict_index == 0:
            GPIO.output(LED_PIN, GPIO.HIGH)
            print("on!")
        elif perdict_index == 1:  
            GPIO.output(LED_PIN, GPIO.LOW)
            print("off!")
        elif perdict_index == 2:  
            dc = dc + 50
            if dc > 100:
                dc = 100
            p.ChangeDutyCycle(dc)
            print("speed up!, now speed:",dc)
        elif perdict_index == 3:
            dc = dc - 50
            if dc < 0:
                dc = 0
            p.ChangeDutyCycle(dc)
            print("speed down!, now speed:",dc)
        elif perdict_index == 4:
            dc = 0
            p.ChangeDutyCycle(dc)
            print("stop!, now speed:",dc)
    else :
        print ('dectect voice:',train_commands[-1])
    print('----------------------------------------------------------------------------')
    '''


    
    
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
                    samplerate=microphone_sample_rate,
                    blocksize=int(microphone_sample_rate * rec_duration),
                    callback=sd_callback):
    while True:
        pass
