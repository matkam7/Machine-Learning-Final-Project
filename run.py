import tensorflow as tf
from tensorflow import keras
import os
import sys
import math
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.io import wavfile as wav
from scipy.fftpack import fft
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

male_dir = "male"
female_dir = "female"

# Setup Tensorflow
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(
    per_process_gpu_memory_fraction=0.8)
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

# Audio to FFT


def audio_to_fft(path):
    rate, data = wav.read(path)
    # wav file is mono.
    channel_1 = data[:]
    fourier = np.fft.fft(channel_1)
    return fourier


# Labelling Females and Males
ffts = []
y = []
for dir_name in os.listdir(male_dir):
    fft = audio_to_fft("./" + male_dir + "/" + dir_name + "/1.wav")
    fft.resize(20000)
    ffts.append(fft)
    y.append(0)
for dir_name in os.listdir(female_dir):
    fft = audio_to_fft("./" + female_dir + "/" + dir_name + "/1.wav")
    fft.resize(20000)
    ffts.append(fft)
    y.append(1)


# Split training/testing data
# TODO fix later
ffts = [e.real for e in ffts]
X_train, X_test, y_train, y_test = train_test_split(ffts,
                                                    y,
                                                    test_size=0.2,
                                                    stratify=y)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

linreg = LinearRegression().fit(X_train, y_train)
linreg.score(X_train, y_train)
predited = linreg.predict(X_test)
actual = y_test
print("predicted: ", predited)
print("actual: ", actual)
