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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical
from scipy import signal
from scipy.io import wavfile
import scipy.misc

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
    fft.resize(30000)
    ffts.append(fft)
    y.append(0)
for dir_name in os.listdir(female_dir):
    fft = audio_to_fft("./" + female_dir + "/" + dir_name + "/1.wav")
    fft.resize(30000)
    ffts.append(fft)
    y.append(1)


# Split training/testing data
# TODO fix later
ffts = [e.real for e in ffts]
X_train, X_test, y_train, y_test = train_test_split(ffts,
                                                    y,
                                                    test_size=0.4,
                                                    stratify=y)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

linreg = LogisticRegression().fit(X_train, y_train)
predited = linreg.predict(X_test)

print("y_test:   ", y_test)
print("predited: ", predited)
print("accuracy: ", accuracy_score(y_test, predited))


# .wav to spectrogram (Frequency Count vs Time)
sample_rate, samples = wavfile.read('female/bfeciyuh/1.wav')
print(samples.shape)
frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
spectrogram = np.log10(spectrogram)

plt.pcolormesh(times, frequencies, spectrogram)
# plt.imshow(spectrogram)
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()

print(spectrogram.shape)

cnn = True
if cnn:
    # Labelling Females and Males
    specs = []
    y = []

    def scale_array(x, new_size):
        min_el = np.min(x)
        max_el = np.max(x)
        from PIL import Image
        y = np.array(Image.fromarray(x).resize(size=new_size))
        y = y / 255 * (max_el - min_el) + min_el
        return y

    for dir_name in os.listdir(male_dir):
        sample_rate, samples = wavfile.read(
            "./" + male_dir + "/" + dir_name + "/1.wav")
        frequencies, times, spectrogram = signal.spectrogram(
            samples, sample_rate)
        spectrogram = scale_array(spectrogram, (128, 128))
        specs.append(spectrogram)

        y.append(0)
    for dir_name in os.listdir(female_dir):
        sample_rate, samples = wavfile.read(
            "./" + female_dir + "/" + dir_name + "/1.wav")
        frequencies, times, spectrogram = signal.spectrogram(
            samples, sample_rate)
        spectrogram = scale_array(spectrogram, (128, 128))
        specs.append(spectrogram)
        y.append(1)

    num_filters = 100
    filter_size = 3
    pool_size = 2

    # Build the model.
    model = Sequential([
        Conv2D(num_filters, filter_size,
               activation='relu', input_shape=(128, 128, 1)),
        Conv2D(num_filters, filter_size,
               activation='relu', input_shape=(128, 128, 1)),
        MaxPooling2D(pool_size=pool_size),
        Flatten(),
        Dense(2, activation='softmax'),
    ])
    # Compile the model.
    model.compile(
        'adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    # Split training/testing data
    X_train, X_test, y_train, y_test = train_test_split(specs,
                                                        y,
                                                        test_size=0.15,
                                                        stratify=y)

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Normalize the images.
    X_train = (X_train / 255) - 0.5
    X_test = (X_test / 255) - 0.5
    # Reshape the images.
    X_train = np.expand_dims(X_train, axis=3)
    X_test = np.expand_dims(X_test, axis=3)

    # Train the model.
    model.fit(
        X_train,
        to_categorical(y_train),
        epochs=20,
        validation_split=0.3,
    )
    # Save the model to disk.
    model.save_weights('cnn.h5')
    # Load the model from disk later using:
    # model.load_weights('cnn.h5')
    # Predict on the first 5 test images.
    predictions = model.predict(X_test)
    # Print our model's predictions.
    predictions = np.argmax(predictions, axis=1)
    print(predictions)  # [7, 2, 1, 0, 4]
    labels = y_test
    # Check our predictions against the ground truths.
    print(labels)  # [7, 2, 1, 0, 4]
    numCorrect = 0
    for i in range(n):
        if predictions[i] == labels[i]:
            numCorrect += 1
    print((numCorrect / n) * 100)
