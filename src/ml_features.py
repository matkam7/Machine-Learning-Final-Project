from scipy.io import wavfile as wav
import numpy as np
import os

import config


def audio_to_fft(path):
    rate, data = wav.read(path)
    # wav file is mono.
    channel_1 = data[:]
    fourier = np.fft.fft(channel_1)
    return fourier


def extract_features():
    # Labelling Females and Males
    ffts = []
    y = []

    for dir_name in os.listdir(config.male_dir):
        fft = audio_to_fft("./" + config.male_dir + "/" + dir_name + "/1.wav")
        fft.resize(30000)
        ffts.append(fft)
        y.append(0)
    for dir_name in os.listdir(config.female_dir):
        fft = audio_to_fft("./" + config.female_dir +
                           "/" + dir_name + "/1.wav")
        fft.resize(30000)
        ffts.append(fft)
        y.append(1)
    # TODO FIX LATER
    ffts = [e.real for e in ffts]
    return y, ffts
