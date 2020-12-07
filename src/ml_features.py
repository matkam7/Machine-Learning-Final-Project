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
        for filename in os.listdir(os.path.join(config.male_dir, dir_name)):
            name = os.path.join(
                config.male_dir, dir_name, filename)
            fft = audio_to_fft(name)
            # fft.resize(30000)

            ffts.append([fft.min().imag, fft.max().imag,
                         np.std(fft).imag,
                         np.mean(fft).imag,
                         np.median(fft).imag
                         ])
            # x = librosa.load(name, sr=None)
            y.append(0)

    for dir_name in os.listdir(config.female_dir):
        for filename in os.listdir(os.path.join(config.female_dir, dir_name)):
            fft = audio_to_fft(os.path.join(
                config.female_dir, dir_name, filename))
            # fft.resize(30000)

            ffts.append([
                fft.min().real,
                fft.max().real,
                np.std(fft).imag,
                np.mean(fft).imag,
                np.median(fft).imag,
            ])
            # ffts.append(fft)
            y.append(1)
    # TODO FIX LATER
    # ffts = [e.imag for e in ffts]
    return y, ffts
