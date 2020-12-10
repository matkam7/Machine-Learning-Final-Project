
import numpy as np
import os
import pandas as pd

import config


def audio_to_fft(path):
    rate, data = wav.read(path)
    # wav file is mono.
    channel_1 = data[:]
    fourier = np.fft.fft(channel_1)
    return fourier


def extract_features():
    # Labelling Females and Males
    all_features = []
    y = []

    for i, filename in enumerate(os.listdir(config.male_dir_rnn)):
        # name = os.path.join(
        #     config.male_dir, filename)
        # fft = audio_to_fft(name)
        # fft.resize(30000)

        # ffts.append([fft.min().imag, fft.max().imag,
        #              np.std(fft).imag,
        #              np.mean(fft).imag,
        #              np.median(fft).imag
        #              ])
        # x = librosa.load(name, sr=None)

        df = pd.read_csv(os.path.join(config.male_dir_rnn, filename))
        features = df.values.tolist()
        # features = np.swapaxes(np.array(features), 0, 1)

        features = np.swapaxes(np.array(features), 0, 1)
        features = features[0:69]
        features = np.swapaxes(np.array(features), 0, 1)

        all_features.append(features[0:300])

        y.append(0)

    for i, filename in enumerate(os.listdir(config.female_dir_rnn)):
        # fft = audio_to_fft(os.path.join(
        #     config.female_dir, filename))
        # fft.resize(30000)

        # ffts.append([
        #     fft.min().real,
        #     fft.max().real,
        #     np.std(fft).imag,
        #     np.mean(fft).imag,
        #     np.median(fft).imag,
        # ])
        # ffts.append(fft)

        df = pd.read_csv(os.path.join(config.female_dir_rnn, filename))
        features = df.values.tolist()
        features = np.swapaxes(np.array(features), 0, 1)
        features = features[0:69]
        features = np.swapaxes(np.array(features), 0, 1)
        all_features.append(features[0:300])

        y.append(1)
    return y, all_features
