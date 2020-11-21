import os
import config
from scipy import signal
from scipy.io import wavfile as wav
import numpy as np
from PIL import Image

# Helper function to resize numpy 2d array to different size


def scale_array(x, new_size):
    min_el = np.min(x)
    max_el = np.max(x)
    y = np.array(Image.fromarray(x).resize(size=new_size))
    y = y / 255 * (max_el - min_el) + min_el
    return y


def extract_features(desired_size=(128, 128)):
    specs = []
    y = []

    for dir_name in os.listdir(config.male_dir):
        sample_rate, samples = wav.read(
            "./" + config.male_dir + "/" + dir_name + "/1.wav")
        frequencies, times, spectrogram = signal.spectrogram(
            samples, sample_rate)
        spectrogram = scale_array(spectrogram, (128, 128))
        specs.append(spectrogram)
        y.append(0)

    for dir_name in os.listdir(config.female_dir):
        sample_rate, samples = wav.read(
            "./" + config.female_dir + "/" + dir_name + "/1.wav")
        frequencies, times, spectrogram = signal.spectrogram(
            samples, sample_rate)
        spectrogram = scale_array(spectrogram, (128, 128))
        specs.append(spectrogram)
        y.append(1)

    return y, specs
