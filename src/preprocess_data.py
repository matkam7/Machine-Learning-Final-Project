from scipy.io import wavfile as wav
from scipy import signal
import numpy as np
import os
import pandas as pd
from pydub import AudioSegment
import csv
from pyAudioAnalysis import ShortTermFeatures as aFs
from pyAudioAnalysis import MidTermFeatures as aFm
from pyAudioAnalysis import audioBasicIO as aIO
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
import config
from PIL import Image
import librosa
import numpy
import skimage.io

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--nn', action='store_true',
                    help='Run all neural network models')
parser.add_argument('-m', '--ml', action='store_true',
                    help='Run machine learning models')
parser.add_argument('-r', '--rnn', action='store_true',
                    help='Run rnn model')
args = parser.parse_args()

if not (args.nn or args.ml or args.rnn):
    exit(0)


def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


def preprocess_data():
    def extract(label, out_dir):
        df = pd.read_csv(os.path.join("data", config.dataset) + ".csv")
        df = df[df.sex == label]
        for filename in df.filename:
            if args.ml:
                if not os.path.exists(os.path.join(out_dir, filename) + ".csv"):
                    try:
                        sound = AudioSegment.from_mp3(os.path.join(
                            "./data", "recordings", filename) + ".mp3")
                        sound.export("tmp.wav", format="wav")

                        # signal, sampling rate
                        fs, s = aIO.read_audio_file("tmp.wav")

                        # get all mid-term features, returning an array of features
                        # Look at the first 10 seconds
                        mid_term_window = 10
                        mt, st, mt_n = aFm.mid_feature_extraction(s, fs, mid_term_window * fs, mid_term_window * fs,
                                                                  0.05 * fs, 0.05 * fs)
                        # Mid-Term Features:
                        # 0:zcr_mean
                        # 1:energy_mean
                        # 2:energy_entropy_mean
                        # 3:spectral_centroid_mean
                        # 4:spectral_spread_mean
                        # 5:spectral_entropy_mean
                        # 6:spectral_flux_mean
                        # 7:spectral_rolloff_mean
                        # 8:mfcc_1_mean
                        # 9:mfcc_2_mean
                        # 10:mfcc_3_mean
                        # 11:mfcc_4_mean
                        # 12:mfcc_5_mean
                        # 13:mfcc_6_mean
                        # 14:mfcc_7_mean
                        # 15:mfcc_8_mean
                        # 16:mfcc_9_mean
                        # 17:mfcc_10_mean
                        # 18:mfcc_11_mean
                        # 19:mfcc_12_mean
                        # 20:mfcc_13_mean
                        # 21:chroma_1_mean
                        # 22:chroma_2_mean
                        # 23:chroma_3_mean
                        # 24:chroma_4_mean
                        # 25:chroma_5_mean
                        # 26:chroma_6_mean
                        # 27:chroma_7_mean
                        # 28:chroma_8_mean
                        # 29:chroma_9_mean
                        # 30:chroma_10_mean
                        # 31:chroma_11_mean
                        # 32:chroma_12_mean
                        # 33:chroma_std_mean
                        # 34:delta zcr_mean
                        # 35:delta energy_mean
                        # 36:delta energy_entropy_mean
                        # 37:delta spectral_centroid_mean
                        # 38:delta spectral_spread_mean
                        # 39:delta spectral_entropy_mean
                        # 40:delta spectral_flux_mean
                        # 41:delta spectral_rolloff_mean
                        # 42:delta mfcc_1_mean
                        # 43:delta mfcc_2_mean
                        # 44:delta mfcc_3_mean
                        # 45:delta mfcc_4_mean
                        # 46:delta mfcc_5_mean
                        # 47:delta mfcc_6_mean
                        # 48:delta mfcc_7_mean
                        # 49:delta mfcc_8_mean
                        # 50:delta mfcc_9_mean
                        # 51:delta mfcc_10_mean
                        # 52:delta mfcc_11_mean
                        # 53:delta mfcc_12_mean
                        # 54:delta mfcc_13_mean
                        # 55:delta chroma_1_mean
                        # 56:delta chroma_2_mean
                        # 57:delta chroma_3_mean
                        # 58:delta chroma_4_mean
                        # 59:delta chroma_5_mean
                        # 60:delta chroma_6_mean
                        # 61:delta chroma_7_mean
                        # 62:delta chroma_8_mean
                        # 63:delta chroma_9_mean
                        # 64:delta chroma_10_mean
                        # 65:delta chroma_11_mean
                        # 66:delta chroma_12_mean
                        # 67:delta chroma_std_mean
                        # 68:zcr_std
                        # 69:energy_std
                        # 70:energy_entropy_std
                        # 71:spectral_centroid_std
                        # 72:spectral_spread_std
                        # 73:spectral_entropy_std
                        # 74:spectral_flux_std
                        # 75:spectral_rolloff_std
                        # 76:mfcc_1_std
                        # 77:mfcc_2_std
                        # 78:mfcc_3_std
                        # 79:mfcc_4_std
                        # 80:mfcc_5_std
                        # 81:mfcc_6_std
                        # 82:mfcc_7_std
                        # 83:mfcc_8_std
                        # 84:mfcc_9_std
                        # 85:mfcc_10_std
                        # 86:mfcc_11_std
                        # 87:mfcc_12_std
                        # 88:mfcc_13_std
                        # 89:chroma_1_std
                        # 90:chroma_2_std
                        # 91:chroma_3_std
                        # 92:chroma_4_std
                        # 93:chroma_5_std
                        # 94:chroma_6_std
                        # 95:chroma_7_std
                        # 96:chroma_8_std
                        # 97:chroma_9_std
                        # 98:chroma_10_std
                        # 99:chroma_11_std
                        # 100:chroma_12_std
                        # 101:chroma_std_std
                        # 102:delta zcr_std
                        # 103:delta energy_std
                        # 104:delta energy_entropy_std
                        # 105:delta spectral_centroid_std
                        # 106:delta spectral_spread_std
                        # 107:delta spectral_entropy_std
                        # 108:delta spectral_flux_std
                        # 109:delta spectral_rolloff_std
                        # 110:delta mfcc_1_std
                        # 111:delta mfcc_2_std
                        # 112:delta mfcc_3_std
                        # 113:delta mfcc_4_std
                        # 114:delta mfcc_5_std
                        # 115:delta mfcc_6_std
                        # 116:delta mfcc_7_std
                        # 117:delta mfcc_8_std
                        # 118:delta mfcc_9_std
                        # 119:delta mfcc_10_std
                        # 120:delta mfcc_11_std
                        # 121:delta mfcc_12_std
                        # 122:delta mfcc_13_std
                        # 123:delta chroma_1_std
                        # 124:delta chroma_2_std
                        # 125:delta chroma_3_std
                        # 126:delta chroma_4_std
                        # 127:delta chroma_5_std
                        # 128:delta chroma_6_std
                        # 129:delta chroma_7_std
                        # 130:delta chroma_8_std
                        # 131:delta chroma_9_std
                        # 132:delta chroma_10_std
                        # 133:delta chroma_11_std
                        # 134:delta chroma_12_std
                        # 135:delta chroma_std_std
                        features = {mt_n[i]: [mt[i][0]]
                                    for i in range(len(mt_n))}
                        ftDf = pd.DataFrame.from_dict(features)
                        ftDf.to_csv(os.path.join(out_dir, filename) + ".csv")
                    except Exception as e:
                        print(e)
            elif args.nn:
                if not os.path.exists(os.path.join(out_dir, filename) + ".png"):
                    try:
                        sound = AudioSegment.from_mp3(os.path.join(
                            "./data", "recordings", filename) + ".mp3")
                        sound.export("tmp.wav", format="wav")

                        y, sr = librosa.load(
                            "tmp.wav", offset=2.0, duration=8.0, sr=22050)

                        # extract a fixed length window

                        # number of samples per time-step in spectrogram
                        hop_length = 512
                        # number of bins in spectrogram. Height of image
                        n_mels = config.cnn_input_size[0]
                        # number of time-steps. Width of image
                        time_steps = config.cnn_input_size[1]
                        # starting at beginning
                        start_sample = 0
                        length_samples = time_steps*hop_length
                        window = y[start_sample:start_sample+length_samples]

                        # use log-melspectrogram
                        mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                                              n_fft=hop_length*2, hop_length=hop_length)
                        # add small number to avoid log(0)
                        mels = np.log(mels + 1e-9)

                        # min-max scale to fit inside 8-bit range
                        img = scale_minmax(mels, 0, 255).astype(numpy.uint8)
                        # put low frequencies at the bottom in image
                        img = np.flip(img, axis=0)
                        img = 255-img  # invert. make black==more energy

                        # save as PNG
                        skimage.io.imsave(os.path.join(
                            out_dir, filename) + ".png", img)
                    except Exception as e:
                        print(e)
                pass
            elif args.rnn:
                if not os.path.exists(os.path.join(out_dir, filename) + ".csv"):
                    try:
                        sound = AudioSegment.from_mp3(os.path.join(
                            "./data", "recordings", filename) + ".mp3")
                        sound.export("tmp.wav", format="wav")

                        # signal, sampling rate
                        fs, s = aIO.read_audio_file("tmp.wav")

                        # get all shoart-term features, returning an array of features
                        # extract short-term features using a 50msec non-overlapping windows
                        duration = len(s) / float(fs)
                        win, step = 0.050, 0.050
                        [f, fn] = aFs.feature_extraction(s, fs, int(fs * win),
                                                         int(fs * step))
                        print(
                            f'{f.shape[1]} frames, {f.shape[0]} short-term features')
                        # Short-Term Features:
                        # 0:zcr
                        # 1:energy
                        # 2:energy_entropy
                        # 3:spectral_centroid
                        # 4:spectral_spread
                        # 5:spectral_entropy
                        # 6:spectral_flux
                        # 7:spectral_rolloff
                        # 8:mfcc_1
                        # 9:mfcc_2
                        # 10:mfcc_3
                        # 11:mfcc_4
                        # 12:mfcc_5
                        # 13:mfcc_6
                        # 14:mfcc_7
                        # 15:mfcc_8
                        # 16:mfcc_9
                        # 17:mfcc_10
                        # 18:mfcc_11
                        # 19:mfcc_12
                        # 20:mfcc_13
                        # 21:chroma_1
                        # 22:chroma_2
                        # 23:chroma_3
                        # 24:chroma_4
                        # 25:chroma_5
                        # 26:chroma_6
                        # 27:chroma_7
                        # 28:chroma_8
                        # 29:chroma_9
                        # 30:chroma_10
                        # 31:chroma_11
                        # 32:chroma_12
                        # 33:chroma_std
                        # 34:delta zcr
                        # 35:delta energy
                        # 36:delta energy_entropy
                        # 37:delta spectral_centroid
                        # 38:delta spectral_spread
                        # 39:delta spectral_entropy
                        # 40:delta spectral_flux
                        # 41:delta spectral_rolloff
                        # 42:delta mfcc_1
                        # 43:delta mfcc_2
                        # 44:delta mfcc_3
                        # 45:delta mfcc_4
                        # 46:delta mfcc_5
                        # 47:delta mfcc_6
                        # 48:delta mfcc_7
                        # 49:delta mfcc_8
                        # 50:delta mfcc_9
                        # 51:delta mfcc_10
                        # 52:delta mfcc_11
                        # 53:delta mfcc_12
                        # 54:delta mfcc_13
                        # 55:delta chroma_1
                        # 56:delta chroma_2
                        # 57:delta chroma_3
                        # 58:delta chroma_4
                        # 59:delta chroma_5
                        # 60:delta chroma_6
                        # 61:delta chroma_7
                        # 62:delta chroma_8
                        # 63:delta chroma_9
                        # 64:delta chroma_10
                        # 65:delta chroma_11
                        # 66:delta chroma_12
                        # 67:delta chroma_std
                        features = {fn[i]: f[i]
                                    for i in range(len(fn))}
                        ftDf = pd.DataFrame.from_dict(features)
                        ftDf.to_csv(os.path.join(out_dir, filename) + ".csv")
                    except Exception as e:
                        print(e)
            else:
                pass

    extract("female", "female_out" if args.ml else "female_out_nn" if args.nn else "female_out_rnn")
    extract(
        "male", "male_out" if args.ml else "male_out_nn" if args.nn else "male_out_rnn")


if __name__ == "__main__":
    preprocess_data()
