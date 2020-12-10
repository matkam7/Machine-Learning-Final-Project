import os
import config
from scipy import signal
from scipy.io import wavfile as wav
import numpy as np
from PIL import Image
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img

# Helper function to resize numpy 2d array to different size


def extract_features(desired_size=(128, 128)):
    def extract(label_num, dir):
        y = []
        imgs = []
        for filename in os.listdir(dir):
            y.append(label_num)
            im_frame = load_img(os.path.join(dir, filename),
                                color_mode="grayscale")
            np_frame = img_to_array(im_frame)
            imgs.append(np_frame)
        return y, imgs

    male_y, male_imgs = extract(0, config.male_dir_nn)
    female_y, female_imgs = extract(1, config.female_dir_nn)

    return male_y + female_y, male_imgs + female_imgs
