import tensorflow as tf
import os

tf_device = "CPU"

if tf_device == "GPU":
    pass
elif tf_device == "CPU":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


# Configuration variables
dataset = "speakers_all"

male_dir = "male_out"
female_dir = "female_out"
male_dir_nn = "male_out_nn"
female_dir_nn = "female_out_nn"
male_dir_rnn = "male_out_rnn"
female_dir_rnn = "female_out_rnn"

# CNN
cnn_input_size = (345, 256)

# Configuration functions


def configure_tensorflow():
    # Setup Tensorflow
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(
                logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
