import tensorflow as tf

# Configuration variables
male_dir = "male"
female_dir = "female"
test_split = 0.4

# Configuration functions


def configure_tensorflow():
    # Setup Tensorflow
    config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(
        per_process_gpu_memory_fraction=0.8)
    )
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)