import numpy as np
from sklearn.metrics import accuracy_score

from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import sklearn

from result import Result
import config
from collections import Counter


def run_model(
        x_train, x_test, y_train, y_test,
        num_filters=100, filter_size=3, pool_size=2):
    # Build the model.
    model = Sequential([
        Conv2D(20, 3,
               activation='relu', input_shape=(config.cnn_input_size[1], config.cnn_input_size[0], 1)),
        MaxPooling2D(pool_size=2),
        Conv2D(60, filter_size, activation='relu'),
        MaxPooling2D(pool_size=2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(2, activation='softmax'),
    ])

    # Compile the model.
    model.compile(
        'adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    # Normalize the images.
    x_train /= 255.0
    x_test /= 255.0
    # Stop as soon as validation loss goes significantly up
    # es = EarlyStopping(monitor='val_loss', mode='min', restore_best_weights=True, patience=20)

    # Train the model.
    model.fit(
        x_train,
        to_categorical(y_train),
        epochs=9,
        validation_split=0.2,
        shuffle=True
    )
    # Save the model to disk.
    # model.save_weights('cnn.h5')
    # Load the model from disk later using:
    # model.load_weights('cnn.h5')
    # Predict on the first 5 test images.
    predictions = model.predict(x_test)
    # Print our model's predictions.
    predictions = np.argmax(predictions, axis=1)
    return Result(result=predictions, y_test=y_test)
