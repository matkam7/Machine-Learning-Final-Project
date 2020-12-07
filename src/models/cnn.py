import numpy as np
from sklearn.metrics import accuracy_score

from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping

from result import Result
import config


def run_model(
        x_train, x_test, y_train, y_test,
        num_filters=100, filter_size=3, pool_size=2):
    # Build the model.
    model = Sequential([
        Conv2D(num_filters, filter_size,
               activation='relu', input_shape=(config.cnn_input_size[1], config.cnn_input_size[0], 1)),
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

    # Normalize the images.
    x_train = (x_train / 255) - 0.5
    x_test = (x_test / 255) - 0.5
    # Reshape the images.
    x_train = np.expand_dims(x_train, axis=3)
    x_test = np.expand_dims(x_test, axis=3)

    # Stop as soon as validation loss goes significantly up
    # es = EarlyStopping(monitor='val_loss', mode='min', restore_best_weights=True, patience=20)

    # Train the model.
    model.fit(
        x_train,
        to_categorical(y_train),
        epochs=15,
        validation_split=0.2,
        # callbacks=[es]
    )
    # Save the model to disk.
    # model.save_weights('cnn.h5')
    # Load the model from disk later using:
    # model.load_weights('cnn.h5')
    # Predict on the first 5 test images.
    predictions = model.predict(x_test)
    # Print our model's predictions.
    predictions = np.argmax(predictions, axis=1)
    test_acc = accuracy_score(y_test, predictions)
    return Result(test_acc=test_acc)
