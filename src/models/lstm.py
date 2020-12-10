
import numpy as np
from result import Result
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.utils import to_categorical


def run_model(x_train, x_test, y_train, y_test):

    # scaler = MinMaxScaler(feature_range=(0, 1))
    # dataset = scaler.fit_transform(x_train)

    # reshape input to be [samples, time steps, features]
    # x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    # x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

    scaling = MinMaxScaler(feature_range=(-1, 1)).fit(x_train[0])
    new_x_train = []
    for x in x_train:
        new_x = scaling.transform(x)
        new_x_train.append(new_x)
    x_train = np.array(new_x_train)

    new_x_test = []
    for x in x_test:
        new_x = scaling.transform(x)
        new_x_test.append(new_x)
    x_test = np.array(new_x_test)

    model = Sequential()
    model.add(LSTM(8, input_shape=(300, 69)))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    model.fit(x_train, to_categorical(y_train), epochs=20, batch_size=1,
              verbose=1, validation_split=0.2)

    # make predictions
    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)

    return Result(result=y_pred, y_test=y_test)
