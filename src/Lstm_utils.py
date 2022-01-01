from collections import Counter
import json
from pathlib import Path
from pickle import dump, load
import numpy as np
import pandas as pd
import logging
from keras.engine.saving import load_model
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import LSTM

from sklearn.preprocessing import MinMaxScaler

FILE_PATHNAME = "E:/forex data/many_to_one"

def createLogger():

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')

    file_handler = logging.FileHandler('E:/forex data/log/Lstm.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    if not logger.hasHandlers():
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger


def walk_forward(array, input_size, predict_gap, predicted_col, input_start=0):
    logger = createLogger()
    logger.info("walk forward started.")

    x, y = list(), list()

    # array = array.values
    for _ in range(len(array)):
        # define the end of the input sequence
        input_end = input_start + input_size
        target_price = input_end + predict_gap
        # ensure we have enough data for this instance
        # You can also use this syntax (L[start:stop:step]):
        if target_price <= len(array) - 1:
            x.append(array[input_start:input_end:, :])
            y.append(array[target_price, predicted_col])
        # move along one time step
        input_start += 1

    logger.info("walk forward finished.")
    return x, y


def data_split(dataset_raw, input_size, predict_gap, predicted_col, split_percentage=0.8, scale=True):
    logger = createLogger()
    logger.info("data split started.")

    split_date = dataset_raw.index[int(dataset_raw.shape[0] * split_percentage)]
    dataset_raw.index = pd.to_datetime(dataset_raw.index)
    if scale:
        scaler = MinMaxScaler()
        scaler = scaler.fit(dataset_raw.values)
        dump(scaler, open(FILE_PATHNAME + '/res/scaler_raw.pkl', 'wb'))
        dataset_raw = scaler.transform(dataset_raw.values)

    x, y = walk_forward(dataset_raw, input_size, predict_gap, predicted_col)

    split_index = int(len(x) * split_percentage)
    train_x, train_y, test_x, test_y = x[:split_index], y[:split_index], x[split_index:], y[split_index:]
    train_x, train_y, test_x, test_y = np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)
    train_y, test_y = train_y.reshape((train_y.shape[0]), 1), test_y.reshape((test_y.shape[0], 1))

    logger.info("data split finished.")
    return train_x, train_y, test_x, test_y

def build_model(train_x, train_y):
    logger = createLogger()
    model_path = FILE_PATHNAME + '/res/lstm_mto.h5'
    if Path(model_path).is_file():
        model = load_model(model_path)
        logger.info("model loaded.")
    else:
        verbose, epochs, batch_size = 1, 14, 512
        model = Sequential()
        model.add(LSTM(200, return_sequences=True, activation='softmax', input_shape=(train_x.shape[1], train_x.shape[2])))
        model.add(LSTM(200, activation='softmax', return_sequences=False))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam')
        print(model.summary())

        logger.info("model was built")
        loss_data = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
        model.save(model_path)
        json.dump(loss_data.history, open(FILE_PATHNAME + '/res/loss_data.json', 'w'))
        logger.info("model fit finished")

    return model






























