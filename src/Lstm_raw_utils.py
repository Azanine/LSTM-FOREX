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
from keras.layers import Dense
from keras.layers import LSTM

from sklearn.preprocessing import MinMaxScaler


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


def split_dataset(dataset_raw, time_steps, scale = True):
    logger = createLogger()
    # TODO: spli dataset from dataframa nd not from numpyarrayes
    current_date = dataset_raw.index[int(dataset_raw.shape[0] * 0.8)]
    dataset_raw.index = pd.to_datetime(dataset_raw.index)

    # to timestemps spaei to data se fetes mikous timestemps kai platous oso ta features
    logger.info("split_database started")
    train_raw_df, test_raw_df = dataset_raw[dataset_raw.index <= current_date],\
                                dataset_raw[dataset_raw.index > current_date]

    # diagrafi prwtwn seiron etsi wste na mproei na kanei akrivi diairesh argotera
    train_raw_df = train_raw_df[(train_raw_df.shape[0] % time_steps):]
    test_raw_df = test_raw_df[(test_raw_df.shape[0] % time_steps):]

    train_parts = int(len(train_raw_df) / time_steps)
    test_parts = int(len(test_raw_df) / time_steps)

    if scale:
        scaler_raw = MinMaxScaler()
        scaler_raw = scaler_raw.fit(train_raw_df.values)
        dump(scaler_raw, open('E:/forex data/res/clean_raw/scaler_raw.pkl', 'wb'))
        train_raw_array = scaler_raw.transform(train_raw_df.values)
        test_raw_array = scaler_raw.transform(test_raw_df.values)

    # restructure into windows of "parts" size data data
    train_raw_array = np.array(np.split(train_raw_array, train_parts))
    test_raw_array = np.array(np.split(test_raw_array, test_parts))

    return train_raw_array, test_raw_array


def to_supervised(train_raw, n_input, n_out, resample_step, predicted_col):
    logger = createLogger()
    logger.info("to_supervised started")
    # flatten data giati kanei flatten ksana enw ta exei temaxisei thn split_database, giati ksana???
    # prepei na erxontai flat!
    train_raw = train_raw.reshape((train_raw.shape[0] * train_raw.shape[1], train_raw.shape[2]))

    # step over the entire history one time step at a time
    x, y = walk_forward(train_raw, 1, n_input, n_out, resample_step, predicted_col)

    logger.info("to_supervised finished")

    return np.array(x), np.array(y)


def walk_forward(array, input_start, n_input, n_out, resample_step, predicted_col):
    # TODO:na thn ftiaksw etsi wsta na ftiaxnei input out arrays apo diaforetika datasets
    x, y = list(), list()
    for _ in range(len(array)):
        # define the end of the input sequence
        input_end = input_start + n_input
        output_end = input_end + n_out
        # ensure we have enough data for this instance
        # You can also use this syntax (L[start:stop:step]):
        if output_end + 1 <= len(array):
            x.append(array[input_start:input_end:, :])
            y.append(array[input_end:output_end:resample_step, predicted_col])
            # use below line of code for many to one model
            # y.append(array[output_end, predicted_col])
            print()
        # move along one time step
        input_start += 1

    return x, y


def build_model(train_raw, n_input, n_output, resample_step, predicted_col):
    logger = createLogger()
    # prepare data
    filepath = "E:/forex data/res/clean_raw/my_model_raw.h5"
    if Path(filepath).is_file():
        model = load_model(filepath)
        logger.info("model loaded")
        train_x, train_y = to_supervised(train_raw, n_input, n_output, resample_step, predicted_col)
    else:
        train_x, train_y = to_supervised(train_raw, n_input, n_output, resample_step, predicted_col)
        # define parameters
        verbose, epochs, batch_size = 1, 1, 64
        n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
        # reshape output into [samples, timesteps, features]
        train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
        # define model
        model = Sequential()
        model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
        model.add(RepeatVector(n_outputs))
        model.add(LSTM(200, activation='relu', return_sequences=True))
        model.add(TimeDistributed(Dense(100, activation='relu')))
        model.add(TimeDistributed(Dense(1)))
        model.compile(loss='mse', optimizer='adam')

        logger.info("model was built")
        # fit network
        loss_data = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
        model.save(filepath)
        json.dump(loss_data.history, open('E:/forex data/res/clean_raw/loss_data.json', 'w'))
        logger.info("model fit finished")
    return model


def evaluate_model(model, train_raw, test_raw, n_input, predicted_col):
    logger = createLogger()
    # fit model
    # history is a list of TIME_STEP data
    # to history pairnei ta train pou exei px shape (30, 100,32) kai to metatrepei se list:30 me kathei item
    # (100,32)
    history = [x for x in test_raw]
    # walk-forward validation over each week
    # to predictins einai mia lista me batches apo predictions
    predictions = list()
    logger.info("forecast started")
    # h loop kanei to eksis: kanei predict me tis teleutaies n_input times, kanei append to test sto
    # history, kai afoyu to kanei flatten again tha parei pali tis n_input teltuatais stigmes gia predict
    # prepei na gieni to idio kai me tis raw times
    scaler_raw = load(open('E:/forex data/res/clean_raw/scaler_raw.pkl', 'rb'))
    for i in range(len(test_raw)):
        yhat_sequence = forecast(model, history, n_input)
        predictions.append(yhat_sequence)
        # get real observation and add to history for predicting the next week
        history.append(test_raw[i, :])
    # evaluate predictions days for each week
    logger.info("forecast finished")
    # sto prediction oi probvlepeis einai kata mikos mia grammis
    predictions = np.array(predictions)
    scaler_raw = load(open('E:/forex data/res/clean_raw/scaler_raw.pkl', 'rb'))

    test_raw_shape0 = test_raw.shape[0]
    test_raw_shape1 = test_raw.shape[1]
    test_raw_shape2 = test_raw.shape[2]

    for i in range(test_raw_shape0):
        p1 = test_raw[i, :100, :predicted_col]
        p2 = predictions[i]
        # to apo katw einai gia na kanei concatente ka to trito kommati an xreiazetai
        # p3 = test_raw[i, :100, (predicted_col+1):]
        temp = scaler_raw.inverse_transform(np.concatenate((p1, p2), axis=1))[:, predicted_col]
        predictions[i] = temp.reshape(temp.shape[0], 1)

    # invert test raw
    test_raw = test_raw.reshape(test_raw_shape0*test_raw_shape1, test_raw_shape2)
    test_raw = scaler_raw.inverse_transform(test_raw)
    test_raw = test_raw.reshape(test_raw_shape0, test_raw_shape1, test_raw_shape2)
    tst = test_raw[:, :, predicted_col]

    predictions = predictions.reshape(predictions.shape[0],predictions.shape[1])
    result_recorder(tst, predictions)
    current_price = tst[0, 0]
    for i in range(predictions.shape[0]):
        ev_diction = ev_calculator(current_price, predictions[i])
        # IF YOU DONT WANT EV EVALUATION COMMENT OUT LINVE BELLOW
        # predictions[i] = predictions_filter(predictions[i], ev_diction)

    score = evaluate_forecasts(tst, predictions)
    return score


def evaluate_forecasts(actual, predicted):
    logger = createLogger()
    logger.info("evaluate forecast")
    counter = 0
    profit_pips = 0
    prof = 0
    for row in range(actual.shape[0]):
        for col in range(predicted.shape[1]):
            # if abs(actual[row, 0] - predicted[row, col]) > 0.00100:
            if predicted[row, col] != 0:
                logger.info(
                    "now actual:" + str(actual[row, 0]) + " actual then:" + str(actual[row, col]) + " predicted:" + str(
                        predicted[row, col]))
                counter += 1
                if predicted[row, col] > actual[row, 0]:
                    prof = (actual[row, col] - actual[row, 0])
                else:
                    prof = (actual[row, 0] - actual[row, col])

                # 3 και καλα ειναι η προμηθεια
                logger.info(" pre prof:" + str(prof))
                prof = prof * 100000
                prof = round(prof)
                profit_pips += prof
                logger.info("this trades prof:" + str(prof))
                logger.info("profit pips per trade" + str(profit_pips / counter))
                logger.info("total profit pips: " + str(profit_pips))

    return profit_pips


def ev_calculator(current_price, predicted):
    file_path = "E:/forex data/res/clean_raw/ev_bins.json"
    current_price = current_price*100000
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)
        ev_dict = {}
        for k in data:
            local_ev = 0
            total = sum(Counter(data[k]).values())
            pred = int((predicted[int(int(k)/5)])*100000)
            for l in data[k]:
                dt = data[k][l]/total
                prob = round(dt, 4)
                # local ev gia thn k xroniki stigmi
                local_ev += (pred - current_price + data[k][l])*prob
            ev_dict[k] = local_ev
    f.close()
    return ev_dict


def result_recorder(actual, predicted):
    delta = []
    dict1 = {}
    for col in range(predicted.shape[1]):
        delta = []
        for row in range(actual.shape[0]):
            pr = int(round(predicted[row, col], 5) * 100000)
            ac = int(round(actual[row, col], 5) * 100000)
            dif = (pr - ac)
            delta.append(dif)

        freq_counter = Counter(delta)
        # total_count = len(delta)

        dict1[col * 5] = freq_counter

    update_bins(dict1)


def update_bins(new_dictionary):
    file_path = "E:/forex data/res/clean_raw/ev_bins.json"
    my_file = Path(file_path)

    if my_file.is_file():
        with open(my_file, "r+", encoding="utf-8") as f:
            data = json.load(f)
            f.close()
            ti = new_dictionary[0]
            with open(my_file, "w") as f:
                new_counter_data = {}
                for k in data:
                    # TODO: edws mipos prepei na ginei check????? an kanei swsta add
                    counter_data = Counter(data[k])
                    new_counter_data[k] = counter_data + new_dictionary[int(k)]
                json.dump(new_counter_data, f, indent=4)
                f.close()
    else:
        with open(file_path, "w+") as f:
            json.dump(new_dictionary, f, indent=4)
            f.close()


def forecast(model, history, n_input):
    # flatten data
    data = np.array(history)
    data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
    # retrieve last observations for input data
    input_x = data[-n_input:, :]
    # reshape into [1, n_input, n]
    # kanei reshape gia na dwsei to swsto shape kai kanei predict
    inx = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
    # forecast the next week
    yhat = model.predict(inx, verbose=0)
    # we only want the vector forecast
    yhat = yhat[0]

    return yhat
