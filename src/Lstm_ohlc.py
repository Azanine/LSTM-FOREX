import json
from pathlib import Path

from keras.engine.saving import load_model
from keras.losses import mean_squared_error
from matplotlib import pyplot
from pandas import concat
import pandas as pd
import logging
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from pickle import dump, load
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import psutil
from collections import Counter
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils import check_array


# split a univariate dataset into train/test sets
def split_dataset(dataset_raw, dataset_ohlc, time_steps, scale = True):
    # TODO: spli dataset from dataframa nd not from numpyarrayes
    current_date = dataset_ohlc.index[int(dataset_ohlc.shape[0] * 0.8)]
    dataset_raw.index = pd.to_datetime(dataset_raw.index)
    raw_masked = dataset_raw.loc[(dataset_raw.index.second == 00.0)]
    raw_masked = raw_masked.loc[(raw_masked.index.minute == 00) | (raw_masked.index.minute == 5) |
                                (raw_masked.index.minute == 10) | (raw_masked.index.minute == 15) |
                                (raw_masked.index.minute == 20) | (raw_masked.index.minute == 25) |
                                (raw_masked.index.minute == 30) | (raw_masked.index.minute == 35) |
                                (raw_masked.index.minute == 40) | (raw_masked.index.minute == 45) |
                                (raw_masked.index.minute == 50) | (raw_masked.index.minute == 55)]
    # print(testin)
    # data_raw = dataset_raw.values
    # data_ohlc = dataset_ohlc.values

    # to timestemps spaei to data se fetes mikous timestemps kai platous oso ta features
    logger.info("split_database started")
    # train_size = int((len(data_raw)) * (0.8))
    # split into training and test
    # train, test = data_raw[:-train_size], data_raw[-train_size:]
    train_raw_df, test_raw_df = raw_masked[raw_masked.index <= current_date], \
                                raw_masked[raw_masked.index > current_date]

    train_ohlc_df, test_ohlc_df = dataset_ohlc[dataset_ohlc.index <= current_date], \
                                  dataset_ohlc[dataset_ohlc.index > current_date]
    # TODO:to train_ohlc_df exei ena value parapanw, einai to prwto

    # diagrafi prwtwn seiron etsi wste na mproei na kanei akrivi diairesh argotera
    train_raw_df = train_raw_df[(train_raw_df.shape[0] % time_steps):]
    train_ohlc_df = train_ohlc_df[(train_ohlc_df.shape[0] % time_steps):]

    test_raw_df = test_raw_df[(test_raw_df.shape[0] % time_steps):]
    test_ohlc_df = test_ohlc_df[(test_ohlc_df.shape[0] % time_steps):]

    train_parts = int(len(train_raw_df) / time_steps)
    test_parts = int(len(test_raw_df) / time_steps)

    if scale:
        scaler_raw = MinMaxScaler()
        scaler_ohlc = MinMaxScaler()
        scaler_raw = scaler_raw.fit(train_raw_df.values)
        scaler_ohlc = scaler_ohlc.fit(train_ohlc_df.values)
        dump(scaler_raw, open('E:/forex data/res/scaler_raw.pkl', 'wb'))
        dump(scaler_ohlc, open('E:/forex data/res/scaler_ohlc.pkl', 'wb'))

        train_raw_array = scaler_raw.transform(train_raw_df.values)
        train_ohlc_array = scaler_ohlc.transform(train_ohlc_df.values)
        test_raw_array = scaler_raw.transform(test_raw_df.values)
        test_ohlc_array = scaler_ohlc.transform(test_ohlc_df.values)


    # restructure into windows of "parts" size data data
    # TODO: an valw .values tote ola kala alla mipws prepei na kratisw to timestamp?
    train_raw_array = np.array(np.split(train_raw_array, train_parts))
    train_ohlc_array = np.array(np.split(train_ohlc_array, train_parts))
    test_raw_array = np.array(np.split(test_raw_array, test_parts))
    test_ohlc_array = np.array(np.split(test_ohlc_array, test_parts))



    return train_raw_array, train_ohlc_array, test_raw_array, test_ohlc_array


# convert history into inputs and outputs n_input sample size n_output vector output size
def to_supervised(train_raw, train_ohlc, n_input, n_out, resample_step, predicted_col):
    logger.info("to_supervised started")
    # flatten data giati kanei flatten ksana enw ta exei temaxisei thn split_database
    data_raw = train_raw.reshape((train_raw.shape[0] * train_raw.shape[1], train_raw.shape[2]))
    data_ohlc = train_ohlc.reshape((train_ohlc.shape[0] * train_ohlc.shape[1], train_ohlc.shape[2]))
    X, y = list(), list()
    # giati input start = 242???? no point nomizw...
    input_start = 1
    # step over the entire history one time step at a time
    # WALK FORWARD EDW
    for _ in range(len(data_raw)):
        # define the end of the input sequence
        input_end = input_start + n_input
        output_end = input_end + n_out
        # ensure we have enough data for this instance
        # You can also use this syntax (L[start:stop:step]):
        if output_end <= len(data_raw):
            X.append(data_ohlc[input_start:input_end:, :])
            y.append(data_raw[input_end:output_end:resample_step, predicted_col])
        # move along one time step
        input_start += 1
        # print()

    logger.info("to_supervised finished")
    return np.array(X), np.array(y)


def col_names_dictionary(cols):
    i = 1
    dictionary = {}
    for name in cols:
        value = "var" + str(i)
        dictionary[name] = value
        i += 1
    return dictionary


def set_correct_column_names(data, column_names):
    pass


def build_model(train_raw, train_ohlc, n_input, n_output, resample_step, predicted_col):
    # prepare data
    filepath = "E:/forex data/res/my_model.h5"
    if Path(filepath).is_file():
        model = load_model(filepath)
        logger.info("model saved")
    else:
        train_x, train_y = to_supervised(train_raw, train_ohlc, n_input, n_output, resample_step, predicted_col)
        # define parameters
        verbose, epochs, batch_size = 1, 45, 64
        n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
        # reshape output into [samples, timesteps, features]
        train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
        # define model
        model = Sequential()
        model.add(LSTM(200, activation='softmax', input_shape=(n_timesteps, n_features)))
        model.add(RepeatVector(n_outputs))
        model.add(LSTM(200, activation='softmax', return_sequences=True))
        model.add(TimeDistributed(Dense(100, activation='relu')))
        model.add(TimeDistributed(Dense(1)))
        model.compile(loss='mse', optimizer='adam')

        logger.info("model was built")
        # fit network
        loss_data = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
        model.save(filepath)
        json.dump(loss_data.history, open('E:/forex data/res/loss_data.json', 'w'))
        logger.info("model fit finished")
    return model


def evaluate_forecasts(actual, predicted):
    logger.info("evaluate forecast")
    # scores = list()
    s = 0
    sqrd = 0
    counter = 0
    # score = 0
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

                # s = abs(actual[row, col] - predicted[row, col])
                # print()
                # 3 και καλα ειναι η προμηθεια

                logger.info(" pre prof:" + str(prof))
                prof = prof * 100000
                prof = round(prof)
                profit_pips += prof
                logger.info("this trades prof:" + str(prof))
                logger.info("profit pips per trade" + str(profit_pips / counter))
                logger.info("total profit pips: " + str(profit_pips))

    # score = sqrt(sqrd / (actual.shape[0] * actual.shape[1]))

    # logger.info("profit pips:" + str(profit_pips))
    # todo: AUTO EINAI SE LATHOS THESI

    return profit_pips


def forecast(model, history, n_input):
    # flatten data
    # TODO: den xreiazetai na kanw ksana flatten eleos, giati den stelnw ton pinaka idi flatted?
    # logger.info("FORECAST ram usage: " + str(psutil.virtual_memory()))
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
    # logger.info("FORECAST ram usage: " + str(psutil.virtual_memory()))
    return yhat


def ev_calculator(current_price, predicted):
    file_path = "E:/forex data/res/ev_bins.json"
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


def evaluate_model(model, train_ohlc, train_raw, test_ohlc, test_raw, n_input, predicted_col):
    # fit model
    # history is a list of TIME_STEP data
    # to history pairnei ta train pou exei px shape (30, 100,32) kai to metatrepei se list:30 me kathei item
    # (100,32)
    history = [x for x in train_ohlc]
    # walk-forward validation over each week
    # to predictins einai mia lista me batches apo predictions
    predictions = list()
    logger.info("forecast started")
    # h loop kanei to eksis: kanei predict me tis teleutaies n_input times, kanei append to test sto
    # history, kai afoyu to kanei flatten again tha parei pali tis n_input teltuatais stigmes gia predict
    # prepei na gieni to idio kai me tis raw times
    # TODO: mipws prepei na ginei kai edw san walk forward??? kai isos shuffle in a unison
    scaler_raw = load(open('E:/forex data/res/scaler_raw.pkl', 'rb'))
    for i in range(len(test_ohlc)):
        # predict the week
        yhat_sequence = forecast(model, history, n_input)
        # yhat_sequence = yhat_sequence.reshape(yhat_sequence.shape[0] * yhat_sequence.shape[1])
        # store the predictions
        # inv_yhat =
        # inv_yhat = np.concatenate((test_raw[:, predicted_col], np.array(yhat_sequence).reshape(1, yhat_sequence.shape[0]), test_raw[:, (predicted_col+1):]))
        # inv_yhat = scaler_raw.inverse_transform(inv_yhat)
        predictions.append(yhat_sequence)
        # predictions.append(inv_yhat[:, predicted_col])
        # get real observation and add to history for predicting the next week
        history.append(test_ohlc[i, :])
    # evaluate predictions days for each week
    logger.info("forecast finished")
    # sto prediction oi probvlepeis einai kata mikos mia grammis
    predictions = np.array(predictions)
    scaler_raw = load(open('E:/forex data/res/scaler_raw.pkl', 'rb'))
    # predictions = scaler_raw.inverse_transform(predictions)
    # https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
    # invert transform
    # vazw 7 giati thelw na provlepsei thn teleutaia stili



    test_raw_shape0 = test_raw.shape[0]
    test_raw_shape1 = test_raw.shape[1]
    test_raw_shape2 = test_raw.shape[2]

    for i in range(test_raw_shape0):
        p1 = test_raw[i, :100, :predicted_col]
        p2 = predictions[i]
        # p3 = test_raw[i, :100, (predicted_col+1):]
        # TODO: na valw kai mia if na kanei concatenate kai to trito kommati an xreiazetai
        temp = scaler_raw.inverse_transform(np.concatenate((p1, p2), axis=1))[:, predicted_col]
        predictions[i] = temp.reshape(temp.shape[0], 1)

    # invert test raw
    test_raw = test_raw.reshape(test_raw_shape0*test_raw_shape1, test_raw_shape2)
    test_raw = scaler_raw.inverse_transform(test_raw)
    test_raw = test_raw.reshape(test_raw_shape0, test_raw_shape1, test_raw_shape2)
    tst = test_raw[:, :, predicted_col]
    # TODO: kapou edw tha mpei to filter
    predictions = predictions.reshape(predictions.shape[0],predictions.shape[1])

    result_recorder(tst, predictions)
    current_price = tst[0, 0]
    for i in range(predictions.shape[0]):
        ev_diction = ev_calculator(current_price, predictions[i])
        # predictions[i] = predictions_filter(predictions[i], ev_diction)

    score = evaluate_forecasts(tst, predictions)
    return score

def predictions_filter(predictions, ev_diction):
    threshold = 50
    for k in ev_diction:
        if ev_diction[k] < threshold:
            predictions[int(int(k)/5)] = 0
    return predictions

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
        total_count = len(delta)

        dict1[col * 5] = freq_counter

    update_bins(dict1)
    # print()


#
# dict1['total_count'] = total_count
# dict1['time_delta'] = col * 5


def update_bins(new_dicitoanry):
    file_path = "E:/forex data/res/ev_bins.json"
    my_file = Path(file_path)

    if my_file.is_file():
        with open(my_file, "r+", encoding="utf-8") as f:
            data = json.load(f)
            f.close()
            ti = new_dicitoanry[0]
            with open(my_file, "w") as f:
                new_counter_data = {}
                for k in data:
                    # TODO: edws mipos prepei na ginei check????? an kanei swsta add
                    counter_data = Counter(data[k])
                    new_counter_data[k] = counter_data + new_dicitoanry[int(k)]

                    # print()

                json.dump(new_counter_data, f, indent=4)

                f.close()
    else:
        with open(file_path, "w+") as f:
            json.dump(new_dicitoanry, f, indent=4)
            f.close()


# summarize scores
def summarize_scores(name, score, scores):
    s_scores = ', '.join(['%.1f' % s for s in scores])
    print('%s: [%.3f] %s' % (name, score, s_scores))


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')

file_handler = logging.FileHandler('E:/forex data/log/Lstm.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

FORMATED_DATA_DIR = "E:/forex data/output/"

logger.info("ram usage: " + str(psutil.virtual_memory()))

dataset_raw = pd.read_csv(FORMATED_DATA_DIR + "final_raw.csv", index_col='timestamp')
dataset_ohlc = pd.read_csv(FORMATED_DATA_DIR + "final_ohlc.csv", index_col='timestamp')

logger.info("data loaded")

start_date = '2019-01-01'
end_date = '20219-06-01'
# max gia mia imera, 1000 timesteps, 1000 n_input 500 n_output
TIME_STEPS = 200
resample_step = 1
# TODO:to predicted col prepei na ginei dictionary den ginetai douleia etsi....
predicted_col = 7
n_input = TIME_STEPS
n_output = (int(n_input / 2)) * resample_step

logger.info("ram usage: " + str(psutil.virtual_memory()))
dataset_raw = (dataset_raw.loc[start_date: end_date])
dataset_ohlc = (dataset_ohlc.loc[start_date: end_date])
logger.info("dataset masked, start date:" + start_date + "end date:" + end_date)

# dataset_ohlc = dataset_ohlc.astype('float32')

# dataset_raw = dataset_raw.astype('float32')

dataset_raw_columns = dataset_raw.columns
dataset_ohlc_columns = dataset_ohlc.columns
# ton scaler prepei na ton valw sto finaldatafroamter meta
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled = scaler.fit_transform(dataset)
# logger.info("data scaled")

logger.info("ram usage: " + str(psutil.virtual_memory()))
# timesteps should match input or output shape not sure yet..
train_raw, train_ohlc, test_raw, test_ohlc = split_dataset(dataset_raw, dataset_ohlc, time_steps=TIME_STEPS)
logger.info("data split to train and test")

model = build_model(train_raw, train_ohlc, n_input, n_output, resample_step, predicted_col)

score = evaluate_model(model, train_ohlc, train_raw, test_ohlc, test_raw, n_input, predicted_col)
