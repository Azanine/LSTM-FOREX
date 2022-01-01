from Lstm_raw_utils import *
import pandas as pd

FORMATED_DATA_DIR = "E:/forex data/output/"
start_date = '2019-01-01'
end_date = '2019-01-05'
TIME_STEPS = 200
resample_step = 1
# TODO:to predicted col prepei na ginei dictionary den ginetai douleia etsi....
predicted_col = 7
n_input = TIME_STEPS
n_output = (int(n_input / 2)) * resample_step

logger = createLogger()

logger.info("data read started...")
dataset_raw = pd.read_csv(FORMATED_DATA_DIR + "final_raw.csv", nrows=10000, index_col='timestamp')
# dataset_raw = pd.read_csv(FORMATED_DATA_DIR + "final_raw.csv", index_col='timestamp')
logger.info("data read finished.")

dataset_raw = (dataset_raw.loc[start_date: end_date])
dataset_raw_columns = dataset_raw.columns

train_raw, test_raw = split_dataset(dataset_raw, time_steps=TIME_STEPS)

model = build_model(train_raw, n_input, n_output, resample_step, predicted_col)
logger.info("model built")

score = evaluate_model(model, train_raw, test_raw, n_input, predicted_col)



