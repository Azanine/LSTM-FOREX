from Lstm_utils import *
import pandas as pd

FORMATED_DATA_DIR = "E:/forex data/output/"
start_date = '2019-01-01'
end_date = '2019-01-05'
resample_step = 1
# TODO:to predicted col prepei na ginei dictionary den ginetai douleia etsi....
predicted_col = 7
input_size = 200
# predict gap in seconds
predict_gap = 100
n_output = (int(input_size / 2)) * resample_step

logger = createLogger()

logger.info("data read started...")
# olo to dataset einai 31mil peripoy, gia tis 5 meres einai 280k peripoy
# dataset_raw = pd.read_csv(FORMATED_DATA_DIR + "final_raw.csv", nrows=10000, index_col='timestamp')
dataset_raw = pd.read_csv(FORMATED_DATA_DIR + "final_raw.csv", nrows=300000, index_col='timestamp')
logger.info("data read finished.")

dataset_raw = (dataset_raw.loc[start_date: end_date])
dataset_raw_columns = dataset_raw.columns

train_x, train_y, test_x, test_y = data_split(dataset_raw, input_size, predict_gap, predicted_col)
model = build_model(train_x, train_y)
print()