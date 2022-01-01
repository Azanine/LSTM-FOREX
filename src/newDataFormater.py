import os
import logging
import pandas as pd


currencyList = []
# logging.basicConfig(filename='E:/forex data/log/Datafromater.log', level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')

file_handler = logging.FileHandler('E:/forex data/log/Datafromater.log')
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

RESAMPLE_STEP = '5Min'
RAW_DATA_DIR = "E:/forex data/new/"
FORMATED_DATA_DIR = "E:/forex data/output/"

for file in os.listdir(RAW_DATA_DIR):
    if file.endswith(".csv"):

        fileCurrencyName = file.split("_")[-3]
        fileYear = (file.split("_")[-1]).split(".")[0][:4]
        fileMonth = (file.split("_")[-1]).split(".")[0][4:6]
        outputFileDirName = FORMATED_DATA_DIR + fileCurrencyName + '_' + fileYear + fileMonth
        logger.info("new file found:" + "file" + outputFileDirName)

        if fileCurrencyName not in currencyList:
            currencyList.append(fileCurrencyName)

        # if file hasnt alread been formated then format it
        if os.path.isfile(outputFileDirName):
            logger.info("file " + outputFileDirName + " already been formated")

        else:
            logging.info("file to format is " + outputFileDirName)
            data = pd.read_csv(RAW_DATA_DIR + file, names=["timestamp",
                                                           fileCurrencyName + "_bid",
                                                           fileCurrencyName + "_ask", "etc"])
            df = pd.DataFrame(data).drop("etc", axis=1)

            df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y%m%d %H%M%S%f')
            # create raw data dataframe
            df_raw = round(df.resample('1S', on='timestamp').mean(), 5)

            df = df.set_index('timestamp')
            # resample per RESAMPLE_STEP and round
            df_bid_ohlc = round(df[fileCurrencyName + "_bid"].resample(RESAMPLE_STEP).ohlc(), 5)
            df_ask_ohlc = round(df[fileCurrencyName + "_ask"].resample(RESAMPLE_STEP).ohlc(), 5)
            # change column names
            df_bid_ohlc = df_bid_ohlc.rename(columns={'open': fileCurrencyName + '_bid' + '_open',
                                                      'high': fileCurrencyName + '_bid' + '_high',
                                                      'low': fileCurrencyName + '_bid' + '_low',
                                                      'close': fileCurrencyName + '_bid' + '_close'
                                                      })
            df_ask_ohlc = df_ask_ohlc.rename(columns={'open': fileCurrencyName + '_ask' + '_open',
                                                      'high': fileCurrencyName + '_ask' + '_high',
                                                      'low': fileCurrencyName + '_ask' + '_low',
                                                      'close': fileCurrencyName + '_ask' + '_close'
                                                      })
            # concat to final dataframe ready to export to csv
            df_ohlc = pd.concat([df_bid_ohlc, df_ask_ohlc], axis=1)

            df_raw.to_csv(outputFileDirName + "_raw" + ".csv")
            logger.info("dataframe resampling complete")
            df_ohlc.to_csv(outputFileDirName + "_ohlc" + ".csv")
            logger.info("dataframe exported to CSV")
            logger.info(file + " format finished")

pd.DataFrame(currencyList).to_csv(FORMATED_DATA_DIR + "currencyList.csv", index=False, header=False)
logger.info("currencyList.csv createdV")
