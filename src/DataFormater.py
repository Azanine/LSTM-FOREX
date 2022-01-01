import os
import logging
import pandas as pd

RAW_DATA_DIR = "E:/forex data/new/"
FORMATED_DATA_DIR = "E:/forex data/output/"
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

RESAMPLE_STEP = '1S'

for file in os.listdir(RAW_DATA_DIR):
    if file.endswith(".csv"):

        fileCurrencyName = file.split("_")[-3]
        fileYearMonth = file.split("_")[-1]
        outputFileDirName = FORMATED_DATA_DIR + fileCurrencyName + '_' + fileYearMonth
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
            # df2 = round(merged_df[fileCurrencyName + "_bid"].resample(RESAMPLE_STEP, how='ohlc').mean(), 5)
            df2 = round(df.resample(RESAMPLE_STEP, on='timestamp').mean(), 5)
            logger.info("dataframe resampling complete")
            df2.to_csv(outputFileDirName)
            logger.info("dataframe exported to CSV")
            logger.info(file + " format finished")

pd.DataFrame(currencyList).to_csv(FORMATED_DATA_DIR+"currencyList.csv", index=False, header=False)
logger.info("currencyList.csv createdV")
