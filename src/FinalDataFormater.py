import pandas as pd
import os
import csv
from dateutil import rrule
from datetime import datetime
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')

file_handler = logging.FileHandler('E:/forex data/log/FinalDataFormater.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

formated_data_folder_dir = "E:/forex data/output/"

final_raw_df = pd.DataFrame()
final_ohlc_df = pd.DataFrame()
df_raw = pd.DataFrame()
df_ohlc = pd.DataFrame()
monthly_raw_df = pd.DataFrame()
monthly_ohlc_df = pd.DataFrame()

with open(formated_data_folder_dir + "currencyList.csv", 'r') as f:
    reader = csv.reader(f)
    currency_list = list(reader)

for dt in rrule.rrule(rrule.MONTHLY, dtstart=datetime(2000, 1, 1), until=datetime.now()):

    for currencyListItem in currency_list:
        file_ohlc = formated_data_folder_dir + currencyListItem[0] + '_' + \
                    str(dt.year) + dt.strftime('%m') + '_ohlc' + '.csv'
        file_raw = formated_data_folder_dir + currencyListItem[0] + '_' + str(dt.year) + dt.strftime(
            '%m') + '_raw' + '.csv'

        if os.path.isfile(file_raw):
            df_raw = pd.read_csv(file_raw, index_col='timestamp')
            logger.info("new file is " + currencyListItem[0] + '_' + str(dt.year) + dt.strftime('%m')
                        + '_raw' + '.csv')
            if monthly_raw_df.empty:
                monthly_raw_df = df_raw
                logger.info("new Dataframe assigned to empty monthly_raw Dataframe")
            else:
                # kanw merge side by side gia ta dedomena me ton idio mhna
                # h merge vazei ta kainouria aristera mipws kalutera na ta evaze deksia???
                monthly_raw_df = monthly_raw_df.merge(df_raw, on='timestamp')
                logger.info("dataframe " + currencyListItem[0] + '_' + str(dt.year) +
                            dt.strftime('%m') + '_raw' + '.csv' + " merged with old")
        if os.path.isfile(file_ohlc):
            df_ohlc = pd.read_csv(file_ohlc, index_col='timestamp')
            logger.info("new file is " + currencyListItem[0] + '_' + str(dt.year) + dt.strftime('%m') + '.csv')
            if monthly_ohlc_df.empty:
                monthly_ohlc_df = df_ohlc
                logger.info("new Dataframe assigned to empty monthly Dataframe")
            else:
                # kanw merge side by side gia ta dedomena me ton idio mhna
                # h merge vazei ta kainouria aristera mipws kalutera na ta evaze deksia???
                monthly_ohlc_df = monthly_ohlc_df.merge(df_ohlc, on='timestamp')
                logger.info("dataframe " + currencyListItem[0] + '_' + str(dt.year) + dt.strftime('%m') + '.csv' +
                            " merged with old")

    if final_raw_df.empty and (not monthly_raw_df.empty):
        final_raw_df = monthly_raw_df
        monthly_raw_df = pd.DataFrame()
        logger.info("final_raw_df is empty, monthly_raw_df assigned to final_raw_df")
    elif not monthly_raw_df.empty:
        concat_list = [final_raw_df, monthly_raw_df]
        final_raw_df = pd.concat(concat_list, sort=True)
        monthly_raw_df = pd.DataFrame()
        logger.info("final_df concatenated with monthly_df")

    if final_ohlc_df.empty and (not monthly_ohlc_df.empty):
        final_ohlc_df = monthly_ohlc_df
        monthly_ohlc_df = pd.DataFrame()
        logger.info("final_raw_df is empty, monthly_raw_df assigned to final_raw_df")
    elif not monthly_ohlc_df.empty:
        concat_list = [final_ohlc_df, monthly_ohlc_df]
        final_ohlc_df = pd.concat(concat_list, sort=True)
        monthly_ohlc_df = pd.DataFrame()
        logger.info("final_df concatenated with monthly_df")

final_raw_df.fillna(method='ffill', inplace=True)
logger.info("raw NaNs replaced with previous value")
final_raw_df.fillna(method='bfill', inplace=True)
logger.info(" remaiming raw NaNs replaced with next value")
final_raw_df.to_csv(formated_data_folder_dir + "final_raw.csv", float_format='%.5f')
logger.info("final_raw_df exported to CSV")

final_ohlc_df.fillna(method='ffill', inplace=True)
logger.info("ohlc NaNs replaced with previous value")
final_ohlc_df.fillna(method='bfill', inplace=True)
logger.info(" remaiming ohlc NaNs replaced with next value")
final_ohlc_df.to_csv(formated_data_folder_dir + "final_ohlc.csv", float_format='%.5f')
logger.info("final_ohlc_df exported to CSV")
