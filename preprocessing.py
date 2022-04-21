import pandas as pd
from datetime import datetime


def combine_data_sets():
    # read data sets
    data1 = pd.read_csv('data sets/taxi-rides.csv')
    data2 = pd.read_csv('data sets/weather.csv')

    # extract y from data1
    y = data1['price']
    data1.drop(["price"], axis=1, inplace=True)

    # convert from 1e12 to 1e9
    data1['time_stamp'] = data1['time_stamp'].apply(lambda t: int(t / 1000))

    # convert from time stamp to date
    data1['time_stamp'] = data1['time_stamp'].apply(lambda t: datetime.utcfromtimestamp(int(t)).strftime('%Y-%m-%d'))
    data2['time_stamp'] = data2['time_stamp'].apply(lambda t: datetime.utcfromtimestamp(int(t)).strftime('%Y-%m-%d'))

    # remove duplicates
    data1.drop_duplicates(inplace=True)
    data2.drop_duplicates(inplace=True)

    # remove duplicates
    data2.drop_duplicates(subset=['time_stamp', 'location'], inplace=True)

    # merge data1 & data2 data sets
    data = pd.merge(data1, data2, how='inner', left_on=['source', 'time_stamp'],
                    right_on=['location', 'time_stamp'])
    data = pd.merge(data, data2, how='inner', left_on=['destination', 'time_stamp'],
                    right_on=['location', 'time_stamp'])

    # remove unnecessary columns
    data.drop(['index_x', 'index_y', 'index', 'location_x', 'location_y'], axis=1, inplace=True)

    x = data
    return x, y


def pre_process():
    x, y = combine_data_sets()

    return x, y
