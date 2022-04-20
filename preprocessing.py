import pandas as pd
from datetime import datetime


def combine_data_sets():
    # read data sets
    data1 = pd.read_csv('data sets/taxi-rides.csv')
    data2 = pd.read_csv('data sets/weather.csv')

    # extract y from data1
    y = data1['price']
    data1 = data1.drop(["price"], axis=1)

    # convert from 1e12 to 1e9
    data1['time_stamp'] = data1['time_stamp'].apply(lambda t: int(t / 1000))

    # convert from time stamp to date
    data1['time_stamp'] = data1['time_stamp'].apply(lambda t: datetime.utcfromtimestamp(int(t)).strftime('%Y-%m-%d'))
    data2['time_stamp'] = data2['time_stamp'].apply(lambda t: datetime.utcfromtimestamp(int(t)).strftime('%Y-%m-%d'))

    # remove duplicates
    data1.drop_duplicates(inplace=True)
    data2.drop_duplicates(inplace=True)
    data1.reset_index(inplace=True)
    data2.reset_index(inplace=True)

    # add date_source and date_destination columns to data1
    date_list = data1['time_stamp']
    source_list = data1['source']
    destination_list = data1['destination']
    date_source_list = []
    date_destination_list = []
    for i in range(int(len(data1))):
        date_source_list.append(str(date_list[i]) + '/' + str(source_list[i]))
        date_destination_list.append(str(date_list[i]) + '/' + str(destination_list[i]))
    data1['date_source'] = date_source_list
    data1['date_destination'] = date_destination_list

    # add date_location columns to data2
    date_list = data2['time_stamp']
    location_list = data2['location']
    date_location_list = []
    for i in range(int(len(data2))):
        date_location_list.append(str(date_list[i]) + '/' + str(location_list[i]))
    data2['date_location'] = date_location_list

    # remove duplicates
    data2.drop_duplicates(subset=['date_location'], inplace=True)

    # merge data1 & data2 data sets
    data = pd.merge(data1, data2, how='inner', left_on='date_source', right_on='date_location')
    data = pd.merge(data, data2, how='inner', left_on='date_destination', right_on='date_location')

    # remove unnecessary columns
    data.drop(['index_x', 'time_stamp_x', 'date_source', 'index_y', 'time_stamp_y', 'date_location_x',
               'date_destination', 'index', 'date_location_y'], axis=1, inplace=True)

    x = data
    return x, y


def pre_process():
    x, y = combine_data_sets()

    return x, y
