import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt


def combine_data_sets():
    # read data sets
    data1 = pd.read_csv('data sets/taxi-rides.csv')
    data2 = pd.read_csv('data sets/weather.csv')

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
    data.drop(['location_x', 'location_y'], axis=1, inplace=True)

    return data


def feature_encoder(x, cols):
    for c in cols:
        lbl = LabelEncoder()
        x[c] = lbl.fit_transform(list(x[c].values))
    return x


def feature_selection(data):
    y = data['price']

    corr = data.corr()
    top_feature = corr.index[abs(corr['price']) > 0.00]

    plt.subplots(figsize=(12, 8))
    top_corr = data[top_feature].corr()
    sns.heatmap(top_corr, annot=True)
    plt.show()

    top_feature = top_feature.delete(-1)
    x = data[top_feature]

    return x, y


def pre_process():
    # combine several data sets into one
    data = combine_data_sets()

    # label encoding
    cols = ['cab_type', 'time_stamp', 'destination', 'source', 'name', 'id', 'product_id']
    data = feature_encoder(data, cols)

    # remove rows with missing y
    data.dropna(axis=0, subset=['price'], inplace=True)

    # remove columns with many NULL values
    data.dropna(axis=1, thresh=250000, inplace=True)

    # remove columns that intuitively don't affect the output
    data.drop(['id', 'product_id'], axis=1, inplace=True)

    x, y = feature_selection(data)

    return x, y
