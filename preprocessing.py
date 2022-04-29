import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt


def combine_data_sets(data1, data2):
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

    corr_threshold = 0.0
    corr = data.corr()
    top_feature = corr.index[abs(corr['price']) > corr_threshold]

    plt.subplots(figsize=(12, 8))
    top_corr = data[top_feature].corr()
    sns.heatmap(top_corr, annot=True)
    plt.show()

    x = data[top_feature]
    x.drop(['price', 'id', 'product_id'], inplace=True, axis=1)

    return x, y


def feature_scaling(x, a, b):
    x = np.array(x)
    normalized_x = np.zeros((x.shape[0], x.shape[1]))
    for i in range(x.shape[1]):
        normalized_x[:, i] = ((x[:, i]-min(x[:, i]))/(max(x[:, i])-min(x[:, i])))*(b-a)+a
    return normalized_x


def pre_process(data1, data2):
    # combine several data sets into one
    data = combine_data_sets(data1, data2)

    # one hot encoding to increasing number of features
    data = pd.get_dummies(data, columns=['name', 'cab_type', 'destination', 'source'])

    # mapping string values to numbers
    cols = ['time_stamp', 'id', 'product_id']
    data = feature_encoder(data, cols)

    # remove rows with missing y
    data.dropna(axis=0, subset=['price'], inplace=True)

    # remove columns with many NULL values
    data.dropna(axis=1, thresh=250000, inplace=True)

    x, y = feature_selection(data)

    # min max scaling
    x = feature_scaling(x, 0, 1)

    return x, y
