import pandas as pd
import preprocessing as pre
import regression_model

if __name__ == '__main__':
    url1 = 'data sets/taxi-rides.csv'
    url2 = 'data sets/weather.csv'

    data1 = pd.read_csv(url1)
    data2 = pd.read_csv(url2)

    x, y = pre.pre_process(data1, data2)

    degree = 2
    regression_model.regression(x, y, degree)
