import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
import time


def regression(x, y, degree):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, shuffle=True, random_state=10)

    # X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size = 0.50,shuffle=True)

    poly_features = PolynomialFeatures(degree=degree)

    # transforms the existing features to higher degree features.
    x_train_poly = poly_features.fit_transform(x_train)
    x_test_poly = poly_features.fit_transform(x_test)

    # fit the transformed features to Linear Regression
    poly_model = linear_model.LinearRegression()
    start = time.time()
    poly_model.fit(x_train_poly, y_train)
    end = time.time()

    # predicting on training data-set
    y_train_predicted = poly_model.predict(x_train_poly)
    y_test_predicted = poly_model.predict(x_test_poly)

    print('Co-efficient regression', poly_model.coef_)
    print('Intercept of regression model', poly_model.intercept_)
    print('Train Mean Square Error', metrics.mean_squared_error(y_train, y_train_predicted))
    print('Test Mean Square Error', metrics.mean_squared_error(y_test, y_test_predicted))

    true_value = np.asarray(y_test)[:10]
    predicted_value = y_test_predicted[:10]
    print('True values: ' + str(true_value))
    print('Predicted values: ' + str(predicted_value))
    print('Training time:', end - start, 'sec')
