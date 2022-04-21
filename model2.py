import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
import time


def polynomial_regression(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, shuffle=True, random_state=10)

    # X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size = 0.50,shuffle=True)

    poly_features = PolynomialFeatures(degree=1)

    # transforms the existing features to higher degree features.
    x_train_poly = poly_features.fit_transform(x_train)

    # fit the transformed features to Linear Regression
    poly_model = linear_model.LinearRegression()
    start = time.time()
    poly_model.fit(x_train_poly, y_train)
    end = time.time()

    # predicting on training data-set
    y_train_predicted = poly_model.predict(x_train_poly)
    y_prediction = poly_model.predict(poly_features.transform(x_test))

    # predicting on test data-set
    prediction = poly_model.predict(poly_features.fit_transform(x_test))

    print('Co-efficient of linear regression', poly_model.coef_)
    print('Intercept of linear regression model', poly_model.intercept_)
    print('Mean Square Error', metrics.mean_squared_error(y_test, prediction))

    true_value = np.asarray(y_test)[:10]
    predicted_value = prediction[:10]
    print('True values: ' + str(true_value))
    print('Predicted values: ' + str(predicted_value))
    print('Training time:', end - start, 'sec')
