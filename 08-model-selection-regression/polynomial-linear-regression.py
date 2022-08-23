# Importing the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Importing the dataset
dataset = pd.read_csv("Data.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the training set and test set

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Training the Polynomial Regression model on the training set

polynomial_regressor = PolynomialFeatures(degree=4)
x_polynomial = polynomial_regressor.fit_transform(x_train)
regressor = LinearRegression()
regressor.fit(x_polynomial, y_train)

# Predicting the test set results

y_predicted = regressor.predict(polynomial_regressor.transform(x_test))
np.set_printoptions(precision=2)
print(np.concatenate((y_predicted.reshape(len(y_predicted),1), y_test.reshape(len(y_test),1)),1))

# Evaluating the model performance

print(r2_score(y_test, y_predicted))