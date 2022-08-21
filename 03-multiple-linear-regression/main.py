# Importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

# Importing datasets

dataset = pd.read_csv("50_Startups.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Encoding categorical data

column_transformer = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [3])], remainder="passthrough")
x = np.array(column_transformer.fit_transform(x))

# Splitting dataset into training and test set

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# Training Multiple Linear Regression model on training set

regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the test set results

y_predicated = regressor.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_predicated.reshape(len(y_predicated), 1), y_test.reshape(len(y_test), 1)), 1))