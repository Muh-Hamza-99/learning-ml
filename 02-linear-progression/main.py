# Importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Importing datasets

dataset = pd.read_csv("Salary_Data.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting dataset into training and test set

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# Training Simple Linear Regression model on training set

regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the test set results

y_predicted = regressor.predict(x_test)

# Visualising the training set results

plt.scatter(x_train, y_train, color="red")
plt.plot(x_train, regressor.predict(x_train), color="blue")
plt.title("Salary vs. Experience (Training Set)")
plt.xlabel("Years Of Experience")
plt.ylabel("Salary")
plt.show()

# Visualising the test set results

plt.scatter(x_test, y_test, color="red")
plt.plot(x_train, regressor.predict(x_train), color="blue")
plt.title("Salary vs. Experience (Test Set)")
plt.xlabel("Years Of Experience")
plt.ylabel("Salary")
plt.show()