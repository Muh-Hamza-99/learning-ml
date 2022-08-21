# Importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Importing datasets

dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training Simple Linear Regression model on the whole dataset

linear_regressor = LinearRegression()
linear_regressor.fit(x, y)

# Training Polynomial Linear Regression model on the whole dataset

polynomial_regressor = PolynomialFeatures(degree=4)
x_polynomial = polynomial_regressor.fit_transform(x)
linear_regressor_polynomial = LinearRegression()
linear_regressor_polynomial.fit(x_polynomial, y)

# Visualising the Simple Linear Regression results

plt.scatter(x, y, color="red")
plt.plot(x, linear_regressor.predict(x), color="blue")
plt.title("Truth or Bluff (Simple Linear Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Visualising the Polynomial Linear Regression results

plt.scatter(x, y, color="red")
plt.plot(x, linear_regressor_polynomial.predict(x_polynomial), color="blue")
plt.title("Truth or Bluff (Polynomial Linear Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Predicting a new result with Simple Linear Regression model

print(linear_regressor.predict([[6.5]]))

# Predicting a new result with Polynomial Linear Regression model

print(linear_regressor_polynomial.predict(polynomial_regressor.fit_transform([[6.5]])))