# Importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Importing datasets

dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Feature scaling

y = y.reshape(len(y), 1)
standard_scaler_x = StandardScaler()
standard_scaler_y = StandardScaler()
x = standard_scaler_x.fit_transform(x)
y = standard_scaler_y.fit_transform(y)

# Training (non-linear) Support Vector Regression model on the whole dataset

regressor = SVR(kernel="rbf")
regressor.fit(x, y.ravel())

# Predicting a new result by using reverse scaling

y_predicated = standard_scaler_y.inverse_transform(regressor.predict(standard_scaler_x.transform([[6.5]])))

# Visualising the (non-linear) Support Vector Regression results
plt.scatter(standard_scaler_x.inverse_transform(x), standard_scaler_y.inverse_transform(y), color="red")
plt.plot(standard_scaler_x.inverse_transform(x), standard_scaler_y.inverse_transform(regressor.predict(x)), color="blue")
plt.title("Truth or Bluff (Support Vector Regression)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()