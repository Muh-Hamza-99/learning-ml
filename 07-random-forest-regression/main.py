# NOTE: Random Forest Regression is not suitable for datasets with a single feature; it is more suited for multi-dimensional datasets, with multiple features.

# Importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Importing datasets

dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training Random Forest Regression model on the whole dataset

regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(x, y)

# Predicting a new result

y_predicted = regressor.predict([[6.5]])

# Visualising the Random Forest Regression results (higher resolution)

x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color="red")
plt.plot(x_grid, regressor.predict(x), color="blue")
plt.title("Truth or Bluff (Random Forest Regression)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()