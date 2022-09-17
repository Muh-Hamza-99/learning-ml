# Importing libraries

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# Importing datasets

dataset = pd.read_csv("Churn_Modelling.csv")
x = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# Encoding categorical data

label_encoder = LabelEncoder()
x[:, 2] = label_encoder.fit_transform(x[:, 2])
column_transformer = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [1])], remainder="passthrough")
x = np.array(column_transformer.fit_transform(x))

# Splitting dataset into training and test set

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# Feature scaling

standard_scaler = StandardScaler()
x_train = standard_scaler.fit_transform(x_train)
x_test = standard_scaler.transform(x_test)

# Initialising the Artificial Neural Network

ann = tf.keras.models.Sequential()

# Adding the input layer and first hidden layer

ann.add(tf.keras.layers.Dense(units=6, activation="relu"))

# Adding the second hidden layer

ann.add(tf.keras.layers.Dense(units=6, activation="relu"))

# Adding the output layer

ann.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

# Compiling the Artificial Neural Network

ann.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Training the Artificial Neural Network on the training set

ann.fit(x_train, y_train, batch_size=32, epochs=100)

# Predicting the result of a single observation

print(ann.predict(standard_scaler.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)

# Predicting the test set results

y_predicted = ann.predict(x_test)
y_predicted = (y_predicted > 0.5)
print(np.concatenate((y_predicted.reshape(len(y_predicted),1), y_test.reshape(len(y_test),1)),1))

# Making the confusion matrix and accuracy score

confusion_matrix = confusion_matrix(y_test, y_predicted)
accuracy_score = accuracy_score(y_test, y_predicted)
print(confusion_matrix)
print(accuracy_score)