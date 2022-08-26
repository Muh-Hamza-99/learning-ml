# Importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

# Importing datasets

dataset = pd.read_csv("Social_Network_Ads.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting dataset into training and test set

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

# Feature scaling

standard_scaler = StandardScaler()
x_train = standard_scaler.fit_transform(x_train)
x_test = standard_scaler.transform(x_test)

# Training Kernel Support Vector Machine model on the training set

classifier = SVC(kernel="rbf", random_state=0)
classifier.fit(x_train, y_train)

# Predicting a new single result

print(classifier.predict([x_test[0]]))

# Predicting the test set results

y_predicted = classifier.predict(x_test)
print(np.concatenate((y_predicted.reshape(len(y_predicted), 1), y_test.reshape(len(y_test), 1)), 1))

# Creating the confusion matrix for predicted values

confusion_matrix = confusion_matrix(y_test, y_predicted)
accuracy_score = accuracy_score(y_test, y_predicted)

# Visualising the training set results

x_set, y_set = standard_scaler.inverse_transform(x_train), y_train
x1, x2 = np.meshgrid(np.arange(start=x_set[:, 0].min()-10, stop=x_set[:, 0].max()+10, step=0.25), np.arange(start=x_set[:, 1].min()-1000, stop=x_set[:, 1].max()+1000, step=0.25))
plt.contourf(x1, x2, classifier.predict(standard_scaler.transform(np.array([x1.ravel(), x2.ravel()]).T)).reshape(x1.shape), alpha=0.75, cmap=ListedColormap(("red", "green")))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j, 0], x_set[y_set==j, 1], c=ListedColormap(("red", "green"))(i), label=j)
plt.title("Kernel Support Vector Machine (Training set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()

# Visualising the test set results

# x_set, y_set = standard_scaler.inverse_transform(x_test), y_test
# x1, x2 = np.meshgrid(np.arange(start=x_set[:, 0].min()-10, stop=x_set[:, 0].max()+10, step=0.25), np.arange(start=x_set[:, 1].min()-1000, stop=x_set[:, 1].max()+1000, step=0.25))
# plt.contourf(x1, x2, classifier.predict(standard_scaler.transform(np.array([x1.ravel(), x2.ravel()]).T)).reshape(x1.shape), alpha=0.75, cmap=ListedColormap(("red", "green")))
# plt.xlim(x1.min(), x1.max())
# plt.ylim(x2.min(), x2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], c = ListedColormap(("red", "green"))(i), label = j)
# plt.title("Kernel Support Vector Machine (Test set)")
# plt.xlabel("Age")
# plt.ylabel("Estimated Salary")
# plt.legend()
# plt.show()