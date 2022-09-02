# Importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

# Importing datasets

dataset = pd.read_csv("Mall_Customers.csv")
x = dataset.iloc[:, [3, 4]].values

# Finding optimal number of clusters

dendrogram = sch.dendrogram(sch.linkage(x, method="ward"))
plt.title("Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Euclidean Distance")
plt.show()

# Training Hierarchical Clustering model on whole dataset

hierarchical = AgglomerativeClustering(n_clusters=5, affinity="euclidean", linkage="ward")
y_predicted = hierarchical.fit_predict(x)

# Visualising the results

plt.scatter(x[y_predicted==0, 0], x[y_predicted==0, 1], s=100, c="red", label="Cluster 1")
plt.scatter(x[y_predicted==1, 0], x[y_predicted==1, 1], s=100, c="blue", label="Cluster 2")
plt.scatter(x[y_predicted==2, 0], x[y_predicted==2, 1], s=100, c="green", label="Cluster 3")
plt.scatter(x[y_predicted==3, 0], x[y_predicted==3, 1], s=100, c="cyan", label="Cluster 4")
plt.scatter(x[y_predicted==4, 0], x[y_predicted==4, 1], s=100, c="magenta", label="Cluster 5")
plt.title("Hierarchical Clustering")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()