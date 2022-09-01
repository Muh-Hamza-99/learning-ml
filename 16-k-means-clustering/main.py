# Importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Importing datasets

dataset = pd.read_csv("Mall_Customers.csv")
x = dataset.iloc[:, [3, 4]].values

# Finding optimal number of clusters

wcss_list = []
for i in range(1, 11):
    k_means = KMeans(n_clusters=i, init="k-means++", random_state=42)
    k_means.fit(x)
    wcss_list.append(k_means.inertia_)
plt.plot(range(1, 11), wcss_list)
plt.title("The Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# Training K-Means Clustering model on whole dataset

k_means = KMeans(n_clusters=5, init="k-means++", random_state=42)
y_predicted = k_means.fit_predict(x)

# Visualising the results

plt.scatter(x[y_predicted==0, 0], x[y_predicted==0, 1], s=100, c="red", label="Cluster 1")
plt.scatter(x[y_predicted==1, 0], x[y_predicted==1, 1], s=100, c="blue", label="Cluster 2")
plt.scatter(x[y_predicted==2, 0], x[y_predicted==2, 1], s=100, c="green", label="Cluster 3")
plt.scatter(x[y_predicted==3, 0], x[y_predicted==3, 1], s=100, c="cyan", label="Cluster 4")
plt.scatter(x[y_predicted==4, 0], x[y_predicted==4, 1], s=100, c="magenta", label="Cluster 5")
plt.scatter(k_means.cluster_centers_[:, 0], k_means.cluster_centers_[:, 1], s=300, c="yellow", label="Centroids", edgecolors="black")
plt.title("K Means Clustering")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()