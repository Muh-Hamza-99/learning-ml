# Importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from apyori import apriori

# Importing dataset

dataset = pd.read_csv("Market_Basket_Optimisation.csv", header=None)

# Converting dataset structure from pandas dataframe to 2D array

transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

# Training Eclat model on the whole dataset

rules = apriori(transactions=transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2, max_length=2)

# Putting results into a Pandas dataframe

def inspect(results):
    product_1 = [tuple(result[2][0][0])[0] for result in results]
    product_2 = [tuple(result[2][0][1])[0] for result in results]
    supports = [result[1] for result in results]
    return list(zip(product_1, product_2, supports))
results = list(rules)
results_dataframe = pd.DataFrame(inspect(results), columns=["Product 1", "Product 2", "Support"])

# Displaying sorted results by descending supports

print(results_dataframe.nlargest(n=10, columns="Support"))