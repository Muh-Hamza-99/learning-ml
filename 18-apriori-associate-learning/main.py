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

# Training Apriori model on the whole dataset

rules = apriori(transactions=transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2, max_length=2)

# Putting results into a Pandas dataframe

def inspect(results):
    lefthand_side = [tuple(result[2][0][0])[0] for result in results]
    righthand_side = [tuple(result[2][0][1])[0] for result in results]
    supports = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts = [result[2][0][3] for result in results]
    return list(zip(lefthand_side, righthand_side, supports, confidences, lifts))
results = list(rules)
results_dataframe = pd.DataFrame(inspect(results), columns=["Left Hand Side", "Right Hand Side", "Support", "Confidence", "Lift"])

# Displaying sorted results by descending lifts

print(results_dataframe.nlargest(n=10, columns="Lift"))