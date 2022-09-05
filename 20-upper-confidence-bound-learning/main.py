# Importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# Importing dataset

dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

# Implementing the Upper Confidence Bound algorithm

NUMBER_OF_ROUNDS = 10000
NUMBER_OF_ADS = 10
ads_selected = []
numbers_of_selections = [0] * NUMBER_OF_ADS
sums_of_rewards = [0] * NUMBER_OF_ADS
total_reward = 0
for n in range(0, NUMBER_OF_ROUNDS):
    ad = 0
    max_upper_confidence_bound = 0
    for i in range(0, NUMBER_OF_ADS):
        if numbers_of_selections[i] > 0:
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            confidence_interval = math.sqrt(1.5 * math.log(n+1) / numbers_of_selections[i])
            upper_confidence_bound = average_reward + confidence_interval
        else:
            upper_confidence_bound = 1e400
        if upper_confidence_bound > max_upper_confidence_bound:
            max_upper_confidence_bound = upper_confidence_bound
            ad = i
    ads_selected.append(ad)
    numbers_of_selections[ad] += 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward += reward

# Visualising results from the Upper Confidence Bound algorithm

plt.hist(ads_selected)
plt.title("Ads Selection (Upper Confidence Bound)")
plt.xlabel("Ads")
plt.ylabel("Ad Selection Count")
plt.show()