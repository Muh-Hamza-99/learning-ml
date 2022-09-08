# Importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import  random

# Importing dataset

dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

# Implementing the Thompson Sampling algorithm

NUMBER_OF_ROUNDS = 10000
NUMBER_OF_ADS = 10
ads_selected = []
numbers_of_rewards_1 = [0] * NUMBER_OF_ADS
numbers_of_rewards_0 = [0] * NUMBER_OF_ADS
total_reward = 0
for n in range(0, NUMBER_OF_ROUNDS):
    ad = 0
    max_random = 0
    for i in range(0, NUMBER_OF_ADS):
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:
        numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1
    else:
        numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1
    total_reward += reward

# Visualising results from the Thompson Sampling algorithm

plt.hist(ads_selected)
plt.title("Ads Selection (Thompson Sampling)")
plt.xlabel("Ads")
plt.ylabel("Ad Selection Count")
plt.show()