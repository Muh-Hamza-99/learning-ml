# Importing libraries

import re
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Importing dataset

dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter="\t", quoting=3)

# Cleaning text

nltk.download("stopwords")
corpus = []
for i in range(0, 1000):
    review = re.sub("[^a-zA-Z]", " ", dataset["Review"][i])
    review = review.lower()
    review = review.split()
    porter_stemmer = PorterStemmer()
    all_stopwords = stopwords.words("english")
    all_stopwords.remove("not")
    review = [porter_stemmer.stem(word) for word in review if not word in set(all_stopwords)]
    review = " ".join(review)
    corpus.append(review)

# Creating the Bag of Words model

count_vectoriser = CountVectorizer(max_features=1500)
x = count_vectoriser.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

# Splitting dataset into training and test set

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Training Naive Bayes model on the training set

classifier = GaussianNB()
classifier.fit(x_train, y_train)

# Predicting the test set results

y_predicted = classifier.predict(x_test)
print(np.concatenate((y_predicted.reshape(len(y_predicted), 1), y_test.reshape(len(y_test), 1)), 1))

# Creating the confusion matrix for predicted values

confusion_matrix = confusion_matrix(y_test, y_predicted)
accuracy_score = accuracy_score(y_test, y_predicted)