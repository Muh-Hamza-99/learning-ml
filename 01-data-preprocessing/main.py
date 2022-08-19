# Importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Importing datasets

dataset = pd.read_csv("Data.csv")
features = dataset.iloc[:, :-1].values
dependent_variable_vector = dataset.iloc[:, -1].values

# Missing data

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(features[:, 1:3])
features[:, 1:3] = imputer.transform(features[:, 1:3])

# Encoding categorical data

# Encoding the independent variables

column_transformer = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [0])], remainder="passthrough")
features = np.array(column_transformer.fit_transform(features))

# Encoding the dependent variables

label_encoder = LabelEncoder()
dependent_variable_vector = label_encoder.fit_transform(dependent_variable_vector)

# Splitting dataset into training and test set

features_train, features_test, dependent_variable_vector_train, dependent_variable_vector_test = train_test_split(features, dependent_variable_vector, dependent_variable_vector, test_size=0.2, random_state=1)

# Feature scaling

standard_scaler = StandardScaler()
features_train[:, 3:] = standard_scaler.fit_transform(features_train[:, 3:])
features_test[:, 3:] = standard_scaler.transform(features_test[:, 3:])