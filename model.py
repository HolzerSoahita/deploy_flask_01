# Importing the libraries
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# Simple dataset of three four features
# First of all load csv dataset file into dataframe
dataset = pd.read_csv('./hiring.csv')

# print the dataset
print(dataset)

# feature engineering
# to replace each Null cell
dataset['experience'].fillna(0, inplace=True)

dataset['test_score'].fillna(dataset['test_score'].mean(), inplace=True)

# now specify the Input features for training
#X = dataset.iloc[['experience', 'test_score', 'interview']]
X=dataset.iloc[:,:3]

# Now some features are string/text, we need to convert them to number
# Converting words to integer values


def convert_to_int(word):
    word_dict = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8,
                 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12, 'zero': 0, 0: 0}
    return word_dict[word]


X['experience'] = X['experience'].apply(lambda x: convert_to_int(x))

y = dataset['salary']

# Splitting Training and Test Set
# Since we have a very small dataset, we will train our model with all availabe data.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)


# train your model using linear Regression
regressor = LinearRegression()

# Fitting model with trainig data
regressor.fit(X_train, y_train)

# Saving model to disk
pickle.dump(regressor, open('model.pkl', 'wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl', 'rb'))
print("Test one prediction")
# using predict() function to test your model

print(model.predict([[5,6,7]]))
