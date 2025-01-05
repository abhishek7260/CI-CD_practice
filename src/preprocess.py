# src/preprocess.py
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd

def load_and_split_data(test_size=0.2, random_state=42):
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
