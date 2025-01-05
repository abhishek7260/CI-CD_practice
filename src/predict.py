# src/predict.py
import pickle
import numpy as np

def predict(input_data):
    with open("artifacts/model.pkl", "rb") as f:
        model = pickle.load(f)
    return model.predict(np.array(input_data).reshape(1, -1))
