# tests/test_pipeline.py
from src.preprocess import load_and_split_data
from src.train import train_model
import os

def test_data_split():
    X_train, X_test, y_train, y_test = load_and_split_data()
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0

def test_model_training():
    train_model()
    assert os.path.exists("artifacts/model.pkl")
