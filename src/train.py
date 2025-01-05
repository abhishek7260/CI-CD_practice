# src/train.py
import pickle
from sklearn.linear_model import LogisticRegression
from src.preprocess import load_and_split_data

def train_model():
    X_train, X_test, y_train, y_test = load_and_split_data()
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # Save the model
    with open("artifacts/model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model trained and saved successfully!")

if __name__ == "__main__":
    train_model()
