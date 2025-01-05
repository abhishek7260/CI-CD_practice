# app.py
from flask import Flask, request, jsonify
from src.predict import predict

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict_route():
    input_data = request.json["data"]
    prediction = predict(input_data)
    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
