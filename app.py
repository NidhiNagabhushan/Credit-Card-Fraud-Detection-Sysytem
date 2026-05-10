from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app, origins="*")

model = joblib.load("fraud_detection_model.pkl")

@app.route("/")
def home():
    return "Fraud Detection API is Running"

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        response = jsonify({})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        return response, 200

    try:
        data = request.json
        amount = data["features"][28]
        features = np.zeros((1, 29))
        features[0][28] = amount
        prediction = model.predict(features)
        result = "Fraud" if prediction[0] == 1 else "Not Fraud"
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)