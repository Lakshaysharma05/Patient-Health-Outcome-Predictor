from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import json
import os

dist_folder = os.path.join(os.path.dirname(__file__), "frontend", "dist")
app = Flask(__name__, static_folder=dist_folder, static_url_path="")
CORS(app)


# -----------------------------
# Load trained model
# -----------------------------

weights = np.load("model/weights.npy")
mean = np.load("model/mean.npy")
std = np.load("model/std.npy")
bias = np.load("model/bias.npy")[0]

with open("model/feature_columns.json", "r") as f:
    feature_columns = json.load(f)


# -----------------------------
# Sigmoid function
# -----------------------------

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


# -----------------------------
# Home page
# -----------------------------

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    if os.path.exists(os.path.join(app.static_folder, "index.html")):
        return send_from_directory(app.static_folder, "index.html")
    return jsonify({
        "status": "online",
        "message": "Patient Health Outcome Predictor API is operational"
    })


# -----------------------------
# Health and Info API
# -----------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "online",
        "message": "Patient Health Outcome Predictor API is operational",
        "feature_count": len(feature_columns)
    })


# -----------------------------
# Prediction API
# -----------------------------

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    try:
        raw_data = request.get_json(silent=True) or {}

        def safe_num(val, default):
            try:
                if val is None or val == "":
                    return float(default)
                return float(val)
            except (ValueError, TypeError):
                return float(default)

        def safe_str(val, default):
            if val is None or str(val).strip() == "":
                return default
            return str(val).strip()

        cleaned_data = {
            "Age": safe_num(raw_data.get("Age"), 50),
            "Diabetes": int(bool(raw_data.get("Diabetes", 0))),
            "Hypertension": int(bool(raw_data.get("Hypertension", 0))),
            "Heart_Disease": int(bool(raw_data.get("Heart_Disease", 0))),
            "Physical_Activity_Hours_Week": safe_num(raw_data.get("Physical_Activity_Hours_Week"), 3.0),
            "Sleep_Hours_Per_Night": safe_num(raw_data.get("Sleep_Hours_Per_Night"), 7.0),
            "BMI": safe_num(raw_data.get("BMI"), 25.0),
            "Glucose": safe_num(raw_data.get("Glucose"), 105.0),
            "Cholesterol": safe_num(raw_data.get("Cholesterol"), 180.0),
            "Treatment_Duration": safe_num(raw_data.get("Treatment_Duration"), 120),
            "Systolic_BP": safe_num(raw_data.get("Systolic_BP"), 120),
            "Diastolic_BP": safe_num(raw_data.get("Diastolic_BP"), 80),
            "Gender": safe_str(raw_data.get("Gender"), "Female"),
            "Living_Area": safe_str(raw_data.get("Living_Area"), "Urban"),
            "Past_Diseases": safe_str(raw_data.get("Past_Diseases"), "None"),
            "Smoking_Status": safe_str(raw_data.get("Smoking_Status"), "Never"),
            "Diet_Quality": safe_str(raw_data.get("Diet_Quality"), "Good"),
            "Stress_Level": safe_str(raw_data.get("Stress_Level"), "Low"),
            "Treatment": safe_str(raw_data.get("Treatment"), "Medication"),
            "Follow_Up": safe_str(raw_data.get("Follow_Up"), "Yes")
        }

        # Convert input into DataFrame
        input_df = pd.DataFrame([cleaned_data])

        # One-hot encode categorical features
        input_df = pd.get_dummies(
            input_df,
            columns=[
                "Gender",
                "Living_Area",
                "Past_Diseases",
                "Smoking_Status",
                "Diet_Quality",
                "Stress_Level",
                "Treatment",
                "Follow_Up"
            ],
            dtype=int
        )

        # Make sure feature order is EXACTLY the same
        # as during model training
        input_df = input_df.reindex(
            columns=feature_columns,
            fill_value=0
        )

        # Convert to NumPy
        input_np = input_df.to_numpy(dtype=float)

        # Scale using training mean/std
        input_scaled = (input_np - mean) / std

        # Calculate probability
        probability = sigmoid(
            input_scaled @ weights + bias
        )[0]

        # Convert probability to class
        prediction = int(probability >= 0.5)

        return jsonify({
            "prediction": prediction,
            "probability": float(probability),
            "status": "success"
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 400


# -----------------------------
# Run Flask
# -----------------------------

if __name__ == "__main__":
    app.run(debug=True)