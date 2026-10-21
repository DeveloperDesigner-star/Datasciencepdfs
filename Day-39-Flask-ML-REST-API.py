"""
Day 39: Flask REST API for a Saved Machine Learning Model
Educational example for the 60-Day Data Science Learning Journey.

Prerequisite: create student_result_model.joblib with the Day 38 project.
The saved pipeline must expect study_hours and attendance features.

Install: pip install flask pandas scikit-learn joblib
Run: python Day-39-Flask-ML-REST-API.py

This is a local learning example, not production-ready. The synthetic student
model must not be used for real academic decisions. Only load trusted joblib files.
"""
import math
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

MODEL_PATH = Path("student_result_model.joblib")
FEATURES = ["study_hours", "attendance"]
app = Flask(__name__)


def load_model():
    """Load the compatible model from a trusted source."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH.resolve()}. "
            "Run the Day 38 training script first."
        )
    return joblib.load(MODEL_PATH)


model = load_model()


@app.get("/health")
def health():
    """Check that the API is responding."""
    return jsonify({"status": "ok"}), 200


@app.post("/predict")
def predict():
    """Validate request JSON, call the model, and return JSON."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Send a valid JSON object."}), 400

    missing = [name for name in FEATURES if name not in data]
    if missing:
        return jsonify({
            "error": "Required fields are missing.",
            "missing_fields": missing
        }), 400

    values = {}
    try:
        for name in FEATURES:
            # bool is a subclass of int in Python, so reject it explicitly.
            if isinstance(data[name], bool):
                raise ValueError
            value = float(data[name])
            if not math.isfinite(value):
                raise ValueError
            values[name] = value
    except (TypeError, ValueError):
        return jsonify({
            "error": "study_hours and attendance must be finite numbers."
        }), 400

    if values["study_hours"] < 0:
        return jsonify({"error": "study_hours cannot be negative."}), 400
    if not 0 <= values["attendance"] <= 100:
        return jsonify({"error": "attendance must be between 0 and 100."}), 400

    features = pd.DataFrame([values], columns=FEATURES)
    try:
        prediction = model.predict(features)[0]
    except (TypeError, ValueError):
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction could not be generated."}), 500

    result = {
        "prediction": str(prediction),
        "message": "Prediction generated successfully.",
        "educational_only": True
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        result["class_probabilities"] = {
            str(label): round(float(prob), 4)
            for label, prob in zip(model.classes_, probabilities)
        }
    return jsonify(result), 200


@app.errorhandler(404)
def not_found(_error):
    return jsonify({"error": "Endpoint not found."}), 404


@app.errorhandler(405)
def method_not_allowed(_error):
    return jsonify({"error": "HTTP method not allowed."}), 405


if __name__ == "__main__":
    # Local development only. Use a production WSGI server for public hosting.
    app.run(host="127.0.0.1", port=5000, debug=False)
