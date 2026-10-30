"""
Day 47: Flask Machine Learning Prediction API
Install: python -m pip install flask scikit-learn joblib numpy
Run: python Day-47-Flask-ML-API-Complete-Project.py
POST endpoint: http://127.0.0.1:5000/predict

Educational demo only. The diabetes dataset target represents disease
progression. This is not a medical diagnostic tool.
"""
from pathlib import Path
import joblib
import numpy as np
from flask import Flask, jsonify, request
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

MODEL_PATH = Path(__file__).with_name("diabetes_model.joblib")
app = Flask(__name__)

def train_and_save_model():
    """Train the sample model and save its fitted parameters."""
    dataset = load_diabetes()
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(dataset.data, dataset.target)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved: {MODEL_PATH}")

def load_model():
    """Load the saved model, training it first if needed."""
    if not MODEL_PATH.exists():
        print("Model file not found. Training model...")
        train_and_save_model()
    # Only load model files from trusted sources.
    return joblib.load(MODEL_PATH)

# Load the model once when the Flask application starts.
model = load_model()

@app.route("/", methods=["GET"])
def home():
    """Return API status and endpoint information."""
    return jsonify({
        "message": "Day 47 Flask ML API is running.",
        "prediction_endpoint": "POST /predict",
        "required_feature_count": 10
    })

@app.route("/predict", methods=["POST"])
def predict():
    """Validate a JSON payload and return a prediction."""
    data = request.get_json(silent=True)

    if not isinstance(data, dict) or "features" not in data:
        return jsonify({"error": "Send JSON containing a 'features' list."}), 400

    features = data["features"]
    if not isinstance(features, list) or len(features) != 10:
        return jsonify({"error": "Exactly 10 numeric features are required."}), 400

    valid = all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and np.isfinite(value)
        for value in features
    )
    if not valid:
        return jsonify({"error": "Every feature must be a finite number."}), 400

    input_array = np.asarray(features, dtype=float).reshape(1, -1)
    try:
        result = float(model.predict(input_array)[0])
    except (ValueError, TypeError):
        return jsonify({"error": "The model could not process this input."}), 400

    return jsonify({
        "prediction": round(result, 2),
        "model": "RandomForestRegressor",
        "feature_count": 10
    })

if __name__ == "__main__":
    # Local development server only. Use a production WSGI server in production.
    app.run(host="127.0.0.1", port=5000, debug=False)
