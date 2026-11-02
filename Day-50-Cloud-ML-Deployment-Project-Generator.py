"""Generate a cloud-ready Flask ML API project in ./cloud_ml_deployment.
Run: python Day-50-Cloud-ML-Deployment-Project-Generator.py
Then follow README.txt in the generated folder.
Educational regression example only. Not a medical tool.
"""
from pathlib import Path
import textwrap

root = Path("cloud_ml_deployment")
files = {
"train_model.py": r"""
from pathlib import Path
import joblib
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

data = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, pred):.2f}")
print(f"R2: {r2_score(y_test, pred):.3f}")
path = Path(__file__).resolve().parent / "diabetes_model.joblib"
joblib.dump(model, path)
print(f"Saved model: {path}")
""",
"app.py": r"""
from pathlib import Path
import math
import joblib
import numpy as np
from flask import Flask, jsonify, request

model_path = Path(__file__).resolve().parent / "diabetes_model.joblib"
if not model_path.exists():
    raise FileNotFoundError("Run python train_model.py before starting the API.")

app = Flask(__name__)
model = joblib.load(model_path)

@app.get("/")
def home():
    return jsonify({"message": "Cloud ML API is running",
                    "prediction_endpoint": "POST /predict",
                    "expected_features": 10})

@app.get("/health")
def health():
    return jsonify({"status": "healthy"})

@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "A JSON object is required."}), 400
    features = payload.get("features")
    if not isinstance(features, list) or len(features) != 10:
        return jsonify({"error": "Provide exactly 10 feature values."}), 400
    try:
        values = [float(x) for x in features]
    except (TypeError, ValueError):
        return jsonify({"error": "All features must be numeric."}), 400
    if not all(math.isfinite(x) for x in values):
        return jsonify({"error": "Values must be finite numbers."}), 400
    result = float(model.predict(np.asarray(values).reshape(1, -1))[0])
    return jsonify({"prediction": round(result, 2),
                    "model": "RandomForestRegressor",
                    "notice": "Educational demonstration only."})

if __name__ == "__main__":
    # Local development. Hosted service uses Gunicorn.
    app.run(host="0.0.0.0", port=5000, debug=False)
""",
"requirements.txt": """Flask>=3.0,<4.0
gunicorn>=21.2,<24.0
joblib>=1.3,<2.0
numpy>=1.26,<3.0
scikit-learn>=1.4,<2.0
""",
"render.yaml": """services:
  - type: web
    name: cloud-ml-prediction-api
    runtime: python
    buildCommand: pip install -r requirements.txt && python train_model.py
    startCommand: gunicorn app:app
    healthCheckPath: /health
""",
".gitignore": """__pycache__/
*.py[cod]
.venv/
venv/
.env
.pytest_cache/
.DS_Store
""",
"README.txt": """DAY 50 CLOUD ML DEPLOYMENT PROJECT

LOCAL TEST
1. Install dependencies:
   python -m pip install -r requirements.txt
2. Train and save the model:
   python train_model.py
3. Start the API:
   python app.py
4. Open http://127.0.0.1:5000/ and /health

PREDICTION REQUEST
POST http://127.0.0.1:5000/predict
Content-Type: application/json

{
  "features": [0.038, 0.05, 0.06, 0.02, -0.04,
               -0.03, -0.04, -0.003, -0.01, 0.02]
}

RENDER DEPLOYMENT
1. Push this folder to GitHub.
2. Create a Render Web Service and connect the repository.
3. If this project is a subfolder, set the service root directory to it.
4. Build command:
   pip install -r requirements.txt && python train_model.py
5. Start command:
   gunicorn app:app
6. Deploy. After a successful deployment, Render provides the service URL.
7. Test the URL, /health, and POST /predict.

IMPORTANT
- Do not commit API keys, passwords, or .env files.
- Free hosting tiers have limits and may sleep when inactive. Check current terms.
- For production, use tested pinned dependencies, authentication, rate limiting,
  monitoring, structured logs, and security controls.
- The scikit-learn diabetes dataset is an educational regression example only.
  It is not a diagnosis, personal risk assessment, or treatment tool.
"""
}

root.mkdir(exist_ok=True)
for name, content in files.items():
    (root / name).write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")
print("Created project folder:", root.resolve())
for name in files:
    print(" -", name)
