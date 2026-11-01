"""Generate a complete Dockerized Flask ML API project in ./docker_ml_api.
Run: python Day-49-Docker-ML-Project-Generator.py
Then follow docker_ml_api/README.txt.
Educational regression demo only. Not a medical diagnostic tool.
"""
from pathlib import Path
import textwrap

project = Path("docker_ml_api")
files = {
"app.py": r"""
from pathlib import Path
import math
import joblib
import numpy as np
from flask import Flask, jsonify, request

MODEL_PATH = Path(__file__).resolve().parent / "diabetes_model.joblib"
if not MODEL_PATH.exists():
    raise FileNotFoundError("Run train_model.py first to create the model.")

app = Flask(__name__)
model = joblib.load(MODEL_PATH)

@app.get("/")
def home():
    return jsonify({"message": "Dockerized ML API is running",
                    "prediction_endpoint": "POST /predict",
                    "expected_features": 10})

@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or "features" not in payload:
        return jsonify({"error": "Send JSON containing a features list."}), 400
    features = payload["features"]
    if not isinstance(features, list) or len(features) != 10:
        return jsonify({"error": "Provide exactly 10 feature values."}), 400
    try:
        values = [float(v) for v in features]
    except (TypeError, ValueError):
        return jsonify({"error": "Every feature must be numeric."}), 400
    if not all(math.isfinite(v) for v in values):
        return jsonify({"error": "Values must be finite numbers."}), 400
    result = float(model.predict(np.array(values).reshape(1, -1))[0])
    return jsonify({"prediction": round(result, 2),
                    "model": "RandomForestRegressor",
                    "notice": "Educational dataset demonstration only."})

if __name__ == "__main__":
    # Listen on the container network interface so Docker can route traffic.
    app.run(host="0.0.0.0", port=5000, debug=False)
""",
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
model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, pred):.2f}")
print(f"R2: {r2_score(y_test, pred):.3f}")
destination = Path(__file__).resolve().parent / "diabetes_model.joblib"
joblib.dump(model, destination)
print(f"Saved model: {destination}")
""",
"requirements.txt": """Flask>=3.0,<4.0
joblib>=1.3,<2.0
numpy>=1.26,<3.0
scikit-learn>=1.4,<2.0
""",
"Dockerfile": """FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python train_model.py
EXPOSE 5000
CMD ["python", "app.py"]
""",
".dockerignore": """__pycache__/
*.py[cod]
.venv/
venv/
.git/
.pytest_cache/
.DS_Store
""",
"README.txt": """DAY 49 DOCKER ML API PROJECT

Prerequisites: Python, Docker Desktop, and an active Docker Engine.

1. Generate this folder by running the supplied generator script.
2. Open a terminal in this folder:
   cd docker_ml_api
3. Build the image:
   docker build -t diabetes-ml-api .
4. Run the container:
   docker run --name diabetes-api -p 5000:5000 diabetes-ml-api
5. Open the home endpoint:
   http://localhost:5000/
6. Send a test request with curl:
   curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d "{\\"features\\":[0.038,0.05,0.06,0.02,-0.04,-0.03,-0.04,-0.003,-0.01,0.02]}"

PowerShell alternative:
   Invoke-RestMethod -Uri http://localhost:5000/predict -Method Post -ContentType "application/json" -Body '{"features":[0.038,0.05,0.06,0.02,-0.04,-0.03,-0.04,-0.003,-0.01,0.02]}'

Useful commands:
   docker images
   docker ps
   docker ps -a
   docker logs diabetes-api
   docker stop diabetes-api
   docker start diabetes-api
   docker rm -f diabetes-api
   docker rmi diabetes-ml-api

Key concepts:
- Dockerfile: build instructions.
- Image: packaged application template.
- Container: running instance of an image.
- -p 5000:5000 maps host port 5000 to container port 5000.
- EXPOSE documents a port. It does not publish it by itself.
- 0.0.0.0 allows the API to listen on the container network interfaces.

Troubleshooting:
- Daemon error: start Docker Desktop and wait for its engine.
- Port conflict: run with -p 5001:5000 and visit localhost:5001.
- Container exits: inspect docker logs diabetes-api.
- API unreachable: verify container status, port mapping, and host=0.0.0.0.

This educational example uses scikit-learn's diabetes regression dataset.
It is not a diagnosis, medical risk assessment, or treatment tool.
Never use it for clinical decisions.
"""
}

project.mkdir(exist_ok=True)
for name, content in files.items():
    (project / name).write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")
print("Generated project files:")
for name in files:
    print(" -", project / name)
print("Follow README.txt to build and run the Docker container.")
