"""
Day 46: Introduction to Model Deployment
Install: python -m pip install scikit-learn joblib
Run: python Day-46-Model-Deployment-Practice.py

Educational example only. The diabetes dataset target represents disease
progression. This model is not a diagnostic tool or medical advice.
"""
from pathlib import Path
import joblib
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Load input features (X) and the target (y).
dataset = load_diabetes()
X, y = dataset.data, dataset.target

# Hold back unseen rows for a fair, simple evaluation.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train a reproducible Random Forest regression model.
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model before making it available for use.
test_predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, test_predictions)
print(f"Test Mean Absolute Error (MAE): {mae:.2f}")

# Save the fitted model so it can be reused without retraining.
model_path = Path("diabetes_model.joblib")
joblib.dump(model, model_path)
print(f"Saved model to: {model_path.resolve()}")

# Reload the stored model, as a separate application could do.
loaded_model = joblib.load(model_path)

# Demonstrate inference using a held-out sample.
# In a real application, use validated new input with the same feature schema.
sample = X_test[0].reshape(1, -1)
prediction = loaded_model.predict(sample)[0]
print(f"Predicted target for demo sample: {prediction:.2f}")
print("Feature order:", ", ".join(dataset.feature_names))
print("Demo complete. This is a regression example, not a medical diagnosis.")
