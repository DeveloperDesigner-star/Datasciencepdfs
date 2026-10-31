"""
Day 48: Streamlit Machine Learning Prediction App
Install: python -m pip install streamlit scikit-learn joblib numpy
Run: streamlit run Day-48-Streamlit-ML-Prediction-App.py

Educational regression demonstration only. Not a medical diagnosis tool.
"""
from pathlib import Path
import joblib
import numpy as np
import streamlit as st
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

MODEL_PATH = Path(__file__).with_name("diabetes_model.joblib")
FEATURE_NAMES = [
    "Age", "Sex", "BMI", "Blood pressure",
    "Serum measurement 1", "Serum measurement 2",
    "Serum measurement 3", "Serum measurement 4",
    "Serum measurement 5", "Serum measurement 6",
]

@st.cache_resource
def load_or_train_model():
    """Load saved model, or train and save it on the first run."""
    if MODEL_PATH.exists():
        saved = joblib.load(MODEL_PATH)
        return saved["model"], saved.get("mae"), False

    dataset = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.20, random_state=42
    )
    model = RandomForestRegressor(
        n_estimators=100, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    mae = mean_absolute_error(y_test, model.predict(X_test))
    joblib.dump(
        {"model": model, "feature_names": FEATURE_NAMES, "mae": float(mae)},
        MODEL_PATH,
    )
    return model, float(mae), True

def predict_value(model, values):
    """Validate one row of ten finite numerical feature values."""
    array = np.asarray(values, dtype=float)
    if array.shape != (10,):
        raise ValueError("Exactly 10 feature values are required.")
    if not np.isfinite(array).all():
        raise ValueError("All values must be finite numbers.")
    return float(model.predict(array.reshape(1, -1))[0])

def main():
    st.set_page_config(
        page_title="Day 48 ML Prediction App",
        page_icon="📊",
        layout="centered",
    )
    st.title("Machine Learning Prediction App")
    st.caption("Day 48 practical project: Deploying a model with Streamlit")
    st.write(
        "Enter ten numerical feature values. The saved Random Forest "
        "regression model will generate an estimate."
    )
    st.info(
        "Educational demonstration only. The dataset target measures disease "
        "progression. This app does not diagnose diabetes or support clinical decisions."
    )

    try:
        model, mae, newly_trained = load_or_train_model()
    except Exception as error:
        st.error(f"Could not load or train the model: {error}")
        st.stop()

    if newly_trained:
        st.success(f"Model trained and saved. Hold-out MAE: {mae:.2f}")
    else:
        st.success("Saved model loaded successfully.")
        if mae is not None:
            st.metric("Stored test MAE", f"{mae:.2f}")

    st.subheader("Enter model features")
    st.caption(
        "The dataset features are scaled numerical values, not ordinary "
        "units such as years, kilograms, or mmHg."
    )

    values = []
    left, right = st.columns(2)
    for index, feature in enumerate(FEATURE_NAMES):
        with (left if index % 2 == 0 else right):
            values.append(st.number_input(
                feature, value=0.0, step=0.01, format="%.5f",
                key=f"feature_{index}"
            ))

    if st.button("Generate prediction", type="primary", use_container_width=True):
        try:
            result = predict_value(model, values)
            st.subheader("Prediction result")
            st.metric("Predicted target value", f"{result:.2f}")
            st.caption(
                "This is an educational model estimate, not a personal health assessment."
            )
        except (TypeError, ValueError) as error:
            st.error(str(error))

    with st.expander("How the app works"):
        st.markdown(
            "1. Streamlit collects ten values.\\n"
            "2. NumPy converts them into a numeric array.\\n"
            "3. The array is reshaped to one row with ten features.\\n"
            "4. The trained model generates a prediction.\\n"
            "5. Streamlit displays the result."
        )

if __name__ == "__main__":
    main()
