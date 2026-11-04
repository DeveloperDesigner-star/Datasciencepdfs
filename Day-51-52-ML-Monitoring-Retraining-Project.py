"""Day 51 + 52: ML Model Monitoring and Automated Retraining.
Install: pip install numpy pandas scikit-learn joblib
Run: python Day-51-52-ML-Monitoring-Retraining-Project.py

Educational example using scikit-learn's diabetes regression dataset.
Not a medical diagnostic tool. Drift alerts require investigation.
"""
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

OUT = Path("day51_52_outputs")
OUT.mkdir(exist_ok=True)
MODEL_PATH = OUT / "approved_model.joblib"
LOG_PATH = OUT / "monitoring_log.csv"
DRIFT_PATH = OUT / "drift_report.csv"
DRIFT_THRESHOLD = 0.5
MAE_LIMIT = 60.0  # Demonstration threshold only. Set from validation/business needs.

def build_pipeline(trees: int) -> Pipeline:
    """Combine median imputation and Random Forest into one reusable artifact."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=trees, random_state=42, n_jobs=-1
        ))
    ])

def detect_mean_shift(reference: pd.DataFrame, current: pd.DataFrame,
                      threshold: float = DRIFT_THRESHOLD) -> pd.DataFrame:
    """Simple standardized mean-shift check, not a complete drift detector."""
    records = []
    for col in reference.columns:
        ref_mean = float(reference[col].mean())
        cur_mean = float(current[col].mean())
        ref_std = float(reference[col].std(ddof=0))
        score = (abs(cur_mean-ref_mean)/ref_std) if ref_std > 0 else (
            0.0 if cur_mean == ref_mean else float("inf"))
        records.append({
            "feature": col, "reference_mean": ref_mean,
            "current_mean": cur_mean, "standardized_mean_shift": score,
            "possible_drift": score > threshold
        })
    return pd.DataFrame(records)

def log_result(version: str, mae: float, status: str, drift_count: int) -> None:
    row = pd.DataFrame([{
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": version, "mae": round(float(mae), 4),
        "status": status, "features_with_possible_drift": drift_count
    }])
    row.to_csv(LOG_PATH, mode="a" if LOG_PATH.exists() else "w",
               header=not LOG_PATH.exists(), index=False)

def main() -> None:
    print("DAY 51 + 52: MODEL MONITORING AND RETRAINING")
    dataset = load_diabetes(as_frame=True)
    X, y = dataset.data, dataset.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Day 51: train and monitor a baseline.
    baseline = build_pipeline(50)
    baseline.fit(X_train, y_train)
    baseline_mae = mean_absolute_error(y_test, baseline.predict(X_test))
    print(f"Baseline MAE: {baseline_mae:.2f}")

    # Simulate incoming data with a shift in one feature.
    incoming = X_test.copy()
    incoming["bmi"] = incoming["bmi"] + 0.08
    drift = detect_mean_shift(X_train, incoming)
    drift_count = int(drift["possible_drift"].sum())
    print(f"Features with possible mean shift: {drift_count}")
    print(drift.to_string(index=False))

    # Day 52: retrain a candidate and compare it with the baseline.
    candidate = build_pipeline(150)
    candidate.fit(X_train, y_train)
    candidate_mae = mean_absolute_error(y_test, candidate.predict(X_test))
    print(f"Candidate MAE: {candidate_mae:.2f}")

    # Candidate must improve and meet an illustrative acceptance limit.
    if candidate_mae < baseline_mae and candidate_mae <= MAE_LIMIT:
        approved, version, status, score = candidate, "candidate_v2", "APPROVED", candidate_mae
    else:
        approved, version, status, score = baseline, "baseline_v1", "BASELINE_RETAINED", baseline_mae

    joblib.dump(approved, MODEL_PATH)
    log_result(version, score, status, drift_count)
    drift.to_csv(DRIFT_PATH, index=False)

    # Reload the saved complete pipeline and make sample predictions.
    loaded = joblib.load(MODEL_PATH)
    print("Sample predictions:", np.round(loaded.predict(X_test.iloc[:5]), 2))
    print("Approved version:", version, "| Status:", status)
    print("Created:", MODEL_PATH, LOG_PATH, DRIFT_PATH)
    print("Review all validation results before production use.")

if __name__ == "__main__":
    main()
