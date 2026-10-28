"""
Day 45: Model Versioning and Experiment Tracking
Install: python -m pip install pandas scikit-learn joblib
Run: python Day-45-Model-Versioning-and-Experiment-Tracking-Practice.py
"""
from pathlib import Path
from datetime import datetime, timezone
import json
import joblib
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

OUTPUT_DIR = Path("day45_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
SEED = 42

def load_and_split_data():
    """Load a built-in regression dataset and make a reproducible split."""
    dataset = load_diabetes(as_frame=True)
    # Keep this split identical for each candidate so scores are comparable.
    return train_test_split(dataset.data, dataset.target,
                            test_size=0.2, random_state=SEED)

def main():
    X_train, X_test, y_train, y_test = load_and_split_data()
    records = []
    best_model = None
    best_record = None
    best_mae = float("inf")

    for tree_count in [50, 100, 200]:
        # Parameters are settings selected before fitting.
        model = RandomForestRegressor(n_estimators=tree_count,
                                      random_state=SEED, n_jobs=-1)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # MAE is a metric. Lower is better when evaluation data is comparable.
        mae = float(mean_absolute_error(y_test, predictions))
        record = {
            "experiment_id": f"rf-trees-{tree_count}",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model_type": "RandomForestRegressor",
            "parameters": json.dumps({"n_estimators": tree_count,
                                      "random_state": SEED}, sort_keys=True),
            "mae": round(mae, 4),
            "dataset": "sklearn_diabetes_builtin",
            "split_seed": SEED,
        }
        records.append(record)
        print(f"{record['experiment_id']}: MAE = {mae:.4f}")

        if mae < best_mae:
            best_mae, best_model, best_record = mae, model, record

    # Persist the experiment log, selected model artifact, and selection metadata.
    pd.DataFrame(records).to_csv(OUTPUT_DIR / "experiment_log.csv", index=False)
    joblib.dump(best_model, OUTPUT_DIR / "best_model.joblib")
    metadata = {
        "selected_experiment_id": best_record["experiment_id"],
        "selected_mae": best_record["mae"],
        "selection_rule": "Lowest MAE among tested candidates",
        "dataset": best_record["dataset"],
        "split_seed": SEED,
        "caution": "For robust model selection, use validation data and reserve a separate test set for final evaluation."
    }
    (OUTPUT_DIR / "best_model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("\nExperiments sorted by MAE:")
    print(pd.DataFrame(records).sort_values("mae")[["experiment_id", "mae"]].to_string(index=False))
    print(f"\nSelected: {best_record['experiment_id']} (MAE {best_record['mae']})")
    print(f"Files saved in: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
