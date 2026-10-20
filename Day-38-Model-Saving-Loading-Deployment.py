"""
Day 38: Model Saving, Loading, and Deployment Basics
60-Day Data Science Learning Journey

Educational project using a small fictional student dataset.
This is not suitable for real academic decisions.

Install:
    pip install pandas scikit-learn joblib matplotlib

Run:
    python Day-38-Model-Saving-Loading-Deployment.py

Security:
    Only load joblib/pickle files from trusted sources. Loading untrusted
    serialized files can execute malicious code.

The script:
1. Creates fictional data.
2. Splits data into train/test sets.
3. Builds a preprocessing + model pipeline.
4. Trains and evaluates the model.
5. Saves the entire pipeline with joblib.
6. Loads it and predicts for a fictional new example.
7. Demonstrates a model reload consistency check.
"""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

RANDOM_STATE = 42
MODEL_PATH = Path("student_result_model.joblib")


def make_dataset() -> pd.DataFrame:
    """Return a small, fictional dataset for learning only."""
    data = {
        "study_hours": [
            1, 2, 2, 3, 3, 4, 4, 5,
            5, 6, 6, 7, 7, 8, 8, 9,
            2, 3, 4, 5, 6, 7, 8, 9,
        ],
        "attendance": [
            40, 50, 55, 45, 60, 55, 65, 60,
            70, 65, 75, 70, 80, 75, 90, 95,
            48, 58, 62, 68, 72, 78, 88, 92,
        ],
        "result": [
            "Needs Support", "Needs Support", "Needs Support",
            "Needs Support", "Needs Support", "Needs Support",
            "Pass", "Needs Support", "Pass", "Pass", "Pass",
            "Pass", "Pass", "Pass", "Pass", "Pass",
            "Needs Support", "Needs Support", "Needs Support",
            "Pass", "Pass", "Pass", "Pass", "Pass",
        ],
    }
    return pd.DataFrame(data)


def build_pipeline() -> Pipeline:
    """Bundle feature scaling and classification in one reusable pipeline."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def main() -> None:
    df = make_dataset()
    feature_columns = ["study_hours", "attendance"]
    X = df[feature_columns]
    y = df["result"]

    print("=" * 68)
    print("DAY 38: MODEL SAVING, LOADING, AND DEPLOYMENT BASICS")
    print("=" * 68)
    print("\nDataset preview:")
    print(df.head())
    print(f"\nDataset shape: {df.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"\nTraining examples: {len(X_train)}")
    print(f"Testing examples: {len(X_test)}")

    model = build_pipeline()
    model.fit(X_train, y_train)

    test_predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, test_predictions)

    print("\nEvaluation on held-out test data")
    print(f"Accuracy: {accuracy:.3f}")
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            test_predictions,
            zero_division=0,
        )
    )
    print("Confusion matrix (labels follow model.classes_ order):")
    print(model.classes_)
    print(confusion_matrix(y_test, test_predictions, labels=model.classes_))

    # Save the complete pipeline, including the scaler and trained classifier.
    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved trained pipeline to: {MODEL_PATH.resolve()}")

    # Load the trusted file back into memory.
    loaded_model = joblib.load(MODEL_PATH)
    print("Loaded the saved pipeline successfully.")

    # New input must use the same feature names and compatible data types.
    new_student = pd.DataFrame(
        {
            "study_hours": [6],
            "attendance": [80],
        }
    )
    prediction = loaded_model.predict(new_student)[0]
    print("\nPrediction for a fictional example:")
    print(new_student.to_string(index=False))
    print(f"Predicted category: {prediction}")

    # Optional probability output is model confidence, not a guarantee.
    if hasattr(loaded_model, "predict_proba"):
        probabilities = loaded_model.predict_proba(new_student)[0]
        probability_map = dict(zip(loaded_model.classes_, probabilities))
        print("Model-estimated class probabilities:")
        for label, probability in probability_map.items():
            print(f"  {label}: {probability:.3f}")

    # A quick consistency check: reloaded model should reproduce predictions.
    before_save_prediction = model.predict(new_student)[0]
    after_load_prediction = loaded_model.predict(new_student)[0]
    print("\nReload consistency check:")
    print(f"Before saving: {before_save_prediction}")
    print(f"After loading: {after_load_prediction}")
    print(f"Same prediction: {before_save_prediction == after_load_prediction}")

    print(
        "\nResponsible-use note: this is a tiny synthetic educational example. "
        "Do not use it to make real decisions about students."
    )


if __name__ == "__main__":
    main()
