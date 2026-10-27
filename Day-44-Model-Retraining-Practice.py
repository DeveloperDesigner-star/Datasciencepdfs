"""
Day 44: Model Retraining Strategies and Automated Retraining
Install: pip install numpy scikit-learn
Run demo: python Day-44-Model-Retraining-Practice.py
Run checks: python Day-44-Model-Retraining-Practice.py --test

Educational simulation only. Evaluate and approve candidates before deployment.
"""
import sys
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


def build_example_data():
    """Create older training data, newer training data, and a later holdout."""
    X_old = np.array([[1], [2], [3], [4], [5], [6]], dtype=float)
    y_old = np.array([7, 9, 11, 13, 15, 17], dtype=float)

    # Newer observations follow a changed relationship.
    X_new = np.array([[7], [8], [9], [10], [11], [12]], dtype=float)
    y_new = np.array([26, 29, 32, 35, 38, 41], dtype=float)

    X_test = np.array([[13], [14], [15]], dtype=float)
    y_test = np.array([44, 47, 50], dtype=float)
    return X_old, y_old, X_new, y_new, X_test, y_test


def evaluate_retraining_example():
    """Compare baseline and candidate models on the same holdout set."""
    X_old, y_old, X_new, y_new, X_test, y_test = build_example_data()

    # Baseline uses only historical observations.
    old_model = LinearRegression()
    old_model.fit(X_old, y_old)

    # Candidate retrains using historical plus newer observations.
    X_combined = np.vstack((X_old, X_new))
    y_combined = np.concatenate((y_old, y_new))
    candidate_model = LinearRegression()
    candidate_model.fit(X_combined, y_combined)

    # Fair comparison: both models use the same test inputs.
    old_predictions = old_model.predict(X_test)
    candidate_predictions = candidate_model.predict(X_test)
    old_mae = mean_absolute_error(y_test, old_predictions)
    candidate_mae = mean_absolute_error(y_test, candidate_predictions)

    return {
        "old_mae": float(old_mae),
        "candidate_mae": float(candidate_mae),
        "improved": bool(candidate_mae < old_mae),
        "old_predictions": old_predictions,
        "candidate_predictions": candidate_predictions,
    }


def print_report():
    result = evaluate_retraining_example()
    print("=" * 60)
    print("DAY 44: MODEL RETRAINING STRATEGIES")
    print("=" * 60)
    print(f"Old model MAE:       {result['old_mae']:.3f}")
    print(f"Candidate model MAE: {result['candidate_mae']:.3f}")
    print(f"Candidate improved:  {result['improved']}")
    print("\nPredictions on the same later-period holdout:")
    print("Actual values:       [44.0, 47.0, 50.0]")
    print("Old model:           ", np.round(result["old_predictions"], 2))
    print("Candidate model:     ", np.round(result["candidate_predictions"], 2))
    print("\nRetraining strategies:")
    print("1. Scheduled: retrain on a planned cadence.")
    print("2. Trigger-based: investigate meaningful alerts first.")
    print("3. Incremental: update in batches if the algorithm supports it.")
    print("4. Manual: an authorized person starts the process.")
    print("\nReminder: an alert is a signal to investigate, not automatic approval.")


def run_tests():
    data = build_example_data()
    assert len(data) == 6
    X_old, y_old, X_new, y_new, X_test, y_test = data
    assert X_old.shape == (6, 1) and y_old.shape == (6,)
    assert X_new.shape == (6, 1) and y_new.shape == (6,)
    assert X_test.shape == (3, 1) and y_test.shape == (3,)
    result = evaluate_retraining_example()
    assert result["old_mae"] >= 0 and result["candidate_mae"] >= 0
    assert result["improved"] is True
    assert len(result["old_predictions"]) == 3
    assert len(result["candidate_predictions"]) == 3
    print("All tests passed.")


if __name__ == "__main__":
    if "--test" in sys.argv:
        run_tests()
    else:
        print_report()
