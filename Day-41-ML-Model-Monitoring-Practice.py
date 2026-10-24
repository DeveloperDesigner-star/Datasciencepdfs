"""
Day 41: Monitoring Machine Learning Models in Production
Install: python -m pip install pytest
Run: pytest -v Day-41-ML-Model-Monitoring-Practice.py
This educational example uses demo models, not a trained ML model.
"""
import time
from statistics import mean

class DemoModel:
    def predict(self, rows):
        study_hours, attendance = rows[0]
        return [(study_hours * 10) + (attendance * 0.3)]

class BrokenDemoModel:
    def predict(self, rows):
        raise RuntimeError("Example internal failure")

def monitor_prediction(model, study_hours, attendance):
    """Record prediction status and elapsed time without exposing raw errors."""
    start = time.perf_counter()
    try:
        prediction = model.predict([[study_hours, attendance]])
        elapsed = (time.perf_counter() - start) * 1000
        return {"prediction": float(prediction[0]),
                "response_time_ms": round(elapsed, 2), "status": "success"}
    except Exception as error:
        elapsed = (time.perf_counter() - start) * 1000
        return {"prediction": None, "response_time_ms": round(elapsed, 2),
                "status": "error", "error_type": type(error).__name__}

def mean_absolute_error(actual, predicted):
    """Calculate MAE and reject empty or unequal lists."""
    if not actual or len(actual) != len(predicted):
        raise ValueError("Lists must be non-empty and have equal length.")
    return mean(abs(a-p) for a, p in zip(actual, predicted))

def mean_shift(training_values, production_values):
    """Simple absolute difference between means, not a formal drift test."""
    if not training_values or not production_values:
        raise ValueError("Both samples must be non-empty.")
    return abs(mean(training_values) - mean(production_values))

def test_successful_prediction_is_monitored():
    result = monitor_prediction(DemoModel(), 4, 80)
    assert result["status"] == "success"
    assert result["prediction"] == 64.0
    assert result["response_time_ms"] >= 0

def test_model_exception_is_recorded_safely():
    result = monitor_prediction(BrokenDemoModel(), 4, 80)
    assert result["status"] == "error"
    assert result["prediction"] is None
    assert result["error_type"] == "RuntimeError"
    assert "Example internal failure" not in str(result)

def test_mae_calculation():
    assert round(mean_absolute_error([70,80,90], [65,85,90]), 2) == 3.33

def test_mae_rejects_empty_data():
    try:
        mean_absolute_error([], [])
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")

def test_mean_shift_indicator():
    assert mean_shift([1,2,3], [4,5,6]) == 3

def test_mean_shift_rejects_empty_sample():
    try:
        mean_shift([1,2], [])
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")

if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
