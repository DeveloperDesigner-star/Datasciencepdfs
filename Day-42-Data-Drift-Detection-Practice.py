"""Day 42: Data drift detection practice.
Install: python -m pip install pandas numpy pytest
Run: pytest -v Day-42-Data-Drift-Detection-Practice.py
Synthetic learning data only. A drift warning does not prove model degradation.
"""
import pandas as pd
import numpy as np

def compare_feature(training_values, production_values):
    """Summarize means and standard deviations and calculate absolute mean shift."""
    train = pd.Series(training_values, dtype="float64")
    prod = pd.Series(production_values, dtype="float64")
    if train.empty or prod.empty:
        raise ValueError("Both datasets must contain at least one value.")
    if train.isna().any() or prod.isna().any():
        raise ValueError("Handle missing values before comparing distributions.")
    return {
        "training_mean": float(train.mean()),
        "production_mean": float(prod.mean()),
        "mean_difference": float(abs(train.mean() - prod.mean())),
        "training_std": float(train.std()) if len(train) > 1 else 0.0,
        "production_std": float(prod.std()) if len(prod) > 1 else 0.0,
    }

def check_mean_shift(stats, threshold=2.0):
    """Simple threshold rule, not a formal statistical drift test."""
    if threshold < 0:
        raise ValueError("Threshold must be zero or greater.")
    return stats["mean_difference"] > threshold

def example():
    training = pd.DataFrame({"study_hours":[1,2,2,3,3,4,4,5]})
    production = pd.DataFrame({"study_hours":[4,5,6,6,7,8,8,9]})
    stats = compare_feature(training["study_hours"], production["study_hours"])
    for key, value in stats.items():
        print(f"{key}: {value:.3f}")
    print("Warning: Possible data drift detected!" if check_mean_shift(stats)
          else "No mean-shift warning.")
    return stats

def test_example_means():
    s = compare_feature([1,2,2,3,3,4,4,5], [4,5,6,6,7,8,8,9])
    assert s["training_mean"] == 3.0
    assert s["production_mean"] == 6.625
    assert s["mean_difference"] == 3.625

def test_threshold_warning():
    assert check_mean_shift(compare_feature([1,2,3],[5,6,7]), 2.0)

def test_no_warning_under_threshold():
    assert not check_mean_shift(compare_feature([1,2,3],[2,3,4]), 2.0)

def test_empty_sample_rejected():
    try:
        compare_feature([], [1,2])
    except ValueError as e:
        assert "at least one value" in str(e)
    else:
        raise AssertionError("Expected ValueError")

def test_missing_values_rejected():
    try:
        compare_feature([1,np.nan,3], [2,3,4])
    except ValueError as e:
        assert "missing values" in str(e)
    else:
        raise AssertionError("Expected ValueError")

def test_negative_threshold_rejected():
    stats = compare_feature([1,2,3], [4,5,6])
    try:
        check_mean_shift(stats, -1)
    except ValueError as e:
        assert "zero or greater" in str(e)
    else:
        raise AssertionError("Expected ValueError")

if __name__ == "__main__":
    example()
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
