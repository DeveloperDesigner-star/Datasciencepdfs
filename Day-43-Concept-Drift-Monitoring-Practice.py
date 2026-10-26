"""
Day 43 Practice: Concept Drift and Model Performance Monitoring
Run: python Day-43-Concept-Drift-Monitoring-Practice.py
Tests: python Day-43-Concept-Drift-Monitoring-Practice.py --test
Dependencies: pandas, numpy
"""
import sys
from typing import Dict
import numpy as np
import pandas as pd


def build_demo_data() -> pd.DataFrame:
    """Synthetic regression predictions from two different weeks."""
    return pd.DataFrame({
        "period": ["Week 1"] * 5 + ["Week 2"] * 5,
        "actual": [100, 120, 130, 110, 140, 100, 120, 130, 110, 140],
        "predicted": [102, 118, 128, 112, 138, 115, 145, 100, 140, 165],
    })


def calculate_absolute_errors(data: pd.DataFrame) -> pd.DataFrame:
    """Validate input and add absolute prediction errors."""
    required = {"period", "actual", "predicted"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    result = data.copy()
    if result[["actual", "predicted"]].isna().any().any():
        raise ValueError("Actual and predicted values must not contain missing values.")
    result["absolute_error"] = (
        result["actual"].astype(float) - result["predicted"].astype(float)
    ).abs()
    return result


def summarize_mae(data: pd.DataFrame) -> pd.DataFrame:
    """Return sample counts and Mean Absolute Error (MAE) by period."""
    errors = calculate_absolute_errors(data)
    return (
        errors.groupby("period", sort=False)
        .agg(sample_count=("absolute_error", "size"),
             mae=("absolute_error", "mean"))
        .reset_index()
    )


def flag_performance_degradation(
    summary: pd.DataFrame, threshold: float, baseline_period: str = "Week 1"
) -> Dict[str, object]:
    """Flag when latest-period MAE rises more than threshold above baseline."""
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    indexed = summary.set_index("period")
    if baseline_period not in indexed.index:
        raise ValueError(f"Baseline period {baseline_period!r} was not found.")
    baseline_mae = float(indexed.loc[baseline_period, "mae"])
    latest_period = str(summary.iloc[-1]["period"])
    latest_mae = float(indexed.loc[latest_period, "mae"])
    increase = latest_mae - baseline_mae
    return {
        "baseline_period": baseline_period,
        "latest_period": latest_period,
        "baseline_mae": baseline_mae,
        "latest_mae": latest_mae,
        "mae_increase": increase,
        "warning": increase > threshold,
    }


def run_demo() -> None:
    """Run the monitoring workflow and print the interpretation."""
    data = build_demo_data()
    errors = calculate_absolute_errors(data)
    summary = summarize_mae(data)
    print("DAY 43: MODEL PERFORMANCE MONITORING")
    print("=" * 40)
    print("\nAbsolute errors:")
    print(errors.to_string(index=False))
    print("\nMAE by period:")
    print(summary.to_string(index=False))
    result = flag_performance_degradation(summary, threshold=10.0)
    print("\nBaseline comparison:")
    for key, value in result.items():
        print(f"{key}: {value}")
    if result["warning"]:
        print("\nWARNING: investigate the MAE increase, data quality, and pipeline.")
    print("A warning is a signal to investigate, not proof of concept drift or its cause.")


def run_tests() -> None:
    """Six lightweight assertions; no pytest installation is required."""
    data = build_demo_data()
    errors = calculate_absolute_errors(data)
    assert len(errors) == 10
    assert errors.loc[0, "absolute_error"] == 2
    assert errors.loc[5, "absolute_error"] == 15
    summary = summarize_mae(data)
    week1 = summary.loc[summary["period"] == "Week 1", "mae"].iloc[0]
    week2 = summary.loc[summary["period"] == "Week 2", "mae"].iloc[0]
    assert np.isclose(week1, 2.0)
    assert np.isclose(week2, 25.0)
    assert flag_performance_degradation(summary, 10.0)["warning"] is True
    assert flag_performance_degradation(summary, 25.0)["warning"] is False
    try:
        calculate_absolute_errors(pd.DataFrame({"period": ["x"], "actual": [1]}))
    except ValueError:
        pass
    else:
        raise AssertionError("Missing columns should raise ValueError")
    try:
        calculate_absolute_errors(pd.DataFrame({
            "period": ["x"], "actual": [np.nan], "predicted": [1]
        }))
    except ValueError:
        pass
    else:
        raise AssertionError("Missing values should raise ValueError")
    print("All tests passed.")


if __name__ == "__main__":
    if "--test" in sys.argv:
        run_tests()
    else:
        run_demo()
