"""
Day 40: Testing Machine Learning APIs with pytest
=================================================

Learning goals
- Build a small Flask prediction API.
- Test HTTP status codes and JSON responses.
- Test valid, missing, and invalid inputs.
- Use a fake model so tests are deterministic.

Install dependencies:
    python -m pip install flask pytest

Run tests from the folder containing this file:
    pytest -v Day-40-Flask-API-Testing-Practice.py

Important:
This example tests API behavior using a FakeModel. It does NOT evaluate
the accuracy of a trained machine-learning model.
"""

from flask import Flask, jsonify, request


class FakeModel:
    """A predictable stand-in for a trained ML model used in API tests."""

    def predict(self, rows):
        # The API passes a list containing one row: [study_hours, attendance].
        study_hours, attendance = rows[0]
        score = (study_hours * 10) + (attendance * 0.3)
        return [round(score, 2)]


def create_app(model=None):
    """Create the Flask app, allowing tests to inject a predictable model."""
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.config["MODEL"] = model if model is not None else FakeModel()

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"}), 200

    @app.post("/predict")
    def predict():
        # silent=True prevents malformed JSON from raising an HTML error page.
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({"error": "Request body must be a JSON object."}), 400

        required_fields = ("study_hours", "attendance")
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                "error": "Missing required field(s).",
                "missing_fields": missing_fields,
            }), 400

        # Reject booleans, strings, and other non-numeric values.
        values = [data["study_hours"], data["attendance"]]
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               for value in values):
            return jsonify({
                "error": "study_hours and attendance must be numeric."
            }), 400

        study_hours = float(data["study_hours"])
        attendance = float(data["attendance"])

        if not 0 <= study_hours <= 24:
            return jsonify({
                "error": "study_hours must be between 0 and 24."
            }), 400

        if not 0 <= attendance <= 100:
            return jsonify({
                "error": "attendance must be between 0 and 100."
            }), 400

        prediction = app.config["MODEL"].predict([[study_hours, attendance]])
        return jsonify({
            "prediction": float(prediction[0]),
            "input": {
                "study_hours": study_hours,
                "attendance": attendance,
            },
        }), 200

    return app


# A module-level app lets Flask tooling import the application if needed.
app = create_app()


def test_health_endpoint():
    """The health endpoint should respond with HTTP 200 and status=ok."""
    client = create_app().test_client()
    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_valid_prediction_returns_json():
    """Valid numeric inputs should produce a numeric prediction."""
    client = create_app().test_client()
    response = client.post(
        "/predict",
        json={"study_hours": 4, "attendance": 80},
    )

    assert response.status_code == 200
    body = response.get_json()
    assert isinstance(body["prediction"], (int, float))
    assert body["prediction"] == 64.0
    assert body["input"]["study_hours"] == 4.0
    assert body["input"]["attendance"] == 80.0


def test_missing_field_returns_400():
    """Missing required input must be rejected with a clear JSON error."""
    client = create_app().test_client()
    response = client.post("/predict", json={"study_hours": 4})

    assert response.status_code == 400
    assert "attendance" in response.get_json()["missing_fields"]


def test_attendance_over_100_returns_400():
    """Attendance above 100 percent is invalid."""
    client = create_app().test_client()
    response = client.post(
        "/predict",
        json={"study_hours": 4, "attendance": 140},
    )

    assert response.status_code == 400
    assert "attendance" in response.get_json()["error"]


def test_string_input_returns_400():
    """Numeric-looking strings are rejected instead of silently converted."""
    client = create_app().test_client()
    response = client.post(
        "/predict",
        json={"study_hours": "four", "attendance": 80},
    )

    assert response.status_code == 400
    assert "numeric" in response.get_json()["error"]


def test_study_hours_out_of_range_returns_400():
    """Study hours cannot be negative or exceed 24 hours per day."""
    client = create_app().test_client()
    response = client.post(
        "/predict",
        json={"study_hours": 25, "attendance": 80},
    )

    assert response.status_code == 400
    assert "study_hours" in response.get_json()["error"]


def test_non_object_json_returns_400():
    """A list or scalar JSON body should be rejected."""
    client = create_app().test_client()
    response = client.post(
        "/predict",
        data='["not", "an", "object"]',
        content_type="application/json",
    )

    assert response.status_code == 400
    assert "JSON object" in response.get_json()["error"]


if __name__ == "__main__":
    # Optional convenience: run this file directly to execute all tests.
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
