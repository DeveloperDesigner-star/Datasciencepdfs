"""
Day 32: Feature Engineering and Scikit-learn Pipelines
Educational practice only. Tiny synthetic data demonstrates workflow, not
reliable real-world prediction or model performance.
"""
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

data = {
    "study_hours": [2, 5, 1, 7, 4, 3, 8, 2, 6, 5, 1, 7, 4, 6, 3, 8],
    "attendance": [55, 80, 40, 95, 70, 65, 98, 50, 88, 75, 45, 92, 72, 85, 60, 97],
    "department": ["CS", "IT", "CS", "SE", "IT", "CS", "SE", "IT",
                   "CS", "SE", "IT", "CS", "SE", "IT", "CS", "SE"],
    "passed": [0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
}
df = pd.DataFrame(data)
df.loc[2, "study_hours"] = float("nan")
df.loc[7, "attendance"] = float("nan")
df.loc[10, "department"] = None

X = df.drop(columns="passed")
y = df["passed"]
numeric_features = ["study_hours", "attendance"]
categorical_features = ["department"]

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])
preprocessor = ColumnTransformer(transformers=[
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight="balanced"
    )),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring="accuracy")
print("Cross-validation accuracy scores:", cv_scores)
print("Mean CV accuracy:", round(cv_scores.mean(), 3))

model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("\nHeld-out test accuracy:", round(accuracy_score(y_test, predictions), 3))
print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

new_student = pd.DataFrame([{
    "study_hours": 4.5, "attendance": 78, "department": "AI"
}])
print("\nPrediction for new example (0 = not passed, 1 = passed):",
      int(model.predict(new_student)[0]))

# This tiny invented dataset is for practice only. Scores are unstable and
# are not evidence of real-world accuracy.
