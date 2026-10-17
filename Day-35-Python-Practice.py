"""Day 35: Hyperparameter Tuning and Model Optimization

Install if needed: pip install scikit-learn scipy
Run: python Day-35-Python-Practice.py

This educational example uses scikit-learn's built-in Breast Cancer Wisconsin
classification dataset. It is for learning model evaluation only, NOT diagnosis.
"""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from scipy.stats import randint

# 1. Load data and create a final holdout test set.
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 2. GridSearchCV tests every specified combination.
# 3 * 3 * 2 = 18 combinations, with 5 CV folds each = 90 fits, plus refit.
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5],
}
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1,
    refit=True,
)
grid_search.fit(X_train, y_train)
print("=== GRID SEARCH ===")
print("Best parameters:", grid_search.best_params_)
print("Best mean CV F1:", round(grid_search.best_score_, 4))

# 3. Test only after choosing settings with training data and CV.
grid_model = grid_search.best_estimator_
grid_pred = grid_model.predict(X_test)
grid_prob = grid_model.predict_proba(X_test)[:, 1]
print("Holdout accuracy:", round(accuracy_score(y_test, grid_pred), 4))
print("Holdout ROC-AUC:", round(roc_auc_score(y_test, grid_prob), 4))
print("Confusion matrix:\n", confusion_matrix(y_test, grid_pred))
print("Classification report:\n", classification_report(y_test, grid_pred, target_names=data.target_names))

# 4. RandomizedSearchCV samples a fixed number of combinations from ranges.
param_distributions = {
    "n_estimators": randint(100, 501),       # integers 100 through 500
    "max_depth": [None, 5, 10, 15, 20, 30],
    "min_samples_split": randint(2, 11),    # integers 2 through 10
    "min_samples_leaf": randint(1, 6),      # integers 1 through 5
    "max_features": ["sqrt", "log2", None],
}
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_distributions,
    n_iter=20,
    scoring="f1",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    refit=True,
)
random_search.fit(X_train, y_train)
print("\n=== RANDOMIZED SEARCH ===")
print("Best parameters:", random_search.best_params_)
print("Best mean CV F1:", round(random_search.best_score_, 4))

# Optional: compare the two selected models on the SAME holdout set.
random_pred = random_search.best_estimator_.predict(X_test)
print("Randomized-search holdout accuracy:", round(accuracy_score(y_test, random_pred), 4))

# Reflection: Which metric matters most for the problem, and why?
# Never tune based on the holdout test score. Keep the test set for final evaluation.
