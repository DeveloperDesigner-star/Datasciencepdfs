"""
DAY 31: Model Evaluation, Cross-Validation, and Hyperparameter Tuning
Install: pip install scikit-learn scipy
Run: python Day-31-Python-Practice.py

The built-in Breast Cancer Wisconsin dataset is used for educational classification
practice only. This script is not a medical diagnostic tool.
"""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score,
    GridSearchCV, RandomizedSearchCV
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from scipy.stats import randint

def main():
    data = load_breast_cancer()
    X, y = data.data, data.target
    print("Dataset shape:", X.shape)
    print("Classes:", data.target_names)

    # Keep the final test set untouched during model selection.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    baseline = RandomForestClassifier(random_state=42)

    # Baseline cross-validation on training data.
    scores = cross_val_score(
        baseline, X_train, y_train, cv=cv,
        scoring="accuracy", n_jobs=-1
    )
    print("\nBaseline fold scores:", scores.round(4))
    print("Mean CV accuracy:", round(scores.mean(), 4))
    print("CV standard deviation:", round(scores.std(), 4))

    # GridSearchCV tests every combination: 2 x 3 = 6 combinations.
    grid = GridSearchCV(
        estimator=baseline,
        param_grid={
            "n_estimators": [100, 200],
            "max_depth": [None, 5, 10],
        },
        cv=cv, scoring="accuracy", n_jobs=-1
    )
    grid.fit(X_train, y_train)
    print("\nGrid best parameters:", grid.best_params_)
    print("Grid best mean CV accuracy:", round(grid.best_score_, 4))

    # Randomized search samples a limited number of candidates.
    random_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        param_distributions={
            "n_estimators": randint(100, 401),
            "max_depth": [None, 5, 10, 15, 20],
            "min_samples_split": randint(2, 11),
            "min_samples_leaf": randint(1, 6),
        },
        n_iter=10, cv=cv, scoring="accuracy",
        random_state=42, n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    print("\nRandomized search best parameters:", random_search.best_params_)
    print("Randomized search best mean CV accuracy:",
          round(random_search.best_score_, 4))

    # Final evaluation only after selecting the model.
    model = grid.best_estimator_
    pred = model.predict(X_test)
    print("\nFINAL HELD-OUT TEST RESULTS")
    print("Accuracy:", round(accuracy_score(y_test, pred), 4))
    print("Precision (class 1):", round(precision_score(y_test, pred), 4))
    print("Recall (class 1):", round(recall_score(y_test, pred), 4))
    print("F1-score (class 1):", round(f1_score(y_test, pred), 4))
    print("\nConfusion matrix:\n", confusion_matrix(y_test, pred))
    print("\nClassification report:\n")
    print(classification_report(y_test, pred, target_names=data.target_names))

if __name__ == "__main__":
    main()
