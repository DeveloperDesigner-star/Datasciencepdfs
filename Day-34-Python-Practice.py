"""
Day 34: Ensemble Learning practice
Install: pip install scikit-learn
Run: python Day-34-Python-Practice.py
Educational classification practice only. Not a medical diagnostic tool.
"""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier

def evaluate(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"\\n{name}")
    print(f"Accuracy : {accuracy_score(y_test, pred):.4f}")
    print(f"Precision: {precision_score(y_test, pred, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_test, pred, zero_division=0):.4f}")
    print(f"F1-score : {f1_score(y_test, pred, zero_division=0):.4f}")
    print(classification_report(y_test, pred, target_names=["malignant", "benign"], zero_division=0))
    return model

def main():
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print("Dataset shape:", X.shape, "| Features:", X.shape[1])
    print("Educational dataset only, not for diagnosis.")

    models = [
        ("Decision Tree", DecisionTreeClassifier(random_state=42)),
        ("Bagging (100 trees)", BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=42),
            n_estimators=100, random_state=42, n_jobs=-1)),
        ("Random Forest (100 trees)", RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1)),
        ("Gradient Boosting", GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, random_state=42)),
    ]
    for name, model in models:
        evaluate(name, model, X_train, X_test, y_train, y_test)

    oob = RandomForestClassifier(n_estimators=200, oob_score=True, random_state=42, n_jobs=-1)
    oob.fit(X_train, y_train)
    print(f"\\nRandom Forest OOB score: {oob.oob_score_:.4f}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(
        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        X_train, y_train, cv=cv, scoring="accuracy"
    )
    print("5-fold CV scores:", [round(float(s), 4) for s in scores])
    print(f"Mean CV accuracy: {scores.mean():.4f}")
    print("Use CV on training data for model selection, then evaluate on the held-out test set.")

if __name__ == "__main__":
    main()
