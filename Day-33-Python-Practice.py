"""
Day 33: Feature Selection and Dimensionality Reduction.
Educational use only. This built-in dataset is not a medical diagnostic tool.
"""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Filter method: choose 10 original features by ANOVA F-score.
selection_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("select", SelectKBest(score_func=f_classif, k=10)),
    ("model", LogisticRegression(max_iter=5000)),
])
selection_pipeline.fit(X_train, y_train)
pred = selection_pipeline.predict(X_test)
selector = selection_pipeline.named_steps["select"]
print("Original feature count:", X_train.shape[1])
print("Selected feature count:", len(X_train.columns[selector.get_support()]))
print("Selected names:", list(X_train.columns[selector.get_support()]))
print("Feature-selection test accuracy:", round(accuracy_score(y_test, pred), 4))
print(classification_report(y_test, pred, zero_division=0))

cv = cross_val_score(selection_pipeline, X_train, y_train, cv=5, scoring="accuracy")
print("Five-fold training CV scores:", cv)
print("Mean CV accuracy:", round(cv.mean(), 4))

# PCA transforms features into components. Fit only on training data.
pca_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=0.95)),
    ("model", LogisticRegression(max_iter=5000)),
])
pca_pipeline.fit(X_train, y_train)
pca_pred = pca_pipeline.predict(X_test)
print("\nPCA components retaining >=95% training variance:",
      pca_pipeline.named_steps["pca"].n_components_)
print("PCA workflow test accuracy:", round(accuracy_score(y_test, pca_pred), 4))

# Wrapper method: Recursive Feature Elimination.
rfe = RFE(LogisticRegression(max_iter=5000), n_features_to_select=10)
rfe.fit(X_train, y_train)
print("RFE selected names:", list(X_train.columns[rfe.get_support()]))

# Scores are educational examples, not clinical or deployment claims.
