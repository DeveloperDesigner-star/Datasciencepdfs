"""Days 53-60 Customer Churn Analytics Capstone.
Synthetic educational data only. Not a production decision system.
Install: pip install pandas numpy scikit-learn joblib
Optional: pip install mlflow
Run: python Days-53-60-End-to-End-Churn-Analytics-Capstone.py
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

OUT = Path("churn_capstone_outputs")
OUT.mkdir(exist_ok=True)

def make_data(n=1200, seed=42):
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 73, n)
    charges = rng.normal(75, 22, n).clip(15, 180).round(2)
    calls = rng.poisson(1.6, n).clip(0, 10)
    contract = rng.choice(["Monthly", "One year", "Two year"], n, p=[.55,.28,.17])
    internet = rng.choice(["Fiber", "DSL", "None"], n, p=[.55,.35,.10])
    payment = rng.choice(["Card", "Bank transfer", "Cash"], n)
    age_group = rng.choice(["18-29", "30-44", "45-59", "60+"], n)
    z = (1.25 - .035*tenure + .018*(charges-70) + .32*calls
         + np.where(contract=="Monthly", .85, 0)
         + np.where(contract=="Two year", -.85, 0)
         + np.where(internet=="Fiber", .18, 0) + rng.normal(0,.65,n))
    prob = 1/(1+np.exp(-z))
    y = rng.binomial(1, prob)
    return pd.DataFrame({
        "tenure_months":tenure, "monthly_charges":charges,
        "support_calls":calls, "contract_type":contract,
        "internet_service":internet, "payment_method":payment,
        "age_group":age_group, "churn":y
    })

def build_pipeline():
    nums = ["tenure_months","monthly_charges","support_calls"]
    cats = ["contract_type","internet_service","payment_method","age_group"]
    numeric = Pipeline([("imputer",SimpleImputer(strategy="median")),
                        ("scale",StandardScaler())])
    categorical = Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),
                            ("encode",OneHotEncoder(handle_unknown="ignore"))])
    prep = ColumnTransformer([("num",numeric,nums),("cat",categorical,cats)])
    clf = RandomForestClassifier(n_estimators=250,max_depth=9,
        min_samples_leaf=3,class_weight="balanced",random_state=42)
    return Pipeline([("prep",prep),("model",clf)])

def main():
    print("Customer Churn Capstone. Synthetic data only.")
    data = make_data()
    data.to_csv(OUT/"synthetic_customer_data.csv",index=False)
    X, y = data.drop(columns="churn"), data["churn"]
    Xtr, Xte, ytr, yte = train_test_split(
        X,y,test_size=.2,random_state=42,stratify=y)
    model = build_pipeline()
    model.fit(Xtr,ytr)
    pred = model.predict(Xte)
    prob = model.predict_proba(Xte)[:,1]
    metrics = {
        "accuracy":float(accuracy_score(yte,pred)),
        "precision":float(precision_score(yte,pred,zero_division=0)),
        "recall":float(recall_score(yte,pred,zero_division=0)),
        "f1":float(f1_score(yte,pred,zero_division=0)),
        "roc_auc":float(roc_auc_score(yte,prob))
    }
    print("\\nMETRICS")
    for k,v in metrics.items(): print(f"{k}: {v:.3f}")
    print("\\nCLASSIFICATION REPORT\\n",classification_report(yte,pred,zero_division=0))
    print("CONFUSION MATRIX\\n",confusion_matrix(yte,pred))

    names = model.named_steps["prep"].get_feature_names_out()
    importance = pd.DataFrame({
        "feature":names,
        "importance":model.named_steps["model"].feature_importances_
    }).sort_values("importance",ascending=False)
    importance.to_csv(OUT/"feature_importance.csv",index=False)

    group = Xte[["age_group"]].copy()
    group["actual"] = yte.to_numpy()
    group["predicted"] = pred
    rows=[]
    for label,part in group.groupby("age_group"):
        rows.append({"age_group":label,"records":len(part),
            "actual_churn_rate":part.actual.mean(),
            "predicted_positive_rate":part.predicted.mean(),
            "accuracy":(part.actual==part.predicted).mean()})
    pd.DataFrame(rows).to_csv(OUT/"subgroup_review.csv",index=False)

    joblib.dump(model,OUT/"churn_pipeline.joblib")
    metadata={"project":"Customer Churn Capstone","version":"1.0.0",
              "data":"synthetic educational data","model":"RandomForestClassifier",
              "metrics":metrics,"seed":42}
    (OUT/"model_metadata.json").write_text(json.dumps(metadata,indent=2))
    predout=Xte.copy()
    predout["actual_churn"]=yte.to_numpy()
    predout["predicted_churn"]=pred
    predout["churn_probability"]=prob.round(4)
    predout.to_csv(OUT/"test_predictions.csv",index=False)

    # Simple mean-shift demonstration, not a complete drift detector.
    shifted=Xte.copy()
    shifted["monthly_charges"] *= 1.25
    drift=[]
    for col in ["tenure_months","monthly_charges","support_calls"]:
        a=float(Xtr[col].mean()); b=float(shifted[col].mean())
        drift.append({"feature":col,"train_mean":a,"new_mean":b,
                      "relative_change":abs(b-a)/(abs(a)+1e-9),
                      "flag_over_20_percent":abs(b-a)/(abs(a)+1e-9)>.20})
    pd.DataFrame(drift).to_csv(OUT/"basic_drift_report.csv",index=False)

    # Optional MLflow logging.
    try:
        import mlflow
        mlflow.set_experiment("Days-53-60-Churn-Capstone")
        with mlflow.start_run():
            mlflow.log_params({"trees":250,"max_depth":9,"min_samples_leaf":3})
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(OUT/"feature_importance.csv"))
        print("MLflow run logged. Use `mlflow ui` to view it.")
    except ImportError:
        print("Optional MLflow not installed. Install with: pip install mlflow")

    reloaded=joblib.load(OUT/"churn_pipeline.joblib")
    print("Reload check, sample prediction:",int(reloaded.predict(Xte.iloc[[0]])[0]))
    print("Saved outputs to:",OUT.resolve())

if __name__=="__main__":
    main()
