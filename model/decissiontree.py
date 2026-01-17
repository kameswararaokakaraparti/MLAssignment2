import os
import sys

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd
from joblib import dump
from loaddata import load_data_from_csv
from sklearn.tree import DecisionTreeClassifier

DATA_FILE_PATH = os.path.join("data", "diabetes_health_indicators_train.csv")
MODEL_FILE_PATH = os.path.join("model", "decissomtree.pkl")

def build_model() -> DecisionTreeClassifier:
    return DecisionTreeClassifier(criterion="entropy", random_state=42)

def dessionTree():
    ## Load data from file
    df = load_data_from_csv(DATA_FILE_PATH)

    # 2 Split features/target
    X = df.drop(columns=['Diabetes_binary'])
    y = df['Diabetes_binary'].astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline(steps=[
        ("model", build_model())
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_val)

    try:
        if y_val.nunique() == 2 and hasattr(pipeline, "predict_proba"):
            y_proba = pipeline.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_proba)
        else:
            auc = None
    except Exception:
        auc = None

    print("Logistic Regression – Validation Report")
    print(classification_report(y_val, y_pred, digits=4))
    if auc is not None:
        print(f"AUC: {auc:.4f}")

    os.makedirs(os.path.dirname(MODEL_FILE_PATH), exist_ok=True)
    dump(pipeline, MODEL_FILE_PATH)
    print(f"Saved to {MODEL_FILE_PATH}")

if __name__ == "__main__":
    try:
        dessionTree()
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)