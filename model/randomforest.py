import os
import sys

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd
from joblib import dump
from loaddata import load_data_from_csv

DATA_FILE_PATH = os.path.join("data", "diabetes_health_indicators_train.csv")
MODEL_FILE_PATH = os.path.join("model", "randomforest.pkl")

def build_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )

def randomforest():
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

    print("Random Forest – Validation Report")
    print(classification_report(y_val, y_pred, digits=4))
    if auc is not None:
        print(f"AUC: {auc:.4f}")

    os.makedirs(os.path.dirname(MODEL_FILE_PATH), exist_ok=True)
    dump(pipeline, MODEL_FILE_PATH, compress=3)
    print(f"Saved to {MODEL_FILE_PATH}")

if __name__ == "__main__":
    try:
        randomforest()
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)