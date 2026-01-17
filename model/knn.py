import os
import sys

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd
from joblib import dump
from loaddata import load_data_from_csv
from sklearn.preprocessing import StandardScaler

DATA_FILE_PATH = os.path.join("data", "diabetes_health_indicators_train.csv")
MODEL_FILE_PATH = os.path.join("model", "knn.pkl")

def build_model() -> KNeighborsClassifier:
    return KNeighborsClassifier(
        n_neighbors=7,
        weights="distance",
        p=2,
        metric="minkowski"
    )

def knn():
    ## Load data from file
    df = load_data_from_csv(DATA_FILE_PATH)

    # 2 Split features/target
    X = df.drop(columns=['Diabetes_binary'])
    y = df['Diabetes_binary'].astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, train_size=50000, random_state=42, stratify=y
    )

    ##pipeline = Pipeline(steps=[
    ##    ("model", build_model())
    ##])

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=10, random_state=42)),
        ("knn", KNeighborsClassifier(
            n_neighbors=7,
            weights="distance",
            algorithm="kd_tree",
            leaf_size=40,
            n_jobs=-1
        ))
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
        knn()
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)