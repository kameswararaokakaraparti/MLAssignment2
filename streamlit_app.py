#Stremlit app file
import sys
from sklearn.metrics import accuracy_score
from sklearn.metrics import RocCurveDisplay, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
import streamlit as st
import pandas as pd
from joblib import load
import matplotlib.pyplot as plt
import seaborn as sns

#set title
st.set_page_config(
    page_title="ML Assignment2 with multi model selection",
    layout="wide"
)

MODEL_LIST = {
    "Logistic Regression": "model/logisticreg.pkl",
    "Decision Tree Classifier": "model/decissomtree.pkl",
    "Naive Bayes Classifier": "model/naivebayes.pkl",
    "K-Nearest Neighbor Classifier": "model/knn.pkl",
    "Random Forest": "model/randomforest.pkl",
    "XGBoost": "model/xgboostclassifier.pkl"
}

@st.cache_resource
def load_pipeline(path: str):
    return load(path)

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('')
    return fig

def plot_roc(y_true, y_proba):
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax, curve_kwargs={'color': 'green'})
    ax.set_title('')
    return fig

def calculate_metrics(y_true, y_pred, y_proba=None, average='binary'):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0),
        'mcc': matthews_corrcoef(y_true, y_pred),
    }
    if y_proba is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_proba)
        except Exception:
            metrics['auc'] = None
    else:
        metrics['auc'] = None
    return metrics

def render_results(model_name: str, y_true, y_pred, y_proba):
    metrics = calculate_metrics(y_true, y_pred, y_proba)

    st.markdown("### Evaluation")
    kpi_cols = st.columns(6)

    acc = round(metrics["accuracy"], 3)
    prec = round(metrics["precision"], 3)
    rec = round(metrics["recall"], 3)
    f1 = round(metrics["f1"], 3)
    auc = round(metrics["auc"], 3)
    mcc = round(metrics["mcc"], 3)

    kpi_cols[0].metric("Accuracy", f"{acc:.3f}")
    kpi_cols[1].metric("Precision", f"{prec:.3f}")
    kpi_cols[2].metric("Recall", f"{rec:.3f}")
    kpi_cols[3].metric("F1 Score", f"{f1:.3f}")
    kpi_cols[4].metric("AUC", f"{auc:.3f}")
    kpi_cols[5].metric("MCC", f"{mcc:.3f}")

    st.markdown("### Visualizations")

    col1, col2 = st.columns(2)

    with col1:
        st.caption("Confusion Matrix")
        fig_cm = plot_confusion_matrix(y_true, y_pred)
        try:
            fig_cm.set_size_inches(3.6, 3.2)
        except Exception:
            pass
        st.pyplot(fig_cm, use_container_width=True)

    with col2:
        st.caption("ROC Curve")
        fig_roc = plot_roc(y_true, y_proba)
        try:
            fig_roc.set_size_inches(3.6, 3.2)
        except Exception:
            pass
        st.pyplot(fig_roc, use_container_width=True)

    st.markdown("### Performance Summary")
    summary = (
        f"accuracy = **{acc:.2f}** , "
        f"precision = **{prec:.2f}** and recall = **{rec:.2f}** achieved by The **{model_name}** model"
    )
    st.info(summary)

def predict_probabilities(pipe, X):
    import numpy as np

    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        else:
            return proba.reshape(-1)
    elif hasattr(pipe, "decision_function"):
        scores = pipe.decision_function(X)
        return 1 / (1 + np.exp(-scores))
    else:
        preds = pipe.predict(X)
        return preds.astype(float)

import requests

TEST_DATA_URL = "https://github.com/kameswararaokakaraparti/MLAssignment2/tree/main/data/diabetes_health_indicators_test.csv"

df = pd.read_csv(TEST_DATA_URL)

csv_bytes = df.to_csv(index=False).encode("utf-8")

st.caption("Click to download Test data")

st.download_button(
    label="Download",
    data=csv_bytes,
    file_name="diabetes_health_indicators_test.csv",
    mime="text/csv"
)



#add a file uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    df = df.drop(columns=['Sex', 'AnyHealthcare', 'NoDocbcCost', 'MentHlth', 'Stroke'])

    model_choice = st.selectbox(
        "Choose a Model",
            list(MODEL_LIST.keys())
    )

    if st.button("Run Model"):
       
        pipe = load_pipeline(MODEL_LIST[model_choice])

        # 2 Split features/target
        X = df.drop(columns=['Diabetes_binary'])
        y = df['Diabetes_binary']

        if model_choice == "Logistic Regression":
             y_pred = pipe.predict(X)
             y_proba = predict_probabilities(pipe, X)
             render_results(model_choice, y, y_pred, y_proba)

        if model_choice == "Decision Tree Classifier":
             y_pred = pipe.predict(X)
             y_proba = predict_probabilities(pipe, X)
             render_results(model_choice, y, y_pred, y_proba)

        if model_choice == "Naive Bayes Classifier":
             y_pred = pipe.predict(X)
             y_proba = predict_probabilities(pipe, X)
             render_results(model_choice, y, y_pred, y_proba)

        if model_choice == "K-Nearest Neighbor Classifier":
             y_pred = pipe.predict(X)
             y_proba = predict_probabilities(pipe, X)
             render_results(model_choice, y, y_pred, y_proba)

        if model_choice == "Random Forest":
             y_pred = pipe.predict(X)
             y_proba = predict_probabilities(pipe, X)
             render_results(model_choice, y, y_pred, y_proba)

        if model_choice == "XGBoost":
             y_pred = pipe.predict(X)
             y_proba = predict_probabilities(pipe, X)
             render_results(model_choice, y, y_pred, y_proba)           
             


