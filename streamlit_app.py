#Stremlit app file
import streamlit as st
import pandas as pd

#set title
st.set_page_config(
    page_title="ML Assignment2 with multi model selection",
    layout="wide"
)

#add a file uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])


#add a dropdown to select mutiple models
# 3. Dropdown menu to select model
model_choice = st.selectbox(
    "Select a model",
    ["Logistic Regression", 
    "Decision Tree Classifier", 
    "K-Nearest Neighbor Classifier",
    "Naive Bayes Classifier", 
    "Ensemble Model - Random Forest",
    "Ensemble Model - XGBoost"
    ]
)

