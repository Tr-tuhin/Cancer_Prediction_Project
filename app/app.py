import streamlit as st
import pandas as pd
from src.preprocess import preprocess_input
from src.predict import load_model, make_prediction

st.title("Cancer Prediction App")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:")
    st.write(df.head())

    model = load_model("models/model.pkl")
    X = preprocess_input(df)
    predictions = make_prediction(model, X)

    st.write("Prediction Result:")
    st.write(predictions)
