import streamlit as st
import pandas as pd

st.title("Upload Dataset")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.session_state["df"] = df

    st.success("Dataset uploaded successfully!")

    st.subheader("Preview")
    st.dataframe(df.head())