import streamlit as st
import pandas as pd

st.title("Dataset Analysis")

if "df" not in st.session_state:
    st.warning("Please upload a dataset first.")
    st.stop()

df = st.session_state["df"]

# ------------------------
# BASIC PROFILING
# ------------------------
rows, cols = df.shape
missing = df.isnull().mean().mean() * 100
duplicates = df.duplicated().sum()

# SIMPLE HEALTH SCORE
health_score = 100
health_score -= missing * 0.5
health_score -= (duplicates / len(df)) * 20

health_score = max(0, round(health_score, 2))

st.subheader("Dataset Overview")
st.write(f"Rows: {rows}")
st.write(f"Columns: {cols}")

st.subheader("Health Score")
st.metric("Dataset Health", f"{health_score}/100")

# ------------------------
# BASIC INSIGHTS
# ------------------------
st.subheader("Insights")

insights = []

if missing > 10:
    insights.append("High missing values detected.")

if duplicates > 0:
    insights.append("Duplicate rows exist.")

if cols > 20:
    insights.append("High dimensional dataset.")

if len(insights) == 0:
    insights.append("Dataset looks clean.")

for i in insights:
    st.write("•", i)

# ------------------------
# TASK RECOMMENDATION (simple heuristic)
# ------------------------
st.subheader("Recommended Task")

if "target" in df.columns:
    if df["target"].dtype == "object":
        task = "Classification"
    else:
        task = "Regression"
else:
    task = "Unknown (Please select target column in next step)"

st.success(f"Recommended Task: {task}")