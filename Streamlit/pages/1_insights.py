import streamlit as st
import pandas as pd

st.title("Insights")

if "df" not in st.session_state:
    st.warning("Please upload a dataset first.")
    st.stop()

df = st.session_state["df"]

rows, cols = df.shape
missing = df.isnull().mean().mean() * 100
duplicates = df.duplicated().sum()

health_score = 100
health_score -= missing * 0.5
health_score -= (duplicates / len(df)) * 20
health_score = max(0, round(health_score, 2))

st.markdown("### Dataset summary")
summary_col1, summary_col2, summary_col3 = st.columns(3)
summary_col1.metric("Rows", rows)
summary_col2.metric("Columns", cols)
summary_col3.metric("Health Score", f"{health_score}/100")

st.markdown("---")
st.markdown("### Data quality review")
quality_col1, quality_col2 = st.columns(2)
quality_col1.write(f"**Missing values**: {missing:.2f}%")
quality_col2.write(f"**Duplicate rows**: {duplicates}")

st.markdown("---")
st.markdown("### Key insights")

insights = []
if missing > 10:
    insights.append("High proportion of missing values detected.")
if duplicates > 0:
    insights.append("Duplicate records were found.")
if cols > 20:
    insights.append("The dataset has high dimensionality.")
if len(insights) == 0:
    insights.append("The dataset appears clean and consistent.")

for insight in insights:
    st.markdown(f"- {insight}")
