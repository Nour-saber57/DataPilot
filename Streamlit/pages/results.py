import streamlit as st
import pandas as pd

st.title("Results Dashboard")

if "results" not in st.session_state:
    st.warning("No results found. Train models first.")
    st.stop()

results = st.session_state["results"]

# ------------------------
# DISPLAY TABLE
# ------------------------
df = pd.DataFrame(list(results.items()), columns=["Model", "Score"])
st.dataframe(df)

# ------------------------
# BEST MODEL
# ------------------------
best_model = df.loc[df["Score"].idxmax()]

st.subheader("Best Model")
st.success(f"{best_model['Model']} with score {best_model['Score']}")