import streamlit as st

st.title("Model Selection & Training")

if "df" not in st.session_state:
    st.warning("Upload dataset first.")
    st.stop()

df = st.session_state["df"]

# ------------------------
# TARGET SELECTION
# ------------------------
target = st.selectbox("Select Target Column", df.columns)
st.session_state["target"] = target

# ------------------------
# MODEL RECOMMENDATION (STATIC LOGIC)
# ------------------------
st.subheader("Recommended Models")

if df[target].dtype == "object":
    recommended = ["Logistic Regression", "Random Forest", "XGBoost"]
else:
    recommended = ["Linear Regression", "Random Forest Regressor"]

for m in recommended:
    st.write("-", m)

# ------------------------
# MODEL SELECTION
# ------------------------
selected_models = st.multiselect(
    "Select Models to Train",
    recommended,
    default=recommended[:1]
)

st.session_state["models"] = selected_models

# ------------------------
# TRAIN BUTTON (SIMULATION)
# ------------------------
if st.button("Train Models"):

    import time
    import random

    st.write("Training in progress...")

    time.sleep(2)

    results = {}
    for model in selected_models:
        results[model] = round(random.uniform(0.7, 0.95), 3)

    st.session_state["results"] = results

    st.success("Training completed!")
    st.write(results)