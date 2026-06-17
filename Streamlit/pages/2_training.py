# import streamlit as st

# st.title("Training")
# st.markdown(
#     "Configure the target variable, review recommended model families, and train selected models "
#     "with a simplified AutoML workflow."
# )

# if "df" not in st.session_state:
#     st.warning("Upload dataset first.")
#     st.stop()

# df = st.session_state["df"]

# target = st.selectbox("Select target column", df.columns)
# st.session_state["target"] = target

# if df[target].dtype == "object":
#     recommended = ["Logistic Regression", "Random Forest", "XGBoost"]
# else:
#     recommended = ["Linear Regression", "Random Forest Regressor"]

# model_col, task_col = st.columns([2, 1])
# with model_col:
#     st.subheader("Recommended models")
#     for model in recommended:
#         st.markdown(f"- {model}")

# with task_col:
#     st.subheader("Recommended task")
#     task = "Classification" if df[target].dtype == "object" else "Regression"
#     st.success(task)

# selected_models = st.multiselect(
#     "Select models to train",
#     recommended,
#     default=recommended[:1]
# )

# st.session_state["models"] = selected_models

# if st.button("Train models"):
#     import time
#     import random

#     st.info("Training in progress...")
#     time.sleep(2)

#     results = {model: round(random.uniform(0.7, 0.95), 3) for model in selected_models}
#     st.session_state["results"] = results

#     st.success("Training completed.")
#     st.write(results)