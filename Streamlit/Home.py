# import streamlit as st
# import pandas as pd

# st.set_page_config(page_title="Home", layout="wide")

# st.header("Data Pilot")
# st.subheader("An Explainable Intelligent AutoML Recommendation System")
# st.markdown(
#     "Upload your dataset to receive explainable data insights, model recommendations, "
#     "and an automated training workflow for production-ready ML evaluation."
# )

# st.markdown("---")

# st.subheader("Upload Dataset")
# uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# st.subheader("Dataset Status")
# if uploaded_file is not None:
#     df = pd.read_csv(uploaded_file)
#     st.session_state["df"] = df
#     st.success("Dataset uploaded successfully.")
# elif "df" in st.session_state:
#     st.info("A dataset is already loaded. Use the Insights, Training, or Results tabs to continue.")
# else:
#     st.caption("Please upload a CSV file to begin the AutoML workflow.")

# st.subheader("Dataset Preview")
# if "df" in st.session_state:
#     df = st.session_state["df"]
#     st.dataframe(df.head())
# else:
#     st.write("No dataset uploaded yet.")

# st.markdown("---")
# st.subheader("How it works")
# st.markdown(
#     "1. Upload your data.\n"
#     "2. Explore insights in the Insights tab.\n"
#     "3. Configure training in the Training tab.\n"
#     "4. Review outcomes in the Results tab."
# )
