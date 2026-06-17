# import streamlit as st
# import pandas as pd

# st.title("Results Dashboard")
# st.markdown("Review the model performance metrics and identify the best candidate for further evaluation.")

# if "results" not in st.session_state:
#     st.warning("No results found. Train models first.")
#     st.stop()

# results = st.session_state["results"]

# df = pd.DataFrame(list(results.items()), columns=["Model", "Score"])

# st.markdown("### Performance summary")
# st.dataframe(df)

# best_model = df.loc[df["Score"].idxmax()]

# st.markdown("---")
# best_col, score_col = st.columns([2, 1])
# best_col.subheader("Best performing model")
# best_col.write(f"**Model:** {best_model['Model']}")
# best_col.write(f"**Score:** {best_model['Score']}")
# score_col.metric("Top score", best_model['Score'])