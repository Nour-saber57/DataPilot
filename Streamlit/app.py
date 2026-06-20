"""
Streamlit AutoML Agent — main application entry point.

Layout:
  Sidebar  → Instructions, dataset info, download actions
  Tab 1    → Upload & Configure
  Tab 2    → Results (leaderboard + best model card + overfit + CV)
  Tab 3    → Analysis (EDA + diagnostics + feature importance)
  Tab 4    → Gemini Chat
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

OUTPUT_DIR = "outputs"
FASTAPI_BASE_URL = "http://localhost:8000"

SUGGESTED_QUESTIONS = [
    "What are the top 3 important features?",
    "Is the model overfitting?",
    "What recommendations do you have?",
    "How balanced is the dataset?",
]

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AutoML Agent",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom theme ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .block-container {
        padding-top: 2rem;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 20px;
    }

    div[data-testid="stMetric"] label {
        color: #94a3b8;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #e2e8f0;
        font-weight: 600;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 500;
    }

    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }

    div[data-testid="stChatMessage"] {
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────

if "result" not in st.session_state:
    st.session_state.result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = None

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🧪 AutoML Agent")
    st.caption("Upload → Train → Explain → Chat")
    st.divider()

    if st.session_state.uploaded_df is not None:
        df = st.session_state.uploaded_df
        st.markdown("### 📊 Dataset Info")
        st.markdown(f"**Rows:** {len(df):,}")
        st.markdown(f"**Columns:** {len(df.columns)}")
        st.markdown(f"**Missing:** {df.isnull().sum().sum():,}")
        st.divider()

    if st.session_state.result is not None:
        res = st.session_state.result
        st.markdown("### 🏆 Best Model Results")
        st.markdown(f"**Best Model:** {res['best_model_name']}")
        score_val = res['best_metrics'].get('f1', res['best_metrics'].get('r2', None))
        if score_val is not None:
            st.markdown(f"**Score:** {score_val:.4f}")
        st.divider()

    st.caption("Built with ❤️ by Sara Musalim | Streamlit + scikit-learn + Gemini")

# ── Main content ──────────────────────────────────────────────────────────────

st.title("🧪 AutoML Agent")
st.markdown("Upload a CSV dataset, train models, and explore results with AI.")

tab_upload, tab_results, tab_analysis, tab_chat = st.tabs([
    "📤 Upload & Configure",
    "📊 Results",
    "📈 Analysis",
    "💬 Gemini Chat",
])

# ── Tab 1: Upload & Configure ─────────────────────────────────────────────────

with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload your CSV dataset",
        type=["csv"],
        help="Maximum recommended size: 100k rows for fast training.",
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.uploaded_df = df

        st.markdown("### Dataset Preview")
        st.dataframe(df.head(20), use_container_width=True)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", f"{len(df):,}")
        col2.metric("Columns", len(df.columns))
        col3.metric("Missing Values", f"{df.isnull().sum().sum():,}")
        col4.metric("Numeric Features", len(df.select_dtypes(include="number").columns))

        st.divider()

        cfg_col1, cfg_col2 = st.columns(2)

        with cfg_col1:
            target_col = st.selectbox(
                "🎯 Select Target Column",
                options=df.columns.tolist(),
                index=len(df.columns) - 1,
                help="The column the model will learn to predict.",
            )

        with cfg_col2:
            task_override = st.selectbox(
                "📋 Task Type",
                options=["classification", "regression"],
                index=0,
                help="Auto-detected by backend. Override if needed.",
            )

        st.divider()

        st.markdown("### 🤖 Model Selection")

        train_mode = st.radio(
            "Choose training mode:",
            options=["Train all models", "Train a specific model"],
            horizontal=True,
            help="Select whether to train all available models or focus on a specific model",
        )

        selected_model = None
        if train_mode == "Train a specific model":
            with st.spinner("Fetching available models..."):
                try:
                    models_response = requests.get(
                        f"{FASTAPI_BASE_URL}/available-models",
                        params={"task": task_override},
                    )
                    if models_response.status_code == 200:
                        models_data = models_response.json()
                        available_models = models_data.get("models", [])
                        selected_model = st.selectbox(
                            "Select a model to train:",
                            options=available_models,
                            help="Choose which model you want to train",
                        )
                    else:
                        st.error("Failed to fetch available models")
                except Exception as e:
                    st.error(f"Error fetching models: {e}")

        st.divider()

        if st.button("🚀 Run AutoML", type="primary", use_container_width=True):
            if train_mode == "Train a specific model" and selected_model is None:
                st.error("❌ Please select a model to train")
            else:
                with st.spinner("Training models… this may take a moment."):
                    try:
                        import os
                        import tempfile

                        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                            df.to_csv(tmp.name, index=False)
                            tmp_path = tmp.name

                        with open(tmp_path, "rb") as f:
                            files = {"file": f}
                            params = {"target_column": target_col}

                            upload_response = requests.post(
                                f"{FASTAPI_BASE_URL}/upload-data",
                                files=files,
                                params=params,
                            )
                            if upload_response.status_code != 200:
                                st.error(f"❌ Upload failed: {upload_response.json()}")
                                raise Exception("Data upload failed")

                            upload_data = upload_response.json()
                            dataset_id = upload_data.get("dataset_id")

                            if train_mode == "Train all models":
                                train_response = requests.post(
                                    f"{FASTAPI_BASE_URL}/train-all-models",
                                    params={
                                        "dataset_id": dataset_id,
                                        "target_column": target_col,
                                        "test_size": 0.2,
                                        "async_training": False,
                                    },
                                )
                            else:
                                train_response = requests.post(
                                    f"{FASTAPI_BASE_URL}/train-model",
                                    params={
                                        "model_name": selected_model,
                                        "target_column": target_col,
                                        "test_size": 0.2,
                                        "dataset_id": dataset_id,
                                        "async_training": False,
                                    },
                                )

                            if train_response.status_code != 200:
                                st.error(f"❌ Training failed: {train_response.json()}")
                                raise Exception("Model training failed")

                            train_data = train_response.json()

                            if train_mode == "Train all models":
                                leaderboard = train_data.get("leaderboard", [])
                                detailed_results = train_data.get("detailed_results", {})
                                best_model_name = max(leaderboard, key=lambda x: x["score"])["model"]
                                best_result = detailed_results[best_model_name]
                            else:
                                best_model_name = train_data.get("model_name", selected_model)
                                leaderboard = [{"model": best_model_name, "score": train_data.get("score", 0)}]
                                detailed_results = {best_model_name: train_data}
                                best_result = train_data

                            task_type = train_data.get("task")
                            best_metrics = best_result.get("metrics", {})

                            st.session_state.result = {
                                "task_type": task_type,
                                "best_model_name": best_model_name,
                                "best_metrics": best_metrics,
                                "results": detailed_results,
                                "leaderboard": leaderboard,
                                "dataset_id": dataset_id,
                                "target_col": target_col,
                                "y_test": best_result.get("y_true", []),
                                "y_pred": best_result.get("y_pred", []),
                                "y_pred_proba": best_result.get("y_pred_proba", []),
                                "feature_importance": best_result.get("feature_importance", {}),
                                "train_score": best_result.get("train_score", None),
                                "cv_scores": best_result.get("cv_scores", []),
                            }
                            st.session_state.chat_history = []

                            st.success(f"✅ Training complete! Best model: **{best_model_name}**")
                            st.rerun()

                    except Exception as exc:
                        st.error(f"❌ AutoML failed: {exc}")
                    finally:
                        if "tmp_path" in locals():
                            try:
                                os.unlink(tmp_path)
                            except Exception:
                                pass

    else:
        st.info("👆 Upload a CSV file to get started.")

# ── Tab 2: Results ────────────────────────────────────────────────────────────

with tab_results:
    if st.session_state.result is None:
        st.info("Run AutoML first to see results.")
    else:
        res = st.session_state.result

        # ── Leaderboard ──
        st.markdown("### 🏆 Model Leaderboard")
        leaderboard_df = pd.DataFrame(res["leaderboard"])
        st.dataframe(leaderboard_df.style.format(precision=4), use_container_width=True)

        # ── Model comparison chart (all models) ──
        if len(res["leaderboard"]) > 1:
            st.divider()
            st.markdown("### 📊 Model Comparison")
            try:
                from Backend.visualization import plot_model_comparison
                fig = plot_model_comparison(res["leaderboard"])
                col_chart, _ = st.columns([2, 1])
                with col_chart:
                    st.pyplot(fig)
            except Exception as e:
                st.warning(f"Could not render model comparison chart: {e}")

        st.divider()

        # ── Best model metrics ──
        st.markdown("### 🥇 Best Model")
        best_name = res["best_model_name"]
        best_metrics = res["best_metrics"]

        if res["task_type"] == "classification":
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("Model", best_name)
            m_col2.metric("F1 Score", f"{best_metrics.get('f1', 0):.4f}")
            m_col3.metric("Accuracy", f"{best_metrics.get('accuracy', 0):.4f}")
            m_col4.metric("Precision", f"{best_metrics.get('precision', 0):.4f}")
        else:
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("Model", best_name)
            m_col2.metric("R²", f"{best_metrics.get('r2', 0):.4f}")
            m_col3.metric("RMSE", f"{best_metrics.get('rmse', 0):.4f}")
            m_col4.metric("MAE", f"{best_metrics.get('mae', 0):.4f}")

        st.divider()

        # ── Overfitting indicator ──
        train_score = res.get("train_score")
        test_score = best_metrics.get("f1") if res["task_type"] == "classification" else best_metrics.get("r2")

        if train_score is not None and test_score is not None:
            st.markdown("### ⚖️ Overfitting Analysis")
            gap = abs(float(train_score) - float(test_score))
            if gap < 0.05:
                risk_label, risk_color = "Low risk ✅", "normal"
            elif gap < 0.15:
                risk_label, risk_color = "Moderate risk ⚠️", "off"
            else:
                risk_label, risk_color = "High risk ❌", "inverse"

            ov_col1, ov_col2, ov_col3 = st.columns(3)
            ov_col1.metric("Train Score", f"{float(train_score):.4f}")
            ov_col2.metric("Test Score", f"{float(test_score):.4f}")
            ov_col3.metric("Gap", f"{gap:.4f}", delta=risk_label, delta_color=risk_color)
            st.divider()

        # ── CV scores ──
        cv_scores = res.get("cv_scores", [])
        if cv_scores:
            st.markdown("### 🔁 Cross-Validation Scores")
            cv_arr = np.array(cv_scores)
            cv_col1, cv_col2, cv_col3 = st.columns(3)
            cv_col1.metric("CV Mean", f"{cv_arr.mean():.4f}")
            cv_col2.metric("CV Std", f"{cv_arr.std():.4f}")
            cv_col3.metric("CV Folds", len(cv_arr))

            fold_df = pd.DataFrame({"Fold": [f"Fold {i+1}" for i in range(len(cv_arr))], "Score": cv_arr})
            st.dataframe(fold_df.style.format({"Score": "{:.4f}"}), use_container_width=True, hide_index=True)
            st.divider()

        # ── All model metrics table ──
        st.markdown("### 📋 All Model Metrics")
        metrics_display = pd.DataFrame([
            {"Model": name, **res["results"][name]["metrics"]}
            for name in res["results"].keys()
        ])
        st.dataframe(metrics_display, use_container_width=True)

# ── Tab 3: Analysis ───────────────────────────────────────────────────────────

with tab_analysis:
    if st.session_state.result is None:
        st.info("Run AutoML first to see data analysis and diagnostics.")
    else:
        res = st.session_state.result

        from Backend.visualization import (
            plot_target_distribution,
            plot_correlation_matrix,
            plot_missing_values,
            plot_feature_importance,
            plot_confusion_matrix,
            plot_regression_diagnostics,
            plot_roc_curve,
        )
        from Backend.analytics import analyze_per_class_metrics

        # Use the actual target column chosen during training
        target_col = res.get("target_col", st.session_state.uploaded_df.columns[-1])

        # ── EDA Section ──────────────────────────────────────────────────────
        st.markdown("### 📈 Exploratory Data Analysis")

        eda_c1, eda_c2 = st.columns(2)

        with eda_c1:
            try:
                st.markdown("**Target Distribution**")
                fig = plot_target_distribution(
                    st.session_state.uploaded_df,
                    target_col,
                    res["task_type"],
                )
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error plotting target distribution: {e}")

        with eda_c2:
            try:
                missing = st.session_state.uploaded_df.isnull().sum().sum()
                if missing > 0:
                    st.markdown("**Missing Values**")
                    fig = plot_missing_values(st.session_state.uploaded_df, target_col)
                    if fig is not None:
                        st.pyplot(fig)
                else:
                    st.success("✅ No missing values in the dataset!")
            except Exception as e:
                st.error(f"Error plotting missing values: {e}")

        try:
            st.markdown("**Feature Correlation Matrix**")
            fig = plot_correlation_matrix(st.session_state.uploaded_df, target_col)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error plotting correlation matrix: {e}")

        st.divider()

        # ── Model Diagnostics ─────────────────────────────────────────────────
        st.markdown("### 🔬 Model Diagnostics")

        y_test = np.array(res.get("y_test", []))
        y_pred = np.array(res.get("y_pred", []))
        y_pred_proba = res.get("y_pred_proba", [])

        if len(y_test) > 0 and len(y_pred) > 0:
            if res["task_type"] == "classification":
                diag_c1, diag_c2 = st.columns(2)

                with diag_c1:
                    try:
                        st.markdown("**Confusion Matrix**")
                        fig = plot_confusion_matrix(y_test, y_pred)
                        st.pyplot(fig)
                    except Exception as e:
                        st.warning(f"Could not generate confusion matrix: {e}")

                with diag_c2:
                    if y_pred_proba:
                        try:
                            st.markdown("**ROC Curve**")
                            fig = plot_roc_curve(y_test, np.array(y_pred_proba))
                            if fig is not None:
                                st.pyplot(fig)
                        except Exception as e:
                            st.warning(f"Could not generate ROC curve: {e}")

                # Per-class metrics table
                try:
                    st.markdown("**Per-Class Metrics**")
                    per_class_df = analyze_per_class_metrics(y_test, y_pred)
                    if per_class_df is not None:
                        st.dataframe(per_class_df, use_container_width=True, hide_index=True)
                except Exception as e:
                    st.warning(f"Could not generate per-class metrics: {e}")

            else:
                try:
                    st.markdown("**Regression Diagnostics**")
                    fig = plot_regression_diagnostics(y_test, y_pred)
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"Could not generate regression diagnostics: {e}")

                # Residual distribution
                try:
                    import matplotlib.pyplot as plt
                    residuals = y_test - y_pred
                    st.markdown("**Residual Distribution**")
                    fig, ax = plt.subplots(figsize=(6, 3.5))
                    ax.hist(residuals, bins=30, color="steelblue", edgecolor="black", alpha=0.7)
                    ax.axvline(0, color="red", linestyle="--", lw=1.5)
                    ax.set_xlabel("Residual (actual − predicted)")
                    ax.set_ylabel("Count")
                    ax.set_title("Residual Distribution")
                    plt.tight_layout()
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"Could not generate residual distribution: {e}")
        else:
            st.info("💡 Prediction data not available. Train models to see diagnostics.")

        st.divider()

        # ── Feature Importance ────────────────────────────────────────────────
        st.markdown("### ⭐ Feature Importance")

        feature_importance_raw = res.get("feature_importance", {})
        if feature_importance_raw:
            importance_df = pd.DataFrame([
                {"Feature": name, "Importance": score}
                for name, score in feature_importance_raw.items()
            ]).sort_values("Importance", ascending=False).reset_index(drop=True)

            imp_c1, imp_c2 = st.columns([1, 1])

            with imp_c1:
                st.markdown("**Top Features**")
                st.dataframe(importance_df.head(10), use_container_width=True, hide_index=True)

            with imp_c2:
                try:
                    st.markdown("**Importance Chart**")
                    fig = plot_feature_importance(importance_df, top_n=15)
                    st.pyplot(fig)
                except Exception as e:
                    st.info(f"Feature importance plot unavailable: {e}")
        else:
            st.info("⏳ Feature importance not returned by the backend. Ensure your FastAPI /train endpoint includes a 'feature_importance' dict in its response.")

# ── Tab 4: Gemini Chat ────────────────────────────────────────────────────────

with tab_chat:
    if st.session_state.result is None:
        st.info("Run AutoML first to chat with Gemini about your results.")
    else:
        res = st.session_state.result

        st.markdown("#### 💡 Suggested Questions")
        q_cols = st.columns(len(SUGGESTED_QUESTIONS))
        for i, question in enumerate(SUGGESTED_QUESTIONS):
            if q_cols[i].button(question, key=f"sq_{i}", use_container_width=True):
                st.session_state.chat_history.append({"role": "user", "content": question})
                with st.spinner("Asking Gemini…"):
                    try:
                        response = requests.post(
                            f"{FASTAPI_BASE_URL}/chat",
                            params={"message": question},
                        )
                        answer = (
                            response.json()["response"]
                            if response.status_code == 200
                            else f"Error: {response.json().get('detail', 'Unknown error')}"
                        )
                    except Exception as exc:
                        answer = f"Connection error: {exc}"
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.rerun()

        st.divider()

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_input = st.chat_input("Ask about your AutoML results…")
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)
            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    try:
                        response = requests.post(
                            f"{FASTAPI_BASE_URL}/chat",
                            params={"message": user_input},
                        )
                        answer = (
                            response.json()["response"]
                            if response.status_code == 200
                            else f"Error: {response.json().get('detail', 'Unknown error')}"
                        )
                    except Exception as exc:
                        answer = f"Connection error: {exc}"
                st.markdown(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
