"""
Streamlit AutoML Agent — main application entry point.

Layout:
  Sidebar  → Instructions, dataset info, download actions
  Tab 1    → Upload & Configure
  Tab 2    → Results (leaderboard + best model card + overfit + CV)
  Tab 3    → Explainability (EDA + feature importance + diagnostics)
  Tab 4    → Gemini Chat
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

import requests
import numpy as np

OUTPUT_DIR = "outputs"
FASTAPI_BASE_URL = "http://localhost:8000"

SUGGESTED_QUESTIONS = [
    "What are the top 3 important features?",
    "Is the model overfitting?",
    "What recommendations do you have?",
    "How balanced is the dataset?",
]

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AutoML Agent",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom theme ─────────────────────────────────────────────────────────────

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

# ── Session state init ───────────────────────────────────────────────────────

if "result" not in st.session_state:
    st.session_state.result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = None
if "show_target_dist" not in st.session_state:
    st.session_state.show_target_dist = False
if "show_corr" not in st.session_state:
    st.session_state.show_corr = False
if "show_missing" not in st.session_state:
    st.session_state.show_missing = False

# ── Sidebar ──────────────────────────────────────────────────────────────────

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
        st.markdown("### � Best Model Results")
        st.markdown(f"**Best Model:** {res['best_model_name']}")
        st.markdown(f"**Score:** {res['best_metrics'].get('f1', res['best_metrics'].get('r2', 'N/A')):.4f}")
        st.divider()
    st.caption("Built with ❤️ by Sara Musalim | Streamlit + scikit-learn + Gemini")

# ── Main content ─────────────────────────────────────────────────────────────

st.title("🧪 AutoML Agent")
st.markdown("Upload a CSV dataset, train models, and explore results with AI.")

tab_upload, tab_results, tab_analysis, tab_chat = st.tabs([
    "📤 Upload & Configure",
    "📊 Results",
    "📈 Analysis",
    "💬 Gemini Chat",
])

# ── Tab 1: Upload & Configure ───────────────────────────────────────────────

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
            # Task will be auto-detected by FastAPI based on target column
            task_override = st.selectbox(
                "📋 Task Type",
                options=["classification", "regression"],
                index=0,
                help="Auto-detected by backend. Override if needed.",
            )

        st.divider()

        # Model Selection Section
        st.markdown("### 🤖 Model Selection")
        
        train_mode = st.radio(
            "Choose training mode:",
            options=["Train all models", "Train a specific model"],
            horizontal=True,
            help="Select whether to train all available models or focus on a specific model"
        )
        
        selected_model = None
        if train_mode == "Train a specific model":
            with st.spinner("Fetching available models..."):
                try:
                    models_response = requests.get(
                        f"{FASTAPI_BASE_URL}/available-models",
                        params={"task": task_override}
                    )
                    
                    if models_response.status_code == 200:
                        models_data = models_response.json()
                        available_models = models_data.get('models', [])
                        
                        selected_model = st.selectbox(
                            "Select a model to train:",
                            options=available_models,
                            help="Choose which model you want to train"
                        )
                    else:
                        st.error("Failed to fetch available models")
                except Exception as e:
                    st.error(f"Error fetching models: {e}")
        
        st.divider()

        if st.button("🚀 Run AutoML", type="primary", use_container_width=True):
            # Validate selection
            if train_mode == "Train a specific model" and selected_model is None:
                st.error("❌ Please select a model to train")
            else:
                with st.spinner("Training models… this may take a moment."):
                    try:
                        # Save uploaded file temporarily for FastAPI
                        import tempfile
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                            df.to_csv(tmp.name, index=False)
                            tmp_path = tmp.name
                        
                        # Send to FastAPI for training
                        with open(tmp_path, 'rb') as f:
                            files = {'file': f}
                            params = {'target_column': target_col}
                            
                            # Step 1: Upload data
                            upload_response = requests.post(
                                f"{FASTAPI_BASE_URL}/upload-data",
                                files=files,
                                params=params
                            )
                            
                            if upload_response.status_code != 200:
                                st.error(f"❌ Upload failed: {upload_response.json()}")
                                raise Exception("Data upload failed")
                            
                            upload_data = upload_response.json()
                            dataset_id = upload_data.get('dataset_id')
                            
                            # Step 2: Train models (all or specific)
                            if train_mode == "Train all models":
                                train_response = requests.post(
                                    f"{FASTAPI_BASE_URL}/train-all-models",
                                    params={
                                        'dataset_id': dataset_id,
                                        'target_column': target_col,
                                        'test_size': 0.2,
                                        'async_training': False
                                    }
                                )
                            else:  # Train a specific model
                                train_response = requests.post(
                                    f"{FASTAPI_BASE_URL}/train-model",
                                    params={
                                        'model_name': selected_model,
                                        'target_column': target_col,
                                        'test_size': 0.2,
                                        'dataset_id': dataset_id,
                                        'async_training': False
                                    }
                                )
                            
                            if train_response.status_code != 200:
                                st.error(f"❌ Training failed: {train_response.json()}")
                                raise Exception("Model training failed")
                            
                            train_data = train_response.json()
                            
                            # Step 3: Parse and store results
                            if train_mode == "Train all models":
                                leaderboard = train_data.get('leaderboard', [])
                                detailed_results = train_data.get('detailed_results', {})
                                best_model_name = max(leaderboard, key=lambda x: x['score'])['model']
                                best_result = detailed_results[best_model_name]
                            else:
                                # For single model, format as leaderboard with one entry
                                best_model_name = train_data.get('model_name', selected_model)
                                leaderboard = [{"model": best_model_name, "score": train_data.get('score', 0)}]
                                detailed_results = {best_model_name: train_data}
                                best_result = train_data
                            
                            task_type = train_data.get('task')
                            best_metrics = best_result.get('metrics', {})
                            
                            st.session_state.result = {
                                "task_type": task_type,
                                "best_model_name": best_model_name,
                                "best_metrics": best_metrics,
                                "results": detailed_results,
                                "leaderboard": leaderboard,
                                "dataset_id": dataset_id,
                                "y_test": best_result.get('y_true', []),  # Store test labels
                                "y_pred": best_result.get('y_pred', []),  # Store predictions
                            }
                            st.session_state.chat_history = []
                            
                            st.success(f"✅ Training complete! Best model: **{best_model_name}**")
                            st.rerun()
                    
                    except Exception as exc:
                        st.error(f"❌ AutoML failed: {exc}")
                    finally:
                        # Clean up temp file
                        import os
                        if 'tmp_path' in locals():
                            try:
                                os.unlink(tmp_path)
                            except:
                                pass

    else:
        st.info("👆 Upload a CSV file to get started.")

# ── Tab 2: Results ───────────────────────────────────────────────────────────

with tab_results:
    if st.session_state.result is None:
        st.info("Run AutoML first to see results.")
    else:
        res = st.session_state.result

        st.markdown("### 🏆 Model Leaderboard")
        leaderboard_df = pd.DataFrame(res["leaderboard"])
        st.dataframe(leaderboard_df.style.format(precision=4), use_container_width=True)

        st.divider()
        st.markdown("### 🥇 Best Model")

        best_name = res["best_model_name"]
        best_metrics = res["best_metrics"]

        if res["task_type"] == "classification":
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("Model", best_name)
            m_col2.metric("F1 Score", f"{best_metrics.get('f1', 'N/A'):.4f}")
            m_col3.metric("Accuracy", f"{best_metrics.get('accuracy', 'N/A'):.4f}")
            m_col4.metric("Precision", f"{best_metrics.get('precision', 'N/A'):.4f}")
        else:
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("Model", best_name)
            m_col2.metric("R²", f"{best_metrics.get('r2', 'N/A'):.4f}")
            m_col3.metric("RMSE", f"{best_metrics.get('rmse', 'N/A'):.4f}")
            m_col4.metric("MAE", f"{best_metrics.get('mae', 'N/A'):.4f}")

        st.divider()

        st.markdown("### 📋 All Model Metrics")
        metrics_display = pd.DataFrame([
            {"Model": name, **res["results"][name]["metrics"]}
            for name in res["results"].keys()
        ])
        st.dataframe(metrics_display, use_container_width=True)

        st.divider()

        # Placeholder for future features
        st.info("💡 **Coming Soon:** Visualizations, feature importance, and overfitting analysis will be added here.")

# ── Tab 3: Analysis ──────────────────────────────────────────────────────

with tab_analysis:
    if st.session_state.result is None:
        st.info("Run AutoML first to see data analysis and diagnostics.")
    else:
        res = st.session_state.result
        
        # Import visualization functions
        from Backend.visualization import (
            plot_target_distribution,
            plot_correlation_matrix,
            plot_missing_values,
            plot_feature_importance,
            plot_confusion_matrix,
            plot_regression_diagnostics,
        )
        from Backend.analytics import extract_feature_importance
        from Backend.preprocessor import identify_feature_types
        
        # Get numeric and categorical columns
        numeric_cols, categorical_cols = identify_feature_types(
            st.session_state.uploaded_df, 
            st.session_state.uploaded_df.columns[-1]  # Target is last column
        )
        
        # ── EDA Section ──
        st.markdown("### 📈 Exploratory Data Analysis")
        
        eda_col1, eda_col2, eda_col3 = st.columns(3)
        
        with eda_col1:
            if st.button("📊 Target Distribution", key="btn_target_dist"):
                st.session_state.show_target_dist = True
        
        with eda_col2:
            if st.button("🔗 Correlations", key="btn_corr"):
                st.session_state.show_corr = True
        
        with eda_col3:
            if st.button("⚠️ Missing Values", key="btn_missing"):
                st.session_state.show_missing = True
        
        # Display EDA plots
        if st.session_state.get('show_target_dist', False):
            try:
                fig = plot_target_distribution(
                    st.session_state.uploaded_df,
                    st.session_state.uploaded_df.columns[-1],
                    res['task_type']
                )
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error plotting target distribution: {e}")
        
        if st.session_state.get('show_corr', False):
            try:
                fig = plot_correlation_matrix(st.session_state.uploaded_df, st.session_state.uploaded_df.columns[-1])
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error plotting correlation matrix: {e}")
        
        if st.session_state.get('show_missing', False):
            try:
                fig = plot_missing_values(st.session_state.uploaded_df, st.session_state.uploaded_df.columns[-1])
                if fig is not None:
                    st.pyplot(fig)
                else:
                    st.success("✅ No missing values in the dataset!")
            except Exception as e:
                st.error(f"Error plotting missing values: {e}")
        
        st.divider()
        
        # ── Model Diagnostics Section ──
        st.markdown("### 🔬 Model Diagnostics")
        
        # Check if we have predictions
        if 'y_test' in res and 'y_pred' in res:
            y_test = res.get('y_test', [])
            y_pred = res.get('y_pred', [])
            
            if res['task_type'] == "classification":
                try:
                    fig = plot_confusion_matrix(y_test, y_pred)
                    st.subheader("Confusion Matrix")
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"Could not generate confusion matrix: {e}")
            else:
                try:
                    fig = plot_regression_diagnostics(y_test, y_pred)
                    st.subheader("Regression Diagnostics")
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"Could not generate regression diagnostics: {e}")
        else:
            st.info("💡 Prediction data not yet available. Train models to see diagnostics.")
        
        st.divider()
        
        # ── Feature Importance Section ──
        st.markdown("### ⭐ Feature Importance")
        
        if 'y_test' in res and 'y_pred' in res and st.session_state.uploaded_df is not None:
            y_test = res.get('y_test', [])
            y_pred = res.get('y_pred', [])
            feature_names = [col for col in st.session_state.uploaded_df.columns if col != st.session_state.uploaded_df.columns[-1]]
            
            try:
                # Extract and display feature importance
                importance_df = extract_feature_importance(
                    None,  # Model not needed for permutation importance fallback
                    st.session_state.uploaded_df[feature_names].values,
                    y_test,
                    feature_names,
                    res['task_type']
                )
                
                col_imp_table, col_imp_chart = st.columns([1, 1])
                
                with col_imp_table:
                    st.subheader("Top Features by Importance")
                    st.dataframe(importance_df.head(10), use_container_width=True, hide_index=True)
                
                with col_imp_chart:
                    st.subheader("Feature Importance Distribution")
                    try:
                        fig = plot_feature_importance(importance_df, feature_names)
                        st.pyplot(fig)
                    except Exception as e:
                        st.info("Feature importance plot unavailable")
                        
            except Exception as e:
                st.info("⏳ Feature importance analysis available after model training")
        else:
            st.info("⏳ Train models first to see feature importance analysis")

# ── Tab 4: Gemini Chat ──────────────────────────────────────────────────────

with tab_chat:
    if st.session_state.result is None:
        st.info("Run AutoML first to chat with Gemini about your results.")
    else:
        res = st.session_state.result

        # Suggested questions
        st.markdown("#### 💡 Suggested Questions")
        q_cols = st.columns(len(SUGGESTED_QUESTIONS))
        for i, question in enumerate(SUGGESTED_QUESTIONS):
            if q_cols[i].button(question, key=f"sq_{i}", use_container_width=True):
                st.session_state.chat_history.append(
                    {"role": "user", "content": question}
                )
                with st.spinner("Asking Gemini…"):
                    try:
                        response = requests.post(
                            f"{FASTAPI_BASE_URL}/chat",
                            params={"message": question}
                        )
                        if response.status_code == 200:
                            answer = response.json()["response"]
                        else:
                            answer = f"Error: {response.json().get('detail', 'Unknown error')}"
                    except Exception as exc:
                        answer = f"Connection error: {exc}"
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": answer}
                )
                st.rerun()

        st.divider()

        # Chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        user_input = st.chat_input("Ask about your AutoML results…")
        if user_input:
            st.session_state.chat_history.append(
                {"role": "user", "content": user_input}
            )
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    try:
                        response = requests.post(
                            f"{FASTAPI_BASE_URL}/chat",
                            params={"message": user_input}
                        )
                        if response.status_code == 200:
                            answer = response.json()["response"]
                        else:
                            answer = f"Error: {response.json().get('detail', 'Unknown error')}"
                    except Exception as exc:
                        answer = f"Connection error: {exc}"
                st.markdown(answer)

            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer}
            )
