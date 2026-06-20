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

tab_upload, tab_results, tab_explain, tab_chat = st.tabs([
    "📤 Upload & Configure",
    "📊 Results",
    "🔍 Explainability",
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

        if st.button("🚀 Run AutoML", type="primary", use_container_width=True):
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
                        
                        # Step 2: Train all models
                        train_response = requests.post(
                            f"{FASTAPI_BASE_URL}/train-all-models",
                            params={
                                'dataset_id': dataset_id,
                                'target_column': target_col,
                                'test_size': 0.2,
                                'async_training': False
                            }
                        )
                        
                        if train_response.status_code != 200:
                            st.error(f"❌ Training failed: {train_response.json()}")
                            raise Exception("Model training failed")
                        
                        train_data = train_response.json()
                        
                        # Step 3: Parse and store results
                        leaderboard = train_data.get('leaderboard', [])
                        detailed_results = train_data.get('detailed_results', {})
                        task_type = train_data.get('task')
                        
                        best_model_name = max(leaderboard, key=lambda x: x['score'])['model']
                        best_metrics = detailed_results[best_model_name]['metrics']
                        
                        st.session_state.result = {
                            "task_type": task_type,
                            "best_model_name": best_model_name,
                            "best_metrics": best_metrics,
                            "results": detailed_results,
                            "leaderboard": leaderboard,
                            "dataset_id": dataset_id,
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

# ── Tab 3: Explainability ────────────────────────────────────────────────────

with tab_explain:
    if st.session_state.result is None:
        st.info("Run AutoML first to see explainability analysis.")
    else:
        res = st.session_state.result
        
        st.info("📊 **Explainability features coming soon!**\n\nWe'll add feature importance, visualizations, and diagnostic plots here.")

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
