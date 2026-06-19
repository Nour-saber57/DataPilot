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
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from automl.orchestrator import run_automl
from llm.gemini_client import ask_gemini
from llm.prompts import (
    SUGGESTED_QUESTIONS,
    SYSTEM_PROMPT,
    build_experiment_context,
    build_user_prompt,
)

OUTPUT_DIR = "outputs"

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
        st.markdown("### 📥 Downloads")

        # Report
        report_path = Path(res.report_path)
        if report_path.exists():
            st.download_button(
                "📄 Download Report",
                data=report_path.read_text(encoding="utf-8"),
                file_name="report.md",
                mime="text/markdown",
                use_container_width=True,
            )

        # Leaderboard CSV
        lb_path = Path(res.leaderboard_path)
        if lb_path.exists():
            st.download_button(
                "📊 Download Leaderboard",
                data=lb_path.read_bytes(),
                file_name="leaderboard.csv",
                mime="text/csv",
                use_container_width=True,
            )

        # Best model
        model_path = Path(res.model_path)
        if model_path.exists():
            st.download_button(
                "🤖 Download Model (.joblib)",
                data=model_path.read_bytes(),
                file_name="best_model.joblib",
                mime="application/octet-stream",
                use_container_width=True,
            )

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
            from automl.preprocessor import detect_task_type

            detected = detect_task_type(df[target_col])
            task_override = st.selectbox(
                "📋 Task Type",
                options=["classification", "regression"],
                index=0 if detected == "classification" else 1,
                help=f"Auto-detected: {detected}. Override if needed.",
            )

        st.divider()

        if st.button("🚀 Run AutoML", type="primary", use_container_width=True):
            with st.spinner("Training models… this may take a moment."):
                try:
                    result = run_automl(
                        df=df,
                        target_col=target_col,
                        task_type=task_override,
                        output_dir=OUTPUT_DIR,
                    )
                    st.session_state.result = result
                    st.session_state.chat_history = []
                    st.success(
                        f"✅ Training complete! Best model: **{result.best_model_name}**"
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"❌ AutoML failed: {exc}")

    else:
        st.info("👆 Upload a CSV file to get started.")

# ── Tab 2: Results ───────────────────────────────────────────────────────────

with tab_results:
    if st.session_state.result is None:
        st.info("Run AutoML first to see results.")
    else:
        res = st.session_state.result

        st.markdown("### 🏆 Model Leaderboard")
        st.dataframe(
            res.leaderboard.style.format(precision=4),
            use_container_width=True,
        )

        # Model comparison chart
        comparison_fig = res.plot_figures.get("model_comparison")
        if comparison_fig is not None:
            st.pyplot(comparison_fig)

        st.divider()
        st.markdown("### 🥇 Best Model")

        if res.task_type == "classification":
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("Model", res.best_model_name)
            m_col2.metric("Weighted F1", res.best_metrics.get("weighted_f1", "N/A"))
            m_col3.metric("Accuracy", res.best_metrics.get("accuracy", "N/A"))
            m_col4.metric("Precision", res.best_metrics.get("weighted_precision", "N/A"))
        else:
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("Model", res.best_model_name)
            m_col2.metric("RMSE", res.best_metrics.get("rmse", "N/A"))
            m_col3.metric("R²", res.best_metrics.get("r2", "N/A"))
            m_col4.metric("MAPE %", res.best_metrics.get("mape_pct", "N/A"))

        # Per-class breakdown
        if res.per_class_df is not None and not res.per_class_df.empty:
            with st.expander("📋 Per-Class Metrics"):
                st.dataframe(
                    res.per_class_df.style.format(precision=4),
                    use_container_width=True,
                )

        st.divider()

        # Overfitting analysis
        st.markdown("### 🔬 Model Health")

        ov = res.best_overfit
        cv = res.best_cv

        if ov:
            risk_emoji = {"low": "🟢", "moderate": "🟡", "high": "🔴"}.get(
                ov.get("risk_level", ""), "⚪"
            )
            h_col1, h_col2, h_col3, h_col4 = st.columns(4)
            h_col1.metric("Overfit Risk", f"{risk_emoji} {ov.get('risk_level', 'N/A').title()}")
            h_col2.metric(f"Train {ov.get('metric_name', 'Score')}", ov.get("train_score", "N/A"))
            h_col3.metric(f"Test {ov.get('metric_name', 'Score')}", ov.get("test_score", "N/A"))
            h_col4.metric("Gap", ov.get("gap", "N/A"))

            if ov.get("verdict"):
                st.caption(ov["verdict"])

        if cv and cv.get("cv_mean") is not None:
            cv_metric = "Weighted F1" if res.task_type == "classification" else "RMSE"
            cv_col1, cv_col2 = st.columns(2)
            cv_col1.metric(f"CV Mean ({cv_metric})", cv["cv_mean"])
            cv_col2.metric("CV Std", f"±{cv['cv_std']}")

            fold_scores = cv.get("cv_scores", [])
            if fold_scores:
                st.caption(f"Fold scores: {', '.join(f'{s:.4f}' for s in fold_scores)}")

        # Dataset summary
        st.divider()
        st.markdown("### 📋 Dataset Summary")
        ds = res.dataset_summary
        ds_col1, ds_col2, ds_col3 = st.columns(3)
        ds_col1.metric("Rows", f"{ds['rows']:,}")
        ds_col2.metric("Features", ds["features"])
        ds_col3.metric("Task", res.task_type.title())

        # Data quality warnings
        dq = res.data_quality
        if dq:
            warnings = []
            if dq.get("duplicate_rows", 0) > 0:
                warnings.append(f"⚠️ {dq['duplicate_rows']} duplicate rows found")
            balance = dq.get("class_balance")
            if balance and balance.get("is_imbalanced"):
                warnings.append(f"⚠️ Class imbalance detected (ratio: {balance['imbalance_ratio']:.1f}x)")
            if dq.get("high_cardinality_columns"):
                n = len(dq["high_cardinality_columns"])
                warnings.append(f"⚠️ {n} high-cardinality categorical column(s)")
            if warnings:
                for w in warnings:
                    st.warning(w)

# ── Tab 3: Explainability ────────────────────────────────────────────────────

with tab_explain:
    if st.session_state.result is None:
        st.info("Run AutoML first to see explainability plots.")
    else:
        res = st.session_state.result

        # EDA section
        st.markdown("### 📈 Exploratory Data Analysis")

        eda_tab1, eda_tab2, eda_tab3 = st.tabs([
            "Target Distribution", "Correlations", "Missing Values"
        ])

        with eda_tab1:
            fig = res.plot_figures.get("target_distribution")
            if fig is not None:
                st.pyplot(fig)

        with eda_tab2:
            fig = res.plot_figures.get("correlation")
            if fig is not None:
                st.pyplot(fig)

        with eda_tab3:
            fig = res.plot_figures.get("missing_values")
            if fig is not None:
                st.pyplot(fig)
            else:
                st.success("No missing values in the dataset ✅")

        st.divider()

        # Feature importance
        st.markdown("### 📊 Feature Importance")
        fi_fig = res.plot_figures.get("feature_importance")
        if fi_fig is not None:
            st.pyplot(fi_fig)

        if res.feature_importance_df is not None:
            with st.expander("View feature importance table"):
                st.dataframe(
                    res.feature_importance_df.head(15).style.format(
                        {"Importance": "{:.4f}"}
                    ),
                    use_container_width=True,
                )

        st.divider()

        # Diagnostic plots
        if res.task_type == "classification":
            diag_tab1, diag_tab2 = st.tabs(["Confusion Matrix", "ROC Curve"])

            with diag_tab1:
                fig = res.plot_figures.get("diagnostic")
                if fig is not None:
                    st.pyplot(fig)

            with diag_tab2:
                fig = res.plot_figures.get("roc")
                if fig is not None:
                    st.pyplot(fig)
                else:
                    st.info("ROC curve not available for this model type.")
        else:
            diag_tab1, diag_tab2 = st.tabs(["Predicted vs Actual", "Residual Analysis"])

            with diag_tab1:
                fig = res.plot_figures.get("diagnostic")
                if fig is not None:
                    st.pyplot(fig)

            with diag_tab2:
                fig = res.plot_figures.get("residual")
                if fig is not None:
                    st.pyplot(fig)

# ── Tab 4: Gemini Chat ──────────────────────────────────────────────────────

with tab_chat:
    if st.session_state.result is None:
        st.info("Run AutoML first to chat with Gemini about your results.")
    else:
        res = st.session_state.result

        # Build context once
        fi_str = ""
        if res.feature_importance_df is not None:
            fi_str = res.feature_importance_df.head(10).to_string(index=False)

        context = build_experiment_context(
            dataset_summary=res.dataset_summary,
            task_type=res.task_type,
            leaderboard_str=res.leaderboard.to_string(),
            best_model_name=res.best_model_name,
            metrics=res.best_metrics,
            feature_importance_str=fi_str,
        )

        # Suggested questions
        st.markdown("#### 💡 Suggested Questions")
        q_cols = st.columns(len(SUGGESTED_QUESTIONS))
        for i, question in enumerate(SUGGESTED_QUESTIONS):
            if q_cols[i].button(question, key=f"sq_{i}", use_container_width=True):
                st.session_state.chat_history.append(
                    {"role": "user", "content": question}
                )
                with st.spinner("Asking Gemini…"):
                    prompt = build_user_prompt(context, question)
                    answer = ask_gemini(SYSTEM_PROMPT, prompt)
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
                    prompt = build_user_prompt(context, user_input)
                    answer = ask_gemini(SYSTEM_PROMPT, prompt)
                st.markdown(answer)

            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer}
            )
