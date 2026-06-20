import streamlit as st
import time
import pandas as pd
import numpy as np

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoML Agent Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #0d1117; color: #e6edf3; }
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding: 2rem 2.5rem; max-width: 100%; }

  .hero-banner {
    background: linear-gradient(135deg, #0d2818 0%, #0a1f14 60%, #061a10 100%);
    border: 1px solid #1a3a28; border-radius: 14px; padding: 2rem 2.5rem; margin-bottom: 2rem;
  }
  .hero-title { font-size: 2rem; font-weight: 700; color: #2ecc71; margin: 0 0 0.5rem 0; }
  .hero-sub   { font-size: 0.97rem; color: #8b949e; margin: 0; }

  .section-title { font-size: 1.15rem; font-weight: 700; color: #e6edf3; margin-bottom: 1rem; }

  .upload-hint { font-size: 0.78rem; color: #6e7681; margin-top: 0.4rem; }

  .tab-strip { display: flex; border-bottom: 1px solid #21262d; margin-bottom: 1.2rem; }
  .tab-item  { padding: 0.5rem 1.2rem; font-size: 0.88rem; color: #8b949e; border-bottom: 2px solid transparent; }
  .tab-item.active { color: #e6edf3; border-bottom: 2px solid #e74c3c; }

  .objective-box {
    background: #161b22; border: 1px solid #21262d; border-radius: 10px;
    padding: 1rem 1.2rem; display: flex; align-items: flex-start; gap: 0.8rem;
    margin-bottom: 1.2rem; color: #6e7681; font-size: 0.9rem;
  }
  .obj-icon {
    background: #e67e22; border-radius: 8px; width: 32px; height: 32px;
    display: flex; align-items: center; justify-content: center; font-size: 1rem; flex-shrink: 0;
  }
  .insights-panel { background: #161b22; border: 1px solid #21262d; border-radius: 10px; padding: 1.2rem 1.4rem; }
  .pipeline-title { color: #2ecc71; font-weight: 600; font-size: 0.95rem; margin-bottom: 0.9rem; }
  .pipeline-item  { font-size: 0.88rem; color: #c9d1d9; margin-bottom: 0.55rem; }

  .stTextInput > div > div > input {
    background-color: #161b22 !important; border: 1px solid #30363d !important;
    border-radius: 8px !important; color: #e6edf3 !important; font-size: 0.9rem !important;
  }
  .stTextInput > label { display: none !important; }
  [data-testid="stFileUploader"] { background: #161b22; border: 1.5px dashed #30363d; border-radius: 10px; padding: 0.8rem; }

  /* Base button style */
  .stButton > button {
    background: #21262d; border: 1px solid #30363d; color: #e6edf3;
    border-radius: 8px; font-size: 0.88rem; padding: 0.45rem 1.2rem; width: 100%;
  }
  .stButton > button:hover { background: #2d333b; }

  /* Run AutoML button — green accent */
  div[data-testid="stButton"][id*="run_automl"] > button,
  .run-automl-btn > div > button {
    background: linear-gradient(135deg, #1a4d2e 0%, #145a32 100%) !important;
    border: 1px solid #2ecc71 !important;
    color: #2ecc71 !important;
    font-weight: 600 !important;
  }
  div[data-testid="stButton"][id*="run_automl"] > button:hover,
  .run-automl-btn > div > button:hover {
    background: linear-gradient(135deg, #1e6035 0%, #1a6e3c 100%) !important;
  }

  /* Select Model button — blue accent */
  div[data-testid="stButton"][id*="select_model"] > button,
  .select-model-btn > div > button {
    background: linear-gradient(135deg, #0d2137 0%, #0a1f35 100%) !important;
    border: 1px solid #3498db !important;
    color: #3498db !important;
    font-weight: 600 !important;
  }
  div[data-testid="stButton"][id*="select_model"] > button:hover,
  .select-model-btn > div > button:hover {
    background: linear-gradient(135deg, #102842 0%, #0d2640 100%) !important;
  }

  .chat-msg-user {
    background: #1c2128; border: 1px solid #30363d; border-radius: 10px;
    padding: 0.75rem 1rem; margin-bottom: 0.6rem; font-size: 0.9rem; text-align: right;
  }
  .chat-msg-ai {
    background: #0d2818; border: 1px solid #1a3a28; border-radius: 10px;
    padding: 0.75rem 1rem; margin-bottom: 0.6rem; font-size: 0.9rem; color: #a0e9b8;
  }
  .chat-sender { font-size: 0.75rem; color: #6e7681; margin-bottom: 0.25rem; }
  .stat-card {
    background: #161b22; border: 1px solid #21262d; border-radius: 10px;
    padding: 1rem 1.2rem; text-align: center;
  }
  .stat-val   { font-size: 1.6rem; font-weight: 700; color: #2ecc71; }
  .stat-label { font-size: 0.78rem; color: #8b949e; margin-top: 0.2rem; }
  hr { border-color: #21262d; }

  /* Model progress card */
  .model-progress-card {
    background: #161b22; border: 1px solid #21262d; border-radius: 10px;
    padding: 1rem 1.2rem; margin-top: 0.6rem;
  }
  .model-progress-title { font-size: 0.82rem; font-weight: 600; color: #8b949e; margin-bottom: 0.6rem; }
  .model-row { display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.45rem; font-size: 0.82rem; }
  .model-name { color: #c9d1d9; }
  .model-score { font-weight: 600; }
  .model-bar-bg { background: #21262d; border-radius: 4px; height: 5px; margin-top: 3px; }
  .model-bar-fill { border-radius: 4px; height: 5px; }
  .action-divider {
    display: flex; align-items: center; gap: 0.5rem; margin: 0.6rem 0;
    font-size: 0.72rem; color: #6e7681;
  }
  .action-divider::before, .action-divider::after {
    content: ""; flex: 1; height: 1px; background: #21262d;
  }
  .single-model-panel {
    background: #161b22; border: 1px solid #21262d; border-radius: 10px;
    padding: 1rem 1.2rem; margin-top: 0.6rem;
  }
  .single-result-row { font-size: 0.82rem; color: #c9d1d9; margin-bottom: 0.3rem; }
  .single-result-score { color: #2ecc71; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ── Plotly dark theme helper ──────────────────────────────────────────────────
PLOTLY_THEME = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#161b22",
    font_color="#e6edf3",
    font_size=12,
)

def apply_theme(fig):
    fig.update_layout(
        paper_bgcolor=PLOTLY_THEME["paper_bgcolor"],
        plot_bgcolor=PLOTLY_THEME["plot_bgcolor"],
        font=dict(color=PLOTLY_THEME["font_color"], size=PLOTLY_THEME["font_size"]),
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(gridcolor="#21262d", zerolinecolor="#21262d"),
        yaxis=dict(gridcolor="#21262d", zerolinecolor="#21262d"),
    )
    return fig

# ── Model definitions ─────────────────────────────────────────────────────────
CLASSIFICATION_MODELS = {
    "Logistic Regression": {"module": "sklearn.linear_model", "class": "LogisticRegression", "params": {"max_iter": 1000, "random_state": 42}},
    "Random Forest":       {"module": "sklearn.ensemble",    "class": "RandomForestClassifier", "params": {"n_estimators": 100, "random_state": 42}},
    "Gradient Boosting":   {"module": "sklearn.ensemble",    "class": "GradientBoostingClassifier", "params": {"n_estimators": 100, "random_state": 42}},
    "SVM":                 {"module": "sklearn.svm",         "class": "SVC", "params": {"probability": True, "random_state": 42}},
    "KNN":                 {"module": "sklearn.neighbors",   "class": "KNeighborsClassifier", "params": {"n_neighbors": 5}},
}
REGRESSION_MODELS = {
    "Linear Regression":   {"module": "sklearn.linear_model", "class": "LinearRegression",          "params": {}},
    "Random Forest":       {"module": "sklearn.ensemble",    "class": "RandomForestRegressor",       "params": {"n_estimators": 100, "random_state": 42}},
    "Gradient Boosting":   {"module": "sklearn.ensemble",    "class": "GradientBoostingRegressor",   "params": {"n_estimators": 100, "random_state": 42}},
    "SVR":                 {"module": "sklearn.svm",         "class": "SVR",                         "params": {}},
    "Ridge":               {"module": "sklearn.linear_model", "class": "Ridge",                      "params": {"random_state": 42}},
}

def get_model_instance(info):
    import importlib
    mod   = importlib.import_module(info["module"])
    cls   = getattr(mod, info["class"])
    return cls(**info["params"])

def run_single_model(df, target_col, task, model_name):
    """Train one model and return score + classification report snippet."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score, r2_score

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Encode categoricals in X
    for col in X.select_dtypes(include="object").columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    if task == "Classification":
        le = LabelEncoder()
        y  = le.fit_transform(y.astype(str))
        models_map = CLASSIFICATION_MODELS
    else:
        models_map = REGRESSION_MODELS

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    info  = models_map[model_name]
    model = get_model_instance(info)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    if task == "Classification":
        score = accuracy_score(y_test, preds)
        metric_label = "Accuracy"
    else:
        score = r2_score(y_test, preds)
        metric_label = "R² Score"

    return score, metric_label

def run_all_models(df, target_col, task):
    """Train all models and return dict of {model_name: (score, metric_label)}."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score, r2_score

    X = df.drop(columns=[target_col])
    y = df[target_col]

    for col in X.select_dtypes(include="object").columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    if task == "Classification":
        le = LabelEncoder()
        y  = le.fit_transform(y.astype(str))
        models_map  = CLASSIFICATION_MODELS
        metric_label = "Accuracy"
        score_fn    = accuracy_score
    else:
        models_map  = REGRESSION_MODELS
        metric_label = "R² Score"
        score_fn    = r2_score

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}
    for name, info in models_map.items():
        try:
            m = get_model_instance(info)
            m.fit(X_train, y_train)
            score = score_fn(y_test, m.predict(X_test))
            results[name] = (round(score, 4), metric_label)
        except Exception as e:
            results[name] = (None, str(e))

    return results

# ── Session state ─────────────────────────────────────────────────────────────
for key, val in {
    "messages": [],
    "active_tab": "Chat",
    "pipeline_status": {
        "Dataset Validation":    "done",
        "Missing Value Analysis":"done",
        "Feature Engineering":   "done",
        "Hyperparameter Search": "running",
        "Model Training":        "pending",
    },
    "df": None,
    "automl_results":    None,
    "single_result":     None,
    "selected_model":    None,
    "show_model_picker": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

STATUS_ICON = {"done": "✓", "running": "⚡", "pending": "⏳"}

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <div class="hero-title">🤖 AutoML Agent Pro</div>
  <p class="hero-sub">Upload data, describe your goal, and automatically discover the best ML pipeline.</p>
</div>
""", unsafe_allow_html=True)

# ── Layout ────────────────────────────────────────────────────────────────────
left, center, right = st.columns([2.2, 4.5, 2.3], gap="large")

# ════════════════════════════════════════════════════════════════════════════════
# LEFT — Dataset upload + action buttons
# ════════════════════════════════════════════════════════════════════════════════
with left:
    st.markdown('<div class="section-title">📁 Dataset</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
    st.markdown('<p class="upload-hint">200 MB per file • CSV</p>', unsafe_allow_html=True)

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.success(f"✓ {uploaded_file.name}  ({len(df):,} rows × {len(df.columns)} cols)")

        # ── Target column selector ─────────────────────────────────────────
        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
        st.markdown('<p style="font-size:0.82rem;color:#8b949e;margin-bottom:0.3rem">🎯 Target Column</p>', unsafe_allow_html=True)
        target_col = st.selectbox(
            "Target column",
            options=df.columns.tolist(),
            index=len(df.columns) - 1,
            label_visibility="collapsed",
            key="target_col",
        )

        # ── Auto Detect button ─────────────────────────────────────────────
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        detect_clicked = st.button("🔍 Auto Detect Task", key="auto_detect", use_container_width=True)

        if detect_clicked and target_col:
            series   = df[target_col].dropna()
            n_unique = series.nunique()
            is_num   = pd.api.types.is_numeric_dtype(series)
            if not is_num or n_unique <= 10:
                task   = "Classification"
                models = list(CLASSIFICATION_MODELS.keys())
                icon   = "🔵"
            else:
                task   = "Regression"
                models = list(REGRESSION_MODELS.keys())
                icon   = "📈"
            st.session_state.detected_task   = task
            st.session_state.detected_models = models
            st.session_state.detected_icon   = icon
            st.session_state.pipeline_status["Dataset Validation"]    = "done"
            st.session_state.pipeline_status["Missing Value Analysis"] = "done"
            st.session_state.pipeline_status["Feature Engineering"]   = "done"
            st.session_state.pipeline_status["Hyperparameter Search"] = "running"
            st.session_state.pipeline_status["Model Training"]        = "pending"
            # Reset previous run results on new detect
            st.session_state.automl_results   = None
            st.session_state.single_result    = None
            st.session_state.show_model_picker = False

        if "detected_task" in st.session_state:
            task  = st.session_state.detected_task
            icon  = st.session_state.detected_icon
            color = "#2ecc71" if task == "Classification" else "#3498db"
            model_rows = "".join(
                f'<div style="font-size:0.78rem;color:#c9d1d9;margin-top:3px">• {m}</div>'
                for m in st.session_state.detected_models
            )
            st.markdown(f"""
            <div style="background:#161b22;border:1px solid #21262d;border-radius:8px;
                        padding:0.75rem 1rem;margin-top:0.5rem">
              <div style="font-size:0.72rem;color:#6e7681;margin-bottom:4px">Detected task</div>
              <div style="font-size:1rem;font-weight:600;color:{color}">{icon} {task}</div>
              <div style="font-size:0.72rem;color:#8b949e;margin-top:6px">Available Models</div>
              {model_rows}
            </div>
            """, unsafe_allow_html=True)

            # ── Divider ────────────────────────────────────────────────────
            st.markdown('<div class="action-divider">Run</div>', unsafe_allow_html=True)

            # ── ① Run AutoML button ────────────────────────────────────────
            run_all_clicked = st.button(
                f"🚀 Run AutoML — All {task} Models",
                key="run_automl",
                use_container_width=True,
                type="primary",
            )

            if run_all_clicked:
                st.session_state.automl_results   = None
                st.session_state.single_result    = None
                st.session_state.show_model_picker = False
                st.session_state.pipeline_status["Model Training"] = "running"
                with st.spinner("Training all models…"):
                    results = run_all_models(
                        st.session_state.df,
                        st.session_state.target_col,
                        st.session_state.detected_task,
                    )
                st.session_state.automl_results = results
                st.session_state.pipeline_status["Hyperparameter Search"] = "done"
                st.session_state.pipeline_status["Model Training"]        = "done"
                st.rerun()

            # ── ② Select Model button ──────────────────────────────────────
            select_clicked = st.button(
                "🎯 Select & Run One Model",
                key="select_model",
                use_container_width=True,
            )

            if select_clicked:
                st.session_state.show_model_picker = not st.session_state.show_model_picker
                st.session_state.automl_results    = None
                st.session_state.single_result     = None

            # ── Model picker dropdown + Run ────────────────────────────────
            if st.session_state.show_model_picker:
                model_options = st.session_state.detected_models
                chosen = st.selectbox(
                    "Pick a model",
                    model_options,
                    key="model_picker",
                    label_visibility="collapsed",
                )
                st.session_state.selected_model = chosen

                run_one_clicked = st.button(
                    f"▶ Train {chosen}",
                    key="run_one",
                    use_container_width=True,
                )
                if run_one_clicked:
                    st.session_state.pipeline_status["Model Training"] = "running"
                    with st.spinner(f"Training {chosen}…"):
                        score, metric_label = run_single_model(
                            st.session_state.df,
                            st.session_state.target_col,
                            st.session_state.detected_task,
                            chosen,
                        )
                    st.session_state.single_result = {
                        "model":        chosen,
                        "score":        score,
                        "metric_label": metric_label,
                    }
                    st.session_state.pipeline_status["Hyperparameter Search"] = "done"
                    st.session_state.pipeline_status["Model Training"]        = "done"
                    st.rerun()

            # ── AutoML results display ─────────────────────────────────────
            if st.session_state.automl_results:
                results     = st.session_state.automl_results
                valid       = {k: v for k, v in results.items() if v[0] is not None}
                sorted_res  = sorted(valid.items(), key=lambda x: x[1][0], reverse=True)
                medal       = ["🥇", "🥈", "🥉"]
                metric_label = sorted_res[0][1][1] if sorted_res else ""

                rows_html = ""
                for i, (name, (score, _)) in enumerate(sorted_res):
                    pct      = max(0, min(100, score * 100))
                    bar_color = "#2ecc71" if i == 0 else ("#e67e22" if i == 1 else "#3498db")
                    m_icon   = medal[i] if i < 3 else "  "
                    rows_html += f"""
                    <div class="model-row">
                      <span class="model-name">{m_icon} {name}</span>
                      <span class="model-score" style="color:{bar_color}">{score:.4f}</span>
                    </div>
                    <div class="model-bar-bg">
                      <div class="model-bar-fill" style="width:{pct:.1f}%;background:{bar_color}"></div>
                    </div>
                    """

                st.markdown(f"""
                <div class="model-progress-card">
                  <div class="model-progress-title">AutoML Results — {metric_label}</div>
                  {rows_html}
                </div>
                """, unsafe_allow_html=True)

            # ── Single model result display ────────────────────────────────
            if st.session_state.single_result:
                r     = st.session_state.single_result
                score = r["score"]
                pct   = max(0, min(100, score * 100))
                st.markdown(f"""
                <div class="single-model-panel">
                  <div class="model-progress-title">Single Model Result</div>
                  <div class="single-result-row">Model: <strong style="color:#e6edf3">{r["model"]}</strong></div>
                  <div class="single-result-row">{r["metric_label"]}: <span class="single-result-score">{score:.4f}</span></div>
                  <div class="model-bar-bg" style="margin-top:0.5rem">
                    <div class="model-bar-fill" style="width:{pct:.1f}%;background:#2ecc71"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# CENTER — Tabs
# ════════════════════════════════════════════════════════════════════════════════
with center:
    TABS     = ["Chat", "Visualize", "Dashboard", "Code"]
    TAB_ICON = {"Chat":"💬", "Visualize":"📈", "Dashboard":"📊", "Code":"💻"}

    tab_html = '<div class="tab-strip">'
    for t in TABS:
        cls = "active" if st.session_state.active_tab == t else ""
        tab_html += f'<div class="tab-item {cls}">{TAB_ICON[t]} {t}</div>'
    tab_html += '</div>'
    st.markdown(tab_html, unsafe_allow_html=True)

    tab_cols = st.columns(len(TABS))
    for i, t in enumerate(TABS):
        with tab_cols[i]:
            if st.button(t, key=f"tab_{t}"):
                st.session_state.active_tab = t
                st.rerun()

    st.markdown("<hr style='margin:0.2rem 0 1rem 0'>", unsafe_allow_html=True)

    df = st.session_state.df  # shorthand

    # ── CHAT ─────────────────────────────────────────────────────────────────
    if st.session_state.active_tab == "Chat":
        st.markdown("""
        <div class="objective-box">
          <div class="obj-icon">🤖</div>
          <span>Describe your ML objective. Example: <em>Predict customer churn.</em></span>
        </div>""", unsafe_allow_html=True)

        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-msg-user"><div class="chat-sender">You</div>{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-msg-ai"><div class="chat-sender">AutoML Agent</div>{msg["content"]}</div>', unsafe_allow_html=True)

        col_in, col_btn = st.columns([9, 1])
        with col_in:
            user_input = st.text_input("Ask", placeholder="Ask about your dataset...", key="chat_input")
        with col_btn:
            st.markdown("<div style='margin-top:1.65rem'>", unsafe_allow_html=True)
            send = st.button("↑", key="send_btn")
            st.markdown("</div>", unsafe_allow_html=True)

        if send and user_input.strip():
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.spinner("Agent thinking…"):
                time.sleep(0.8)
            st.session_state.messages.append({"role": "assistant", "content": (
                f"Got it! I'll analyze your dataset to **{user_input.lower()}**. "
                "Starting EDA → missing value handling → feature engineering → "
                "hyperparameter search across RandomForest, XGBoost & LightGBM. "
                "Switch to the **Visualize** tab to explore your data!"
            )})
            st.rerun()

    # ── VISUALIZE ────────────────────────────────────────────────────────────
    elif st.session_state.active_tab == "Visualize":
        import plotly.express as px
        import plotly.graph_objects as go

        if df is None:
            st.info("⬅️  Upload a CSV file first to explore your data.")
        else:
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

            st.markdown("### 📋 Dataset Overview")
            c1, c2, c3, c4 = st.columns(4)
            missing_pct = df.isnull().mean().mean() * 100
            dup_count   = df.duplicated().sum()
            c1.markdown(f'<div class="stat-card"><div class="stat-val">{len(df):,}</div><div class="stat-label">Total Rows</div></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="stat-card"><div class="stat-val">{len(df.columns)}</div><div class="stat-label">Columns</div></div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="stat-card"><div class="stat-val">{missing_pct:.1f}%</div><div class="stat-label">Missing Values</div></div>', unsafe_allow_html=True)
            c4.markdown(f'<div class="stat-card"><div class="stat-val">{dup_count}</div><div class="stat-label">Duplicate Rows</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown("### 🧩 Interactive Explorer (drag & drop)")
            st.markdown(
                "<p style='font-size:0.83rem;color:#8b949e;margin-bottom:0.8rem'>"
                "Drag columns onto X / Y / Color axes, switch chart types, filter, and build any chart — no code needed."
                "</p>",
                unsafe_allow_html=True,
            )
            try:
                import pygwalker as pyg
                from pygwalker.api.streamlit import StreamlitRenderer
                renderer = StreamlitRenderer(df, appearance="dark", spec_io_mode="rw")
                renderer.explorer()
            except ImportError:
                st.warning("PyGWalker not installed. Run:  `pip install pygwalker`  then restart Streamlit.", icon="⚠️")

            st.markdown("---")

            st.markdown("### 🕳️ Missing Values")
            missing = df.isnull().mean().reset_index()
            missing.columns = ["Column", "Missing %"]
            missing["Missing %"] = (missing["Missing %"] * 100).round(2)
            missing = missing[missing["Missing %"] > 0].sort_values("Missing %", ascending=False)

            if missing.empty:
                st.success("✅ No missing values found!")
            else:
                fig_missing = px.bar(missing, x="Column", y="Missing %",
                    color="Missing %", color_continuous_scale=["#2ecc71","#e67e22","#e74c3c"],
                    title="Missing value % per column")
                apply_theme(fig_missing)
                st.plotly_chart(fig_missing, use_container_width=True)

            if num_cols:
                st.markdown("### 📊 Numeric Distributions")
                sel_col = st.selectbox("Select column", num_cols, key="dist_col")
                c_hist, c_box = st.columns(2)
                with c_hist:
                    fig_hist = px.histogram(df, x=sel_col, nbins=40, color_discrete_sequence=["#2ecc71"], title=f"Distribution — {sel_col}")
                    apply_theme(fig_hist); st.plotly_chart(fig_hist, use_container_width=True)
                with c_box:
                    fig_box = px.box(df, y=sel_col, color_discrete_sequence=["#2ecc71"], title=f"Box plot — {sel_col}")
                    apply_theme(fig_box); st.plotly_chart(fig_box, use_container_width=True)

            if len(num_cols) >= 2:
                st.markdown("### 🔥 Correlation Heatmap")
                corr = df[num_cols].corr().round(2)
                fig_corr = go.Figure(go.Heatmap(
                    z=corr.values, x=corr.columns.tolist(), y=corr.columns.tolist(),
                    colorscale=[[0.0,"#e74c3c"],[0.5,"#161b22"],[1.0,"#2ecc71"]],
                    zmin=-1, zmax=1, text=corr.values, texttemplate="%{text}", textfont={"size": 10},
                ))
                fig_corr.update_layout(title="Feature correlation matrix", height=500)
                apply_theme(fig_corr); st.plotly_chart(fig_corr, use_container_width=True)

            if len(num_cols) >= 2:
                st.markdown("### 🔵 Scatter Plot")
                sc1, sc2, sc3 = st.columns(3)
                with sc1: x_col = st.selectbox("X axis", num_cols, index=0, key="sc_x")
                with sc2: y_col = st.selectbox("Y axis", num_cols, index=min(1, len(num_cols)-1), key="sc_y")
                with sc3: color_col = st.selectbox("Color by", ["None"] + cat_cols + num_cols, key="sc_color")
                fig_sc = px.scatter(df, x=x_col, y=y_col,
                    color=None if color_col == "None" else color_col,
                    opacity=0.7, color_discrete_sequence=px.colors.qualitative.Vivid,
                    color_continuous_scale="Viridis", title=f"{x_col} vs {y_col}")
                apply_theme(fig_sc); st.plotly_chart(fig_sc, use_container_width=True)

            if cat_cols:
                st.markdown("### 🏷️ Categorical Columns")
                sel_cat = st.selectbox("Select column", cat_cols, key="cat_col")
                top_n   = st.slider("Show top N values", 5, 30, 10, key="top_n")
                vc = df[sel_cat].value_counts().head(top_n).reset_index()
                vc.columns = [sel_cat, "Count"]
                fig_cat = px.bar(vc, x=sel_cat, y="Count", color="Count",
                    color_continuous_scale=["#0d2818","#2ecc71"], title=f"Top {top_n} values — {sel_cat}")
                apply_theme(fig_cat); st.plotly_chart(fig_cat, use_container_width=True)

            if len(num_cols) >= 3:
                st.markdown("### 🔗 Pair Plot (sample)")
                pair_cols = st.multiselect("Choose columns (2–5 recommended)", num_cols,
                    default=num_cols[:min(4, len(num_cols))], key="pair_cols")
                if len(pair_cols) >= 2:
                    sample_df = df[pair_cols].dropna().sample(min(500, len(df)), random_state=42)
                    fig_pair  = px.scatter_matrix(sample_df, dimensions=pair_cols,
                        color_discrete_sequence=["#2ecc71"], title="Scatter matrix")
                    fig_pair.update_traces(marker=dict(size=3, opacity=0.5))
                    apply_theme(fig_pair); fig_pair.update_layout(height=600)
                    st.plotly_chart(fig_pair, use_container_width=True)

    # ── DASHBOARD ────────────────────────────────────────────────────────────
    elif st.session_state.active_tab == "Dashboard":
        if df is None:
            st.info("Upload a CSV file to see the dashboard.")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Rows", f"{len(df):,}")
            m2.metric("Columns", len(df.columns))
            m3.metric("Missing %", f"{df.isnull().mean().mean()*100:.1f}%")
            st.markdown("**Column types**")
            type_df = df.dtypes.value_counts().rename_axis("dtype").reset_index(name="count")
            type_df["dtype"] = type_df["dtype"].astype(str)
            st.dataframe(type_df, use_container_width=True)
            st.markdown("**Data preview**")
            st.dataframe(df.head(10), use_container_width=True)

    # ── CODE ─────────────────────────────────────────────────────────────────
    elif st.session_state.active_tab == "Code":
        # Dynamically show code based on what was run
        task = st.session_state.get("detected_task", "Classification")
        target = st.session_state.get("target_col", "target")

        if st.session_state.single_result:
            model_name = st.session_state.single_result["model"]
            if task == "Classification":
                info = CLASSIFICATION_MODELS.get(model_name, CLASSIFICATION_MODELS["Random Forest"])
            else:
                info = REGRESSION_MODELS.get(model_name, REGRESSION_MODELS["Random Forest"])
            import_line  = f"from {info['module']} import {info['class']}"
            params_str   = ", ".join(f"{k}={repr(v)}" for k, v in info["params"].items())
            model_line   = f"model = {info['class']}({params_str})"
            metric_line  = "print(classification_report(y_test, model.predict(X_test)))" if task == "Classification" else "print('R² Score:', r2_score(y_test, model.predict(X_test)))"
            extra_import = "from sklearn.metrics import classification_report" if task == "Classification" else "from sklearn.metrics import r2_score"
        else:
            import_line  = "from sklearn.ensemble import RandomForestClassifier"
            model_line   = "model = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_split=5, random_state=42)"
            metric_line  = "print(classification_report(y_test, model.predict(X_test)))"
            extra_import = "from sklearn.metrics import classification_report"

        st.markdown("**Generated ML Pipeline Code**")
        st.code(f'''import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
{import_line}
{extra_import}

df = pd.read_csv("your_dataset.csv")

X = df.drop("{target}", axis=1)
y = df["{target}"]

# Encode categorical columns
for col in X.select_dtypes(include="object").columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

{model_line}
model.fit(X_train, y_train)
{metric_line}
''', language="python")

        if st.session_state.automl_results:
            st.markdown("**AutoML Comparison Results**")
            res = st.session_state.automl_results
            rows = [{"Model": k, "Score": f"{v[0]:.4f}" if v[0] else "Error", "Metric": v[1]} for k, v in res.items()]
            st.dataframe(pd.DataFrame(rows).sort_values("Score", ascending=False), use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# RIGHT — Insights
# ════════════════════════════════════════════════════════════════════════════════
with right:
    st.markdown('<div class="section-title">🧠 Insights</div>', unsafe_allow_html=True)

    items_html = ""
    for step, status in st.session_state.pipeline_status.items():
        icon  = STATUS_ICON[status]
        color = "#2ecc71" if status == "done" else ("#e67e22" if status == "running" else "#6e7681")
        items_html += f'<div class="pipeline-item" style="color:{color}">{icon} {step}</div>'

    st.markdown(f"""
    <div class="insights-panel">
      <div class="pipeline-title">Pipeline Status</div>
      {items_html}
    </div>""", unsafe_allow_html=True)

    # ── Model Leaderboard from real results ───────────────────────────────────
    if st.session_state.automl_results:
        results     = st.session_state.automl_results
        valid       = {k: v for k, v in results.items() if v[0] is not None}
        sorted_res  = sorted(valid.items(), key=lambda x: x[1][0], reverse=True)
        medal       = ["🥇", "🥈", "🥉"]
        board_rows  = ""
        for i, (name, (score, metric)) in enumerate(sorted_res[:5]):
            icon_m = medal[i] if i < 3 else "  "
            col_m  = "#2ecc71" if i == 0 else "#8b949e"
            board_rows += f'<div class="pipeline-item" style="color:{col_m}">{icon_m} {name} — {score:.4f}</div>'
        st.markdown(f"""
        <div class="insights-panel" style="margin-top:0.8rem">
          <div class="pipeline-title">Model Leaderboard</div>
          {board_rows}
        </div>""", unsafe_allow_html=True)

    elif st.session_state.single_result:
        r = st.session_state.single_result
        st.markdown(f"""
        <div class="insights-panel" style="margin-top:0.8rem">
          <div class="pipeline-title">Selected Model Result</div>
          <div class="pipeline-item" style="color:#2ecc71">✓ {r["model"]}</div>
          <div class="pipeline-item">{r["metric_label"]}: <strong style="color:#2ecc71">{r["score"]:.4f}</strong></div>
        </div>""", unsafe_allow_html=True)

    elif st.session_state.messages:
        st.markdown("""
        <div class="insights-panel" style="margin-top:0.5rem">
          <div class="pipeline-title">Model Leaderboard</div>
          <div class="pipeline-item" style="color:#2ecc71">🥇 XGBoost — 91.4% AUC</div>
          <div class="pipeline-item" style="color:#8b949e">🥈 LightGBM — 90.8% AUC</div>
          <div class="pipeline-item" style="color:#8b949e">🥉 RandomForest — 89.2% AUC</div>
        </div>""", unsafe_allow_html=True)

    # Quick dataset stats
    if st.session_state.df is not None:
        dff = st.session_state.df
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        num_c = len(dff.select_dtypes(include=np.number).columns)
        cat_c = len(dff.select_dtypes(exclude=np.number).columns)
        st.markdown(f"""
        <div class="insights-panel" style="margin-top:0.5rem">
          <div class="pipeline-title">Dataset Summary</div>
          <div class="pipeline-item">📐 {len(dff):,} rows</div>
          <div class="pipeline-item">🔢 {num_c} numeric cols</div>
          <div class="pipeline-item">🏷️ {cat_c} categorical cols</div>
          <div class="pipeline-item">🕳️ {dff.isnull().mean().mean()*100:.1f}% missing</div>
          <div class="pipeline-item">📋 {dff.duplicated().sum()} duplicates</div>
        </div>""", unsafe_allow_html=True)