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
  .stButton > button {
    background: #21262d; border: 1px solid #30363d; color: #e6edf3;
    border-radius: 8px; font-size: 0.88rem; padding: 0.45rem 1.2rem; width: 100%;
  }
  .stButton > button:hover { background: #2d333b; }
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
# LEFT — Dataset upload
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
                models = ["Logistic Regression", "Random Forest", "Gradient Boosting"]
                icon   = "🔵"
            else:
                task   = "Regression"
                models = ["Linear Regression", "Random Forest", "Gradient Boosting"]
                icon   = "📈"
            st.session_state.detected_task   = task
            st.session_state.detected_models = models
            st.session_state.detected_icon   = icon
            st.session_state.pipeline_status["Dataset Validation"]    = "done"
            st.session_state.pipeline_status["Missing Value Analysis"] = "done"
            st.session_state.pipeline_status["Feature Engineering"]   = "done"
            st.session_state.pipeline_status["Hyperparameter Search"] = "running"
            st.session_state.pipeline_status["Model Training"]        = "pending"

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
              <div style="font-size:0.72rem;color:#8b949e;margin-top:6px">Models</div>
              {model_rows}
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

            # ── Section 1: Overview stats ──────────────────────────────────
            st.markdown("### 📋 Dataset Overview")
            c1, c2, c3, c4 = st.columns(4)
            missing_pct = df.isnull().mean().mean() * 100
            dup_count   = df.duplicated().sum()
            c1.markdown(f'<div class="stat-card"><div class="stat-val">{len(df):,}</div><div class="stat-label">Total Rows</div></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="stat-card"><div class="stat-val">{len(df.columns)}</div><div class="stat-label">Columns</div></div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="stat-card"><div class="stat-val">{missing_pct:.1f}%</div><div class="stat-label">Missing Values</div></div>', unsafe_allow_html=True)
            c4.markdown(f'<div class="stat-card"><div class="stat-val">{dup_count}</div><div class="stat-label">Duplicate Rows</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Section 2: PyGWalker — drag-and-drop Tableau-like explorer ─
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
                st.warning(
                    "PyGWalker not installed. Run:  `pip install pygwalker`  then restart Streamlit.",
                    icon="⚠️",
                )

            st.markdown("---")

            # ── Section 3: Missing values ──────────────────────────────────
            st.markdown("### 🕳️ Missing Values")
            missing = df.isnull().mean().reset_index()
            missing.columns = ["Column", "Missing %"]
            missing["Missing %"] = (missing["Missing %"] * 100).round(2)
            missing = missing[missing["Missing %"] > 0].sort_values("Missing %", ascending=False)

            if missing.empty:
                st.success("✅ No missing values found!")
            else:
                fig_missing = px.bar(
                    missing, x="Column", y="Missing %",
                    color="Missing %", color_continuous_scale=["#2ecc71","#e67e22","#e74c3c"],
                    title="Missing value % per column",
                )
                apply_theme(fig_missing)
                st.plotly_chart(fig_missing, use_container_width=True)

            # ── Section 4: Distributions ───────────────────────────────────
            if num_cols:
                st.markdown("### 📊 Numeric Distributions")
                sel_col = st.selectbox("Select column", num_cols, key="dist_col")
                c_hist, c_box = st.columns(2)

                with c_hist:
                    fig_hist = px.histogram(
                        df, x=sel_col, nbins=40,
                        color_discrete_sequence=["#2ecc71"],
                        title=f"Distribution — {sel_col}",
                    )
                    apply_theme(fig_hist)
                    st.plotly_chart(fig_hist, use_container_width=True)

                with c_box:
                    fig_box = px.box(
                        df, y=sel_col,
                        color_discrete_sequence=["#2ecc71"],
                        title=f"Box plot — {sel_col}",
                    )
                    apply_theme(fig_box)
                    st.plotly_chart(fig_box, use_container_width=True)

            # ── Section 5: Correlation heatmap ────────────────────────────
            if len(num_cols) >= 2:
                st.markdown("### 🔥 Correlation Heatmap")
                corr = df[num_cols].corr().round(2)
                fig_corr = go.Figure(go.Heatmap(
                    z=corr.values,
                    x=corr.columns.tolist(),
                    y=corr.columns.tolist(),
                    colorscale=[[0.0,"#e74c3c"],[0.5,"#161b22"],[1.0,"#2ecc71"]],
                    zmin=-1, zmax=1,
                    text=corr.values,
                    texttemplate="%{text}",
                    textfont={"size": 10},
                ))
                fig_corr.update_layout(title="Feature correlation matrix", height=500)
                apply_theme(fig_corr)
                st.plotly_chart(fig_corr, use_container_width=True)

            # ── Section 6: Scatter plot ───────────────────────────────────
            if len(num_cols) >= 2:
                st.markdown("### 🔵 Scatter Plot")
                sc1, sc2, sc3 = st.columns(3)
                with sc1:
                    x_col = st.selectbox("X axis", num_cols, index=0, key="sc_x")
                with sc2:
                    y_col = st.selectbox("Y axis", num_cols, index=min(1, len(num_cols)-1), key="sc_y")
                with sc3:
                    color_col = st.selectbox("Color by", ["None"] + cat_cols + num_cols, key="sc_color")

                fig_sc = px.scatter(
                    df, x=x_col, y=y_col,
                    color=None if color_col == "None" else color_col,
                    opacity=0.7,
                    color_discrete_sequence=px.colors.qualitative.Vivid,
                    color_continuous_scale="Viridis",
                    title=f"{x_col} vs {y_col}",
                )
                apply_theme(fig_sc)
                st.plotly_chart(fig_sc, use_container_width=True)

            # ── Section 7: Categorical bar charts ────────────────────────
            if cat_cols:
                st.markdown("### 🏷️ Categorical Columns")
                sel_cat = st.selectbox("Select column", cat_cols, key="cat_col")
                top_n   = st.slider("Show top N values", 5, 30, 10, key="top_n")

                vc = df[sel_cat].value_counts().head(top_n).reset_index()
                vc.columns = [sel_cat, "Count"]

                fig_cat = px.bar(
                    vc, x=sel_cat, y="Count",
                    color="Count",
                    color_continuous_scale=["#0d2818","#2ecc71"],
                    title=f"Top {top_n} values — {sel_cat}",
                )
                apply_theme(fig_cat)
                st.plotly_chart(fig_cat, use_container_width=True)

            # ── Section 7: Pairplot (sample) ─────────────────────────────
            if len(num_cols) >= 3:
                st.markdown("### 🔗 Pair Plot (sample)")
                pair_cols = st.multiselect(
                    "Choose columns (2–5 recommended)",
                    num_cols,
                    default=num_cols[:min(4, len(num_cols))],
                    key="pair_cols",
                )
                if len(pair_cols) >= 2:
                    sample_df = df[pair_cols].dropna().sample(min(500, len(df)), random_state=42)
                    fig_pair  = px.scatter_matrix(
                        sample_df,
                        dimensions=pair_cols,
                        color_discrete_sequence=["#2ecc71"],
                        title="Scatter matrix",
                    )
                    fig_pair.update_traces(marker=dict(size=3, opacity=0.5))
                    apply_theme(fig_pair)
                    fig_pair.update_layout(height=600)
                    st.plotly_chart(fig_pair, use_container_width=True)

    # ── DASHBOARD ────────────────────────────────────────────────────────────
    elif st.session_state.active_tab == "Dashboard":
        if df is None:
            st.info("Upload a CSV file to see the dashboard.")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Rows",       f"{len(df):,}")
            m2.metric("Columns",    len(df.columns))
            m3.metric("Missing %",  f"{df.isnull().mean().mean()*100:.1f}%")
            st.markdown("**Column types**")
            type_df = df.dtypes.value_counts().rename_axis("dtype").reset_index(name="count")
            type_df["dtype"] = type_df["dtype"].astype(str)
            st.dataframe(type_df, use_container_width=True)
            st.markdown("**Data preview**")
            st.dataframe(df.head(10), use_container_width=True)

    # ── CODE ─────────────────────────────────────────────────────────────────
    elif st.session_state.active_tab == "Code":
        st.markdown("**Generated ML Pipeline Code**")
        st.code('''import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df = pd.read_csv("your_dataset.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=200, max_depth=12,
    min_samples_split=5, random_state=42
)
model.fit(X_train, y_train)
print(classification_report(y_test, model.predict(X_test)))
''', language="python")

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

    if st.session_state.messages:
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="insights-panel" style="margin-top:0.5rem">
          <div class="pipeline-title">Model Leaderboard</div>
          <div class="pipeline-item" style="color:#2ecc71">🥇 XGBoost — 91.4% AUC</div>
          <div class="pipeline-item" style="color:#8b949e">🥈 LightGBM — 90.8% AUC</div>
          <div class="pipeline-item" style="color:#8b949e">🥉 RandomForest — 89.2% AUC</div>
        </div>""", unsafe_allow_html=True)

    # Quick dataset stats in sidebar if data loaded
    if df is not None:
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        num_c = len(df.select_dtypes(include=np.number).columns)
        cat_c = len(df.select_dtypes(exclude=np.number).columns)
        st.markdown(f"""
        <div class="insights-panel" style="margin-top:0.5rem">
          <div class="pipeline-title">Dataset Summary</div>
          <div class="pipeline-item">📐 {len(df):,} rows</div>
          <div class="pipeline-item">🔢 {num_c} numeric cols</div>
          <div class="pipeline-item">🏷️ {cat_c} categorical cols</div>
          <div class="pipeline-item">🕳️ {df.isnull().mean().mean()*100:.1f}% missing</div>
          <div class="pipeline-item">📋 {df.duplicated().sum()} duplicates</div>
        </div>""", unsafe_allow_html=True)
    
