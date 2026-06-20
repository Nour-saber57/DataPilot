import streamlit as st
import time

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoML Agent Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Global ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
  }

  /* hide default Streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding: 2rem 2.5rem 2rem 2.5rem; max-width: 100%; }

  /* ── Deploy button (top-right) ── */
  .deploy-btn {
    position: fixed;
    top: 1rem;
    right: 1.5rem;
    background: transparent;
    border: 1px solid #30363d;
    color: #e6edf3;
    padding: 0.35rem 1rem;
    border-radius: 6px;
    font-size: 0.85rem;
    cursor: pointer;
    z-index: 999;
  }

  /* ── Hero banner ── */
  .hero-banner {
    background: linear-gradient(135deg, #0d2818 0%, #0a1f14 60%, #061a10 100%);
    border: 1px solid #1a3a28;
    border-radius: 14px;
    padding: 2.2rem 2.5rem;
    margin-bottom: 2.2rem;
  }
  .hero-title {
    font-size: 2rem;
    font-weight: 700;
    color: #2ecc71;
    margin: 0 0 0.5rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  .hero-sub {
    font-size: 0.97rem;
    color: #8b949e;
    margin: 0;
  }

  /* ── Section headers ── */
  .section-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #e6edf3;
    display: flex;
    align-items: center;
    gap: 0.4rem;
    margin-bottom: 1rem;
  }

  /* ── Upload card ── */
  .upload-card {
    background: #161b22;
    border: 1px dashed #30363d;
    border-radius: 10px;
    padding: 1.4rem;
    text-align: center;
    margin-top: 0.5rem;
  }
  .upload-hint { font-size: 0.78rem; color: #6e7681; margin-top: 0.5rem; }

  /* ── Tab strip ── */
  .tab-strip {
    display: flex;
    border-bottom: 1px solid #21262d;
    margin-bottom: 1.2rem;
    gap: 0;
  }
  .tab-item {
    padding: 0.55rem 1.4rem;
    font-size: 0.9rem;
    color: #8b949e;
    cursor: pointer;
    border-bottom: 2px solid transparent;
    transition: color 0.2s;
  }
  .tab-item.active {
    color: #e6edf3;
    border-bottom: 2px solid #e74c3c;
  }

  /* ── AI objective box ── */
  .objective-box {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    display: flex;
    align-items: flex-start;
    gap: 0.8rem;
    margin-bottom: 1.2rem;
    color: #6e7681;
    font-size: 0.9rem;
  }
  .obj-icon {
    background: #e67e22;
    border-radius: 8px;
    width: 32px;
    height: 32px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    flex-shrink: 0;
  }

  /* ── Insights panel ── */
  .insights-panel {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
  }
  .pipeline-title {
    color: #2ecc71;
    font-weight: 600;
    font-size: 0.95rem;
    margin-bottom: 0.9rem;
  }
  .pipeline-item {
    font-size: 0.88rem;
    color: #c9d1d9;
    margin-bottom: 0.55rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
  }

  /* ── Chat input override ── */
  .stTextInput > div > div > input {
    background-color: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    color: #e6edf3 !important;
    font-size: 0.9rem !important;
    padding: 0.7rem 1rem !important;
  }
  .stTextInput > div > div > input::placeholder { color: #6e7681 !important; }
  .stTextInput > label { display: none !important; }

  /* ── File uploader ── */
  [data-testid="stFileUploader"] {
    background: #161b22;
    border: 1.5px dashed #30363d;
    border-radius: 10px;
    padding: 0.8rem;
  }
  [data-testid="stFileUploader"] label { color: #8b949e !important; }

  /* ── Buttons ── */
  .stButton > button {
    background: #21262d;
    border: 1px solid #30363d;
    color: #e6edf3;
    border-radius: 8px;
    font-size: 0.88rem;
    padding: 0.45rem 1.2rem;
    width: 100%;
    transition: background 0.2s;
  }
  .stButton > button:hover { background: #2d333b; border-color: #444c56; }

  /* ── Chat messages ── */
  .chat-msg-user {
    background: #1c2128;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.6rem;
    font-size: 0.9rem;
    text-align: right;
  }
  .chat-msg-ai {
    background: #0d2818;
    border: 1px solid #1a3a28;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.6rem;
    font-size: 0.9rem;
    color: #a0e9b8;
  }
  .chat-sender { font-size: 0.75rem; color: #6e7681; margin-bottom: 0.25rem; }

  /* ── Divider ── */
  hr { border-color: #21262d; }
</style>

<!-- Deploy button -->
<button class="deploy-btn">Deploy</button>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Chat"
if "pipeline_status" not in st.session_state:
    st.session_state.pipeline_status = {
        "Dataset Validation": "done",
        "Missing Value Analysis": "done",
        "Feature Engineering": "done",
        "Hyperparameter Search": "running",
        "Model Training": "pending",
    }

STATUS_ICON = {"done": "✓", "running": "⚡", "pending": "⏳"}

# ── Hero banner ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <div class="hero-title">🤖 AutoML Agent Pro 🔗</div>
  <p class="hero-sub">Upload data, describe your goal, and automatically discover the best ML pipeline.</p>
</div>
""", unsafe_allow_html=True)

# ── Three-column layout ───────────────────────────────────────────────────────
left, center, right = st.columns([2.2, 4.5, 2.3], gap="large")

# ── LEFT: Dataset ─────────────────────────────────────────────────────────────
with left:
    st.markdown('<div class="section-title">📁 Dataset</div>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.85rem;color:#8b949e;margin-bottom:0.4rem;">📊 Upload CSV File</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        label_visibility="collapsed",
    )
    st.markdown('<p class="upload-hint">200MB per file • CSV</p>', unsafe_allow_html=True)

    if uploaded_file:
        st.success(f"✓ {uploaded_file.name} loaded")
        # auto-mark dataset validation done
        st.session_state.pipeline_status["Dataset Validation"] = "done"

# ── CENTER: Chat / Dashboard / Code tabs ──────────────────────────────────────
with center:
    # Tab strip
    tabs = ["Chat", "Dashboard", "Code"]
    tab_icons = {"Chat": "💬", "Dashboard": "📊", "Code": "💻"}

    tab_html = '<div class="tab-strip">'
    for t in tabs:
        active_cls = "active" if st.session_state.active_tab == t else ""
        tab_html += f'<div class="tab-item {active_cls}">{tab_icons[t]} {t}</div>'
    tab_html += '</div>'
    st.markdown(tab_html, unsafe_allow_html=True)

    # Actual clickable tabs using Streamlit
    tab_cols = st.columns(3)
    for i, t in enumerate(tabs):
        with tab_cols[i]:
            if st.button(t, key=f"tab_{t}"):
                st.session_state.active_tab = t
                st.rerun()

    st.markdown("<hr style='margin: 0.2rem 0 1rem 0;'>", unsafe_allow_html=True)

    # ── CHAT TAB ──
    if st.session_state.active_tab == "Chat":
        # Objective prompt box
        st.markdown("""
        <div class="objective-box">
          <div class="obj-icon">🤖</div>
          <span>Describe your ML objective. Example: <em>Predict customer churn.</em></span>
        </div>
        """, unsafe_allow_html=True)

        # Render chat history
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-msg-user"><div class="chat-sender">You</div>{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-msg-ai"><div class="chat-sender">AutoML Agent</div>{msg["content"]}</div>', unsafe_allow_html=True)

        # Input row
        col_input, col_send = st.columns([9, 1])
        with col_input:
            user_input = st.text_input("Ask", placeholder="Ask about your dataset...", key="chat_input")
        with col_send:
            st.markdown("<div style='margin-top:1.65rem;'>", unsafe_allow_html=True)
            send = st.button("↑", key="send_btn")
            st.markdown("</div>", unsafe_allow_html=True)

        if send and user_input.strip():
            st.session_state.messages.append({"role": "user", "content": user_input})
            # Simulate agent response
            with st.spinner("Agent thinking…"):
                time.sleep(0.8)
            response = (
                f"Got it! I'll analyze your dataset to **{user_input.lower()}**. "
                "I'll start with exploratory data analysis, handle missing values, "
                "engineer relevant features, then run hyperparameter search across "
                "Random Forest, XGBoost, and LightGBM. Results will appear in the Dashboard tab."
            )
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

    # ── DASHBOARD TAB ──
    elif st.session_state.active_tab == "Dashboard":
        if not uploaded_file:
            st.info("Upload a CSV file to see dashboard metrics.")
        else:
            import pandas as pd, io
            df = pd.read_csv(uploaded_file)
            m1, m2, m3 = st.columns(3)
            m1.metric("Rows", f"{len(df):,}")
            m2.metric("Columns", len(df.columns))
            m3.metric("Missing %", f"{df.isnull().mean().mean()*100:.1f}%")
            st.markdown("**Column Types**")
            type_counts = df.dtypes.value_counts().rename_axis("dtype").reset_index(name="count")
            type_counts["dtype"] = type_counts["dtype"].astype(str)
            st.dataframe(type_counts, use_container_width=True)
            st.markdown("**Preview**")
            st.dataframe(df.head(10), use_container_width=True)

    # ── CODE TAB ──
    elif st.session_state.active_tab == "Code":
        st.markdown("**Generated ML Pipeline Code**")
        sample_code = '''import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load data
df = pd.read_csv("your_dataset.csv")

# Feature engineering
X = df.drop("target", axis=1)
y = df["target"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train best model (RandomForest — tuned)
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_split=5,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
print(classification_report(y_test, model.predict(X_test)))
'''
        st.code(sample_code, language="python")

# ── RIGHT: Insights ───────────────────────────────────────────────────────────
with right:
    st.markdown('<div class="section-title">🧠 Insights</div>', unsafe_allow_html=True)

    items_html = ""
    for step, status in st.session_state.pipeline_status.items():
        icon = STATUS_ICON[status]
        color = "#2ecc71" if status == "done" else ("#e67e22" if status == "running" else "#6e7681")
        items_html += f'<div class="pipeline-item" style="color:{color};">{icon} {step}</div>'

    st.markdown(f"""
    <div class="insights-panel">
      <div class="pipeline-title">Pipeline Status</div>
      {items_html}
    </div>
    """, unsafe_allow_html=True)

    # Extra: model leaderboard placeholder
    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
    if st.session_state.messages:
        st.markdown("""
        <div class="insights-panel" style="margin-top:1rem;">
          <div class="pipeline-title">Model Leaderboard</div>
          <div class="pipeline-item" style="color:#2ecc71;">🥇 XGBoost — 91.4% AUC</div>
          <div class="pipeline-item" style="color:#8b949e;">🥈 LightGBM — 90.8% AUC</div>
          <div class="pipeline-item" style="color:#8b949e;">🥉 RandomForest — 89.2% AUC</div>
        </div>
        """, unsafe_allow_html=True)