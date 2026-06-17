
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="AutoML Agent Pro", page_icon="🤖", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

* {
    font-family: 'Poppins', sans-serif;
}

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0b0f14 0%, #111820 100%);
    color: white;
}

.hero {
    background: linear-gradient(135deg, rgba(16, 163, 127, 0.25), rgba(16, 163, 127, 0.08));
    padding: 40px;
    border-radius: 28px;
    border: 1px solid rgba(16, 163, 127, 0.3);
    margin-bottom: 32px;
    box-shadow: 0 8px 24px rgba(16, 163, 127, 0.1);
    transition: all 0.3s ease;
}

.hero:hover {
    border-color: rgba(16, 163, 127, 0.5);
    box-shadow: 0 12px 32px rgba(16, 163, 127, 0.15);
}

.hero h1 {
    font-size: 2.5em;
    font-weight: 800;
    margin-bottom: 16px;
    background: linear-gradient(135deg, #10a37f, #00d9a3);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero p {
    font-size: 1.1em;
    color: rgba(255, 255, 255, 0.85);
    line-height: 1.6;
    margin: 0;
}

.card {
    background: linear-gradient(135deg, rgba(19, 26, 34, 0.9), rgba(25, 35, 45, 0.9));
    border: 1px solid rgba(255, 255, 255, 0.12);
    padding: 28px;
    border-radius: 24px;
    margin-bottom: 24px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
    transition: all 0.3s ease;
    backdrop-filter: blur(10px);
}

.card:hover {
    border-color: rgba(16, 163, 127, 0.4);
    box-shadow: 0 8px 24px rgba(16, 163, 127, 0.15);
}

.card h4 {
    font-size: 1.3em;
    font-weight: 700;
    margin-bottom: 18px;
    color: #10a37f;
}

.metric {
    background: linear-gradient(135deg, rgba(19, 26, 34, 0.95), rgba(25, 35, 45, 0.95));
    border-radius: 24px;
    padding: 28px;
    text-align: center;
    border: 1px solid rgba(16, 163, 127, 0.2);
    margin-bottom: 0;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
    transition: all 0.3s ease;
    backdrop-filter: blur(10px);
}

.metric:hover {
    border-color: rgba(16, 163, 127, 0.5);
    box-shadow: 0 8px 24px rgba(16, 163, 127, 0.15);
    transform: translateY(-4px);
}

.metric h2 {
    font-size: 2.2em;
    font-weight: 800;
    margin: 0 0 12px 0;
    color: #10a37f;
}

.metric p {
    font-size: 0.95em;
    color: rgba(255, 255, 255, 0.75);
    margin: 0;
    font-weight: 500;
}

[data-testid="stTabBar"] {
    margin-bottom: 24px;
}

[data-testid="stSubheader"] {
    margin-top: 28px;
    margin-bottom: 20px;
}

[data-testid="stFileUploadDropzone"] {
    margin-bottom: 24px;
}

h3 {
    font-size: 1.4em;
    font-weight: 700;
    margin-bottom: 20px;
    color: rgba(255, 255, 255, 0.95);
}

button {
    font-family: 'Poppins', sans-serif !important;
    font-weight: 500 !important;
    font-size: 1.05em !important;
    color: rgba(255, 255, 255, 0.9) !important;
    letter-spacing: 0.3px;
}

[data-testid="stTabs"] {
    margin-bottom: 24px;
}

[data-testid="stTabs"] [data-testid="stTabBar"] {
    display: flex;
    gap: 12px;
}

[data-testid="stTabs"] [role="tablist"] button {
    flex: 1;
    min-width: 100px;
}

</style>
""", unsafe_allow_html=True)

if "df" not in st.session_state:
    st.session_state.df = None

# Hero Section at Top
st.markdown("""
<div class='hero'>
<h1>🤖 AutoML Agent Pro</h1>
<p>Upload data, describe your goal, and automatically discover the best ML pipeline.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='margin-bottom: 28px;'></div>", unsafe_allow_html=True)

left, center, right = st.columns([1.1,2.2,1.1], gap="large")

with left:
    st.markdown("### 📂 Dataset")
    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)
    file = st.file_uploader("📤 Upload CSV File", type=["csv"], key="main_upload")

    if file:
        df = pd.read_csv(file)
        st.session_state.df = df

    st.markdown("<div style='margin-bottom: 28px;'></div>", unsafe_allow_html=True)

    if st.session_state.df is not None:
        df = st.session_state.df
        rows, cols = df.shape
        st.markdown(f"""
        <div class='card'>
        <h4>Dataset Overview</h4>
        <b>Rows:</b> {rows:,}<br>
        <b>Columns:</b> {cols}<br>
        <b>Missing:</b> {df.isna().sum().sum()}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='margin-bottom: 28px;'></div>", unsafe_allow_html=True)

        target = st.selectbox("Target", df.columns)
        
        st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)
        
        task = st.selectbox(
            "Task",
            ["Auto Detect","Classification","Regression","Clustering"]
        )

        st.markdown("<div style='margin-bottom: 28px;'></div>", unsafe_allow_html=True)

        with st.expander("⚙️ Advanced Settings"):
            st.slider("Time Budget",30,600,120)
            st.markdown("<div style='margin-bottom: 16px;'></div>", unsafe_allow_html=True)
            st.selectbox("Validation Split",["80/20","70/30","90/10"])

        st.markdown("<div style='margin-bottom: 28px;'></div>", unsafe_allow_html=True)

        run = st.button("🚀 Run AutoML", use_container_width=True, key="run_button")

with center:

    tabs = st.tabs(["💬 Chat","📊 Dashboard","💻 Code"])

    with tabs[0]:
        st.chat_message("assistant").write(
            "Describe your ML objective. Example: Predict customer churn."
        )
        st.markdown("<div style='margin-bottom: 16px;'></div>", unsafe_allow_html=True)
        st.chat_input("Ask about your dataset...")

    with tabs[1]:
        if st.session_state.df is not None:
            c1,c2,c3,c4 = st.columns(4, gap="medium")

            for c,label,val in [
                (c1,"Accuracy","94.2%"),
                (c2,"F1","0.941"),
                (c3,"AUC","0.978"),
                (c4,"Models","12")
            ]:
                with c:
                    st.markdown(
                        f"<div class='metric'><h2>{val}</h2><p>{label}</p></div>",
                        unsafe_allow_html=True
                    )

            st.markdown("<div style='margin-bottom: 28px;'></div>", unsafe_allow_html=True)

            st.subheader("🏆 Model Leaderboard")

            leaderboard = pd.DataFrame({
                "Model":["Random Forest","XGBoost","LightGBM","CatBoost","SVM"],
                "Score":[0.942,0.939,0.936,0.928,0.917]
            })

            st.dataframe(leaderboard, use_container_width=True)

            st.markdown("<div style='margin-bottom: 28px;'></div>", unsafe_allow_html=True)

            st.subheader("📈 Feature Importance")

            feats = st.session_state.df.columns[:min(8,len(st.session_state.df.columns))]
            imp = np.random.dirichlet(np.ones(len(feats))) * 100

            fig = px.bar(
                x=imp,
                y=feats,
                orientation="h",
                title="Feature Importance"
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("<div style='margin-bottom: 28px;'></div>", unsafe_allow_html=True)

            st.subheader("📊 Dataset Health")

            health = 92

            fig2 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=health,
                title={"text":"Health Score"},
            ))
            st.plotly_chart(fig2, use_container_width=True)

    with tabs[2]:
        st.code("""
# Generated AutoML Pipeline

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

model.fit(X_train,y_train)
pred = model.predict(X_test)
        """, language="python")

with right:
    st.markdown("### 🧠 Insights")
    
    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>
    <h4>Pipeline Status</h4>
    ✓ Dataset Validation<br>
    ✓ Missing Value Analysis<br>
    ✓ Feature Engineering<br>
    ⚡ Hyperparameter Search<br>
    ⏳ Model Training
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>
    <h4>AI Suggestions</h4>
    • Remove high-cardinality columns<br>
    • Check target imbalance<br>
    • Consider feature scaling<br>
    • Review outliers
    </div>
    """, unsafe_allow_html=True)
