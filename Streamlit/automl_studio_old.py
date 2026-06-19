"""
AutoML Frontend - Streamlit App
Connects to FastAPI backend for ML pipeline
"""

import streamlit as st
import requests
import pandas as pd
import json
from io import StringIO
import os
import sys

# Ensure proper module path resolution
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Page configuration
st.set_page_config(
    page_title="AutoML Studio",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Poppins', sans-serif; }

.metric-card {
    background: linear-gradient(135deg, rgba(16, 163, 127, 0.1), rgba(16, 163, 127, 0.05));
    border: 1px solid rgba(16, 163, 127, 0.3);
    border-radius: 12px;
    padding: 20px;
    margin: 10px 0;
}

.success-box {
    background: rgba(16, 163, 127, 0.2);
    border-left: 4px solid #10a37f;
    padding: 15px;
    border-radius: 8px;
}

.error-box {
    background: rgba(220, 20, 60, 0.2);
    border-left: 4px solid #dc143c;
    padding: 15px;
    border-radius: 8px;
}

</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'target_column' not in st.session_state:
    st.session_state.target_column = None
if 'available_models' not in st.session_state:
    st.session_state.available_models = []

# Main title
st.title("🤖 AutoML Studio")
st.markdown("*End-to-End Machine Learning Pipeline with Automatic Model Selection & Hyperparameter Tuning*")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    api_status = requests.get(f"{API_BASE_URL}/").json() if requests.get(f"{API_BASE_URL}/", timeout=2).status_code == 200 else None
    if api_status:
        st.success("✓ Backend Connected")
    else:
        st.error("✗ Backend Not Connected")
        st.info("Make sure to run: `python -m uvicorn FastAPI.test:app --reload --port 8000`")
    
    test_size = st.slider("Test Size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)

# Main layout
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data", "🧠 Models", "📈 Results", "📋 Details"])

# ==================== TAB 1: DATA ====================
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📁 Upload Dataset")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            # Show preview
            df_preview = pd.read_csv(uploaded_file)
            st.info(f"**Dataset Shape:** {df_preview.shape[0]} rows × {df_preview.shape[1]} columns")
            
            with st.expander("👁️ Data Preview"):
                st.dataframe(df_preview.head(10))
            
            # Select target column
            st.markdown("---")
            target_col = st.selectbox("Select Target Column", df_preview.columns)
            
            # Upload to backend
            if st.button("📤 Upload & Analyze", key="upload_btn"):
                with st.spinner("Uploading data to backend..."):
                    try:
                        files = {
                             "file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")
                             }
                        params = {"target_column": target_col}
                        response = requests.post(
                             f"{API_BASE_URL}/upload-data",
                             files=files,
                             params=params
                             )
                        result = response.json()
                        if not response.ok:
                            detail = result.get("detail", result)
                            if isinstance(detail, dict):
                                st.error(f"Error: {detail.get('message', detail)}")
                            else:
                                st.error(f"Error: {detail}")
                                st.stop()
                        if result.get("status") == "success":
                            st.session_state.data_loaded = True
                            st.session_state.target_column = target_col
                            
                            st.success("✓ Data loaded successfully!")
                            
                            # Display analysis
                            st.markdown("### 📊 Data Analysis")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Rows", result['data_shape'][0])
                            with col2:
                                st.metric("Columns", result['data_shape'][1])
                            with col3:
                                st.metric("Task", result['task'].upper())
                            
                            # Feature analysis
                            st.markdown("### 🔍 Feature Analysis")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.info(f"**Numeric Features:** {len(result['numeric_columns'])}")
                                st.write(", ".join(result['numeric_columns']))
                            
                            with col2:
                                st.info(f"**Categorical Features:** {len(result['categorical_columns'])}")
                                st.write(", ".join(result['categorical_columns']))
                            
                            # Missing values
                            missing = result['missing_values']
                            if any(missing.values()):
                                st.warning("⚠️ Missing values detected")
                                st.dataframe(pd.DataFrame({
                                    'Column': list(missing.keys()),
                                    'Missing': list(missing.values())
                                }))
                        else:
                            st.error(f"Error: {result.get('message', 'Upload failed')}")
                    
                    except Exception as e:
                        st.error(f"Connection error: {str(e)}")
    
    with col2:
        st.subheader("📋 Summary")
        st.info("**Steps:**\n1. Upload CSV\n2. Select target\n3. Click 'Upload & Analyze'")

# ==================== TAB 2: MODELS ====================
with tab2:
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload data first in the 'Data' tab")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🚀 Model Training")
            
            # Get available models
            if st.button("🔄 Refresh Available Models"):
                with st.spinner("Loading available models..."):
                    try:
                        response = requests.get(f"{API_BASE_URL}/available-models")
                        result = response.json()
                        st.session_state.available_models = result['models']
                        st.success(f"✓ Found {len(result['models'])} models")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            
            if st.session_state.available_models:
                st.info(f"**Available Models ({len(st.session_state.available_models)}):**")
                for i, model in enumerate(st.session_state.available_models, 1):
                    st.write(f"{i}. {model}")
        
        with col2:
            st.subheader("⚡ Quick Actions")
            
            train_option = st.radio(
                "Training Mode",
                ["Train Single", "Train All"],
                horizontal=False
            )
        
        st.markdown("---")
        
        # Training
        if train_option == "Train Single":
            st.subheader("🎯 Train Single Model")
            
            if st.session_state.available_models:
                selected_model = st.selectbox("Select Model", st.session_state.available_models)
                
                if st.button("🚀 Train Model", key="train_single"):
                    with st.spinner(f"Training {selected_model}..."):
                        try:
                            response = requests.post(
                                f"{API_BASE_URL}/train-model",
                                params={
                                    "model_name": selected_model,
                                    "target_column": st.session_state.target_column,
                                    "test_size": test_size
                                }
                            )
                            
                            result = response.json()
                            
                            if result['status'] == 'success':
                                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                                st.success(f"✓ Training Complete: {selected_model}")
                                st.markdown('</div>', unsafe_allow_html=True)
                                
                                # Display metrics
                                st.markdown("### 📊 Performance Metrics")
                                
                                metrics = result['metrics']
                                cols = st.columns(len(metrics))
                                
                                for i, (metric_name, metric_value) in enumerate(metrics.items()):
                                    with cols[i]:
                                        st.metric(
                                            metric_name.upper(),
                                            f"{metric_value:.4f}"
                                        )
                                
                                # Best parameters
                                st.markdown("### ⚙️ Best Hyperparameters")
                                st.json(result['best_params'])
                                
                                st.info(f"Train: {result['train_samples']} | Test: {result['test_samples']}")
                            else:
                                st.error(f"Error: {result.get('message', 'Training failed')}")
                        
                        except Exception as e:
                            st.error(f"Connection error: {str(e)}")
        
        else:  # Train All
            st.subheader("⚡ Train All Models & Compare")
            
            if st.button("🚀 Train All Models", key="train_all"):
                with st.spinner("Training all models... This may take a few minutes"):
                    try:
                        response = requests.post(
                            f"{API_BASE_URL}/train-all-models",
                            params={
                                "target_column": st.session_state.target_column,
                                "test_size": test_size
                            }
                        )
                        
                        result = response.json()
                        
                        if result['status'] == 'success':
                            st.markdown('<div class="success-box">', unsafe_allow_html=True)
                            st.success("✓ Training Complete!")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Summary
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Models Trained", result['successful'])
                            with col2:
                                st.metric("Models Failed", result['failed'])
                            with col3:
                                st.metric("Total", result['total_models'])
                            
                            # Leaderboard
                            st.markdown("### 🏆 Model Leaderboard")
                            
                            leaderboard_data = []
                            for rank, item in enumerate(result['leaderboard'], 1):
                                leaderboard_data.append({
                                    'Rank': rank,
                                    'Model': item['model'],
                                    'Score': f"{item['score']:.4f}"
                                })
                            
                            st.dataframe(
                                pd.DataFrame(leaderboard_data),
                                use_container_width=True,
                                hide_index=True
                            )
                        else:
                            st.error(f"Error: {result['message']}")
                    
                    except Exception as e:
                        st.error(f"Connection error: {str(e)}")

# ==================== TAB 3: RESULTS ====================
with tab3:
    st.subheader("📈 Training Results")
    
    if st.button("🔄 Refresh Results"):
        with st.spinner("Loading results..."):
            try:
                response = requests.get(f"{API_BASE_URL}/results")
                result = response.json()
                
                if result['status'] == 'success' and result['trained_models']:
                    for model_name in result['trained_models']:
                        model_result = result['results'][model_name]
                        
                        with st.expander(f"📊 {model_name}"):
                            if model_result['status'] == 'success':
                                # Metrics
                                st.markdown("**Performance Metrics:**")
                                metrics_df = pd.DataFrame({
                                    'Metric': list(model_result['metrics'].keys()),
                                    'Value': [f"{v:.4f}" for v in model_result['metrics'].values()]
                                })
                                st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                                
                                # Parameters
                                st.markdown("**Best Hyperparameters:**")
                                st.json(model_result['best_params'])
                            else:
                                st.error(f"Error: {model_result['message']}")
                else:
                    st.info("No training results yet. Train models first!")
            
            except Exception as e:
                st.error(f"Connection error: {str(e)}")

# ==================== TAB 4: DETAILS ====================
with tab4:
    st.subheader("📋 Backend Information")
    
    if st.button("ℹ️ Get API Status"):
        try:
            response = requests.get(f"{API_BASE_URL}/")
            data = response.json()
            
            st.json({
                "status": data['status'],
                "message": data['message'],
                "version": data['version']
            })
        except Exception as e:
            st.error(f"Connection error: {str(e)}")
    
    st.markdown("---")
    st.markdown("""
    ### 🔧 API Endpoints
    
    - `POST /upload-data` - Upload CSV and analyze
    - `GET /available-models` - Get available models for task
    - `POST /train-model` - Train single model with GridSearchCV
    - `POST /train-all-models` - Train all models and compare
    - `GET /results` - Get training results
    
    ### 📚 Documentation
    
    Visit `http://localhost:8000/docs` for interactive API documentation (Swagger UI)
    """)
