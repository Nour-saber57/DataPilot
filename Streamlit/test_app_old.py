import streamlit as st

st.title("Test App")
st.write("If you see this, Streamlit is working!")
st.write(f"API Backend Status: Testing...")

import requests
try:
    response = requests.get("http://localhost:8000/")
    st.success("✓ Backend API is running!")
except:
    st.error("✗ Backend API not reachable")
