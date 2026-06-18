#!/usr/bin/env python
"""
Streamlit App Launcher
Properly sets up the environment and runs the app
"""
import os
import sys
import subprocess

# Set working directory to the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Run streamlit with the app
app_path = os.path.join(script_dir, "Streamlit", "automl_studio.py")

# Run streamlit
result = subprocess.run(
    [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        app_path,
        "--server.port=8501",
        "--logger.level=info"
    ],
    env={**os.environ, "PYTHONPATH": script_dir}
)

sys.exit(result.returncode)
