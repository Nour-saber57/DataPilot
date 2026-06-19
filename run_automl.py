#!/usr/bin/env python
"""
Startup script for AutoML Studio
Runs FastAPI backend and Streamlit frontend together
"""

import subprocess
import time
import sys
import os

# Change to Data_Pilot directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

print("=" * 70)
print("🚀 AUTOML STUDIO - STARTUP")
print("=" * 70)

# Check if venv exists
venv_path = os.path.join(script_dir, 'venv', 'Scripts', 'python.exe')
if not os.path.exists(venv_path):
    print("❌ Virtual environment not found!")
    print(f"Expected: {venv_path}")
    sys.exit(1)

print("\n✓ Virtual environment found")

# Start FastAPI backend
print("\n" + "=" * 70)
print("1️⃣  Starting FastAPI Backend on http://localhost:8000")
print("=" * 70)

backend_process = subprocess.Popen(
    [venv_path, "-m", "uvicorn", "FastAPI.test:app", "--reload", "--port", "8000"],
    cwd=script_dir,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

print("✓ FastAPI backend starting...")
time.sleep(3)

# Start Streamlit frontend
print("\n" + "=" * 70)
print("2️⃣  Starting Streamlit Frontend on http://localhost:8501")
print("=" * 70)

frontend_process = subprocess.Popen(
    [venv_path, "-m", "streamlit", "run", "Streamlit/automl_studio.py", "--server.port", "8501"],
    cwd=script_dir,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

print("✓ Streamlit frontend starting...")
time.sleep(2)

print("\n" + "=" * 70)
print("✅ AUTOML STUDIO STARTED!")
print("=" * 70)
print("""
📊 Open your browser and go to:
   Frontend: http://localhost:8501
   Backend API Docs: http://localhost:8000/docs

🛑 To stop, press Ctrl+C

Press Ctrl+C to stop all services...
""")

try:
    # Wait for processes
    backend_process.wait()
    frontend_process.wait()
except KeyboardInterrupt:
    print("\n\n🛑 Shutting down services...")
    backend_process.terminate()
    frontend_process.terminate()
    print("✓ Services stopped")
    sys.exit(0)
