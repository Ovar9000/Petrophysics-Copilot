"""Convenient launcher for the Hybrid Well Log RAG & Petrophysics Application.
"""

import subprocess
import sys
import time
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PYTHON_EXE = BASE_DIR / ".venv" / "Scripts" / "python.exe"
STREAMLIT_EXE = BASE_DIR / ".venv" / "Scripts" / "streamlit.exe"


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("==================================================================")
    print(" Hybrid Agentic Well Log RAG & Petrophysical Analytics Platform  ")
    print("==================================================================")
    
    # 1. Initialize catalog
    print("\n[1/3] Initializing well catalog & geological metadata...")
    subprocess.run([str(PYTHON_EXE), "-c", "from backend.catalog import init_catalog; init_catalog(); print('Catalog initialized.')"], cwd=str(BASE_DIR), check=True)
    
    # 2. Launch FastAPI backend
    print("\n[2/3] Starting FastAPI Backend on http://127.0.0.1:8000 ...")
    backend_proc = subprocess.Popen(
        [str(PYTHON_EXE), "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=str(BASE_DIR)
    )
    
    time.sleep(2)
    
    # 3. Launch React Dashboard
    print("\n[3/3] Starting Modern React Studio Dashboard on http://localhost:3000 ...")
    npx_cmd = "npx.cmd" if os.name == "nt" else "npx"
    frontend_proc = subprocess.Popen(
        [npx_cmd, "vite", "--port", "3000", "--host"],
        cwd=str(BASE_DIR / "frontend-react")
    )
    
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("\n==================================================================")
    print(" Subsurface Petrophysical Studio is Live!")
    print(" - Modern React Dashboard: http://localhost:3000")
    print(" - FastAPI Swagger API Docs: http://127.0.0.1:8000/docs")
    print(" - Docker Compose Setup: run 'docker compose up --build'")
    print("==================================================================")
    print("\nPress Ctrl+C to terminate services.")
    
    try:
        backend_proc.wait()
        frontend_proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down servers...")
        backend_proc.terminate()
        frontend_proc.terminate()


if __name__ == "__main__":
    main()
