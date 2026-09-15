"""Convenient launcher for the Hybrid Well Log RAG & Petrophysics Application.
"""

import subprocess
import sys
import time
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PYTHON_EXE = BASE_DIR / ".venv" / "Scripts" / "python.exe"

sys.path.insert(0, str(BASE_DIR))
from backend.config import BACKEND_HOST, BACKEND_PORT, FRONTEND_PORT


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
    print(f"\n[2/3] Starting FastAPI Backend on http://{BACKEND_HOST}:{BACKEND_PORT} ...")
    backend_proc = subprocess.Popen(
        [str(PYTHON_EXE), "-m", "uvicorn", "backend.main:app", "--host", BACKEND_HOST, "--port", str(BACKEND_PORT)],
        cwd=str(BASE_DIR)
    )

    time.sleep(2)

    # 3. Launch React Dashboard
    print(f"\n[3/3] Starting Modern React Studio Dashboard on http://localhost:{FRONTEND_PORT} ...")
    npx_cmd = "npx.cmd" if os.name == "nt" else "npx"
    frontend_proc = subprocess.Popen(
        [npx_cmd, "vite", "--port", str(FRONTEND_PORT), "--host"],
        cwd=str(BASE_DIR / "frontend-react")
    )
    
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("\n==================================================================")
    print(" Subsurface Petrophysical Studio is Live!")
    print(f" - Modern React Dashboard: http://localhost:{FRONTEND_PORT}")
    print(f" - FastAPI Swagger API Docs: http://{BACKEND_HOST}:{BACKEND_PORT}/docs")
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
