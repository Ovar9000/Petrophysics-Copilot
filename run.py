"""Convenient launcher for the Hybrid Well Log RAG & Petrophysics Application.
"""

import shutil
import socket
import subprocess
import sys
import time
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PYTHON_EXE = BASE_DIR / ".venv" / "Scripts" / "python.exe"
if not PYTHON_EXE.exists():
    # Fall back to the interpreter running this script (e.g. system python)
    PYTHON_EXE = Path(sys.executable)

sys.path.insert(0, str(BASE_DIR))
from backend.config import BACKEND_HOST, BACKEND_PORT, FRONTEND_PORT


def is_port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def resolve_frontend_cmd():
    """Return argv to launch Vite, avoiding broken npx shims on PATH.

    Prefers the local vite install, then a real Node.js npx.
    """
    local_vite = BASE_DIR / "frontend-react" / "node_modules" / ".bin" / (
        "vite.cmd" if os.name == "nt" else "vite"
    )
    if local_vite.exists():
        return [str(local_vite), "--port", str(FRONTEND_PORT), "--host"]
    # Avoid PATH shadowing (e.g. Goose bin/npx.cmd); prefer Node's own dir.
    candidates = []
    for name in (["npx.cmd", "npx"] if os.name == "nt" else ["npx"]):
        found = shutil.which(name)
        if found:
            candidates.append(found)
    for hard in [
        r"C:\nvm4w\nodejs\npx.cmd",
        r"C:\Program Files\nodejs\npx.cmd",
        r"C:\Program Files (x86)\nodejs\npx.cmd",
    ]:
        if os.path.exists(hard) and hard not in candidates:
            candidates.append(hard)
    # Drop known-bad shims (Goose) unless nothing else exists.
    real = [c for c in candidates if "Goose" not in c]
    npx_cmd = (real or candidates or ["npx.cmd" if os.name == "nt" else "npx"])[0]
    return [npx_cmd, "vite", "--port", str(FRONTEND_PORT), "--host"]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("==================================================================")
    print(" Hybrid Agentic Well Log RAG & Petrophysical Analytics Platform  ")
    print("==================================================================")
    
    # 1. Initialize catalog
    print("\n[1/3] Initializing well catalog & geological metadata...")
    subprocess.run([str(PYTHON_EXE), "-c", "from backend.catalog import init_catalog; init_catalog(); print('Catalog initialized.')"], cwd=str(BASE_DIR), check=True)
    
    # 2. Launch FastAPI backend (reuse if already running)
    backend_proc = None
    if is_port_in_use(BACKEND_HOST, BACKEND_PORT):
        print(f"\n[2/3] Backend port {BACKEND_PORT} already in use - reusing existing server.")
    else:
        print(f"\n[2/3] Starting FastAPI Backend on http://{BACKEND_HOST}:{BACKEND_PORT} ...")
        backend_proc = subprocess.Popen(
            [str(PYTHON_EXE), "-m", "uvicorn", "backend.main:app", "--host", BACKEND_HOST, "--port", str(BACKEND_PORT)],
            cwd=str(BASE_DIR)
        )
        time.sleep(2)

    # 3. Launch React Dashboard (reuse if already running)
    print(f"\n[3/3] Starting Modern React Studio Dashboard on http://localhost:{FRONTEND_PORT} ...")
    frontend_proc = None
    if is_port_in_use("127.0.0.1", FRONTEND_PORT):
        print(f"Frontend port {FRONTEND_PORT} already in use - reusing existing server.")
    else:
        frontend_cmd = resolve_frontend_cmd()
        print(f"Using frontend launcher: {' '.join(frontend_cmd)}")
        frontend_proc = subprocess.Popen(
            frontend_cmd,
            cwd=str(BASE_DIR / "frontend-react")
        )
    
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("\n==================================================================")
    print(" Subsurface Petrophysical Studio is Live!")
    print(f" - Modern React Dashboard: http://localhost:{FRONTEND_PORT}")
    print(f" - FastAPI Swagger API Docs: http://{BACKEND_HOST}:{BACKEND_PORT}/docs")
    print("==================================================================")
    print("\nPress Ctrl+C to terminate services.")
    
    try:
        procs = [p for p in (backend_proc, frontend_proc) if p is not None]
        if not procs:
            print("\nBoth services already running. Nothing to do - open the URLs above.")
            return
        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        print("\nShutting down servers...")
        for p in (backend_proc, frontend_proc):
            if p is not None:
                try:
                    p.terminate()
                except Exception:
                    pass


if __name__ == "__main__":
    main()
