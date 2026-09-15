"""
main.py
-------
Root entry point to start the Football Predictor FastAPI backend server.

Usage:
    python main.py
"""

import sys
import uvicorn
from pathlib import Path

# Add src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if __name__ == "__main__":
    print("Starting Football Predictor FastAPI Server on http://127.0.0.1:8000...")
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
