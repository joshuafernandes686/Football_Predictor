"""
config.py
---------
Configuration settings for the FootballPredictor RAG system.
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "Data" / "processed"
RESULTS_FILE = DATA_DIR / "results_clean.csv"
ELO_FILE = DATA_DIR / "elo_clean.csv"
FEATURES_FILE = DATA_DIR / "features.csv"

RAG_DIR = PROJECT_ROOT / "rag"
VECTOR_STORE_DIR = RAG_DIR / "vector_store_data"

# Embedding Model Configuration
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Retrieval & Vector Store Configuration
TOP_K_RETRIEVAL = 5
SIMILARITY_THRESHOLD = 0.25

# LLM Configuration
DEFAULT_LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
