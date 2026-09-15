"""
api.py
------
FastAPI Backend Server for Football Predictor application.

Endpoints:
- GET /teams : Returns sorted JSON list of all valid teams.
- POST /predict : Accepts { "home_team": "Spain", "away_team": "France" } and returns predictions, probabilities, reasons, and SHAP data.
- POST /rag/chat : RAG Q&A chatbot endpoint retrieving context from FIFA match datasets.
- GET /health : Simple health status endpoint.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure src directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
RAG_DIR = PROJECT_ROOT / "rag"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from predict import _get_runtime_context, get_available_teams, predict_match
from rag.config import VECTOR_STORE_DIR
from rag.vector_store import LocalVectorStore
from rag.rag_pipeline import RAGPipeline

app = FastAPI(
    title="Football Match Predictor API",
    description="API for predicting international football match outcomes using machine learning and RAG Q&A.",
    version="1.1.0",
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_RAG_PIPELINE: Optional[RAGPipeline] = None


def _get_rag_pipeline() -> RAGPipeline:
    """Lazy singleton loader for the RAG pipeline."""
    global _RAG_PIPELINE
    if _RAG_PIPELINE is None:
        vector_store = LocalVectorStore(VECTOR_STORE_DIR)
        if not vector_store.load():
            print("[RAG] Index not found on startup. Running ingestion...")
            from rag.ingestion import run_ingestion
            vector_store = run_ingestion()
        _RAG_PIPELINE = RAGPipeline(vector_store)
    return _RAG_PIPELINE


class PredictRequest(BaseModel):
    home_team: str = Field(..., example="Spain", description="Home Team Name")
    away_team: str = Field(..., example="France", description="Away Team Name")
    tournament: Optional[str] = Field("Friendly", description="Tournament Context")
    neutral: Optional[bool] = Field(False, description="Whether match is played on neutral ground")


class RagChatRequest(BaseModel):
    question: str = Field(..., example="How has France performed recently?", description="User question")
    home_team: Optional[str] = Field(None, example="France", description="Optional Home Team context")
    away_team: Optional[str] = Field(None, example="Spain", description="Optional Away Team context")
    prediction_context: Optional[Dict[str, Any]] = Field(None, description="Optional ML prediction results context")


@app.on_event("startup")
def startup_event():
    """Pre-load ML engine and RAG pipeline into memory on server start."""
    print("Initializing Football Predictor backend engine...")
    _get_runtime_context()
    print("[OK] Backend ML model & trackers loaded successfully.")
    _get_rag_pipeline()
    print("[OK] RAG engine initialized successfully.")


@app.get("/health")
def health_check():
    return {"status": "ok", "app": "Football Predictor API with RAG"}


@app.get("/teams", response_model=List[str])
def get_teams():
    """Return all valid teams available in the dataset, sorted alphabetically."""
    try:
        return get_available_teams()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving teams: {str(e)}")


@app.post("/predict")
def predict(req: PredictRequest):
    """Predict match outcome, probabilities, natural language explanations, and SHAP feature importances."""
    home = req.home_team.strip()
    away = req.away_team.strip()

    if not home or not away:
        raise HTTPException(status_code=400, detail="Please select both Home and Away teams.")

    if home.lower() == away.lower():
        raise HTTPException(status_code=400, detail="Home Team and Away Team must be different.")

    valid_teams = get_available_teams()
    valid_lower = {team.lower(): team for team in valid_teams}

    if home.lower() not in valid_lower:
        raise HTTPException(status_code=400, detail=f"Team '{home}' is not available in the dataset.")
    if away.lower() not in valid_lower:
        raise HTTPException(status_code=400, detail=f"Team '{away}' is not available in the dataset.")

    canonical_home = valid_lower[home.lower()]
    canonical_away = valid_lower[away.lower()]

    try:
        prediction_result = predict_match(
            canonical_home,
            canonical_away,
            tournament=req.tournament or "Friendly",
            neutral=bool(req.neutral),
        )
        return prediction_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/rag/chat")
def rag_chat(req: RagChatRequest):
    """
    RAG Q&A Endpoint: Retrieves relevant historical FIFA documents,
    grounds context, and generates a natural language answer with source attributions.
    """
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        rag = _get_rag_pipeline()
        response = rag.run(
            question=question,
            home_team=req.home_team,
            away_team=req.away_team,
            prediction_context=req.prediction_context,
            top_k=5,
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG Q&A error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
