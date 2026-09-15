# ⚽ FootballPredictor: Machine Learning Match Predictor & RAG Q&A System

An end-to-end production software engineering project combining a **Machine Learning Classifier with Elo & Form Feature Engineering** and a **Grounded Retrieval-Augmented Generation (RAG) System** with a React frontend and FastAPI backend.

---

## 🏛️ System Architecture

```
                    ┌────────────────────────────────────────────────────────┐
                    │                    FOOTBALLPREDICTOR                   │
                    └───────────────────────────┬────────────────────────────┘
                                                │
                 ┌──────────────────────────────┴──────────────────────────────┐
                 ▼                                                             ▼
  ┌─────────────────────────────┐                               ┌─────────────────────────────┐
  │     ML PREDICTION ENGINE    │                               │       RAG Q&A ENGINE        │
  ├─────────────────────────────┤                               ├─────────────────────────────┤
  │ Structured FIFA CSV Data    │                               │ Structured FIFA CSV Data    │
  │            │                │                               │            │                │
  │            ▼                │                               │            ▼                │
  │ Feature Engineering (Elo/   │                               │ Document Generator          │
  │ Form/H2H/Streaks)           │                               │ (17,000+ NL Match &         │
  │            │                │                               │ Team Summaries)             │
  │            ▼                │                               │            │                │
  │ ML Model (best_model.pkl)   │                               │            ▼                │
  │            │                │                               │ Local Embedding Engine      │
  │            ▼                │                               │ (all-MiniLM-L6-v2)          │
  │ Winner & Class Probabilities│                               │            │                │
  │ + SHAP Explanations         │                               │            ▼                │
  └──────────────┬──────────────┘                               │ Persisted Vector Store      │
                 │                                              │ (rag/vector_store_data/)    │
                 │                                              │            │                │
                 │                                              │            ▼                │
                 │                                              │ Semantic Retriever with     │
                 │                                              │ Team Metadata Filters       │
                 │                                              │            │                │
                 │                                              │            ▼                │
                 │                                              │ Grounded LLM Provider       │
                 │                                              │ (OpenAI / Gemini / Local)   │
                 └──────────────────────────────┬───────────────┴──────────────┬──────────────┘
                                                │                              │
                                                ▼                              ▼
                                      ┌──────────────────────────────────────────────────┐
                                      │         FastAPI Backend (src/api.py)             │
                                      │   GET /teams  |  POST /predict  | POST /rag/chat │
                                      └─────────────────────────┬────────────────────────┘
                                                                │
                                                                ▼
                                      ┌──────────────────────────────────────────────────┐
                                      │        React Frontend App (frontend/)            │
                                      │      Match Selector & Interactive Chatbot UI     │
                                      └──────────────────────────────────────────────────┘
```

---

## 🧠 Key RAG System Features

1. **Structured Data to Natural Language Conversion**: Automatically parses historical FIFA match results (`results_clean.csv`), Elo ratings (`elo_clean.csv`), and rolling form stats (`features.csv`) into 17,000+ natural language documents with rich metadata (`team`, `opponent`, `date`, `tournament`, `document_type`).
2. **Local Vector Database Persistence**: Stores vector embeddings in `rag/vector_store_data/` supporting cosine similarity search and metadata filtering.
3. **Strict Grounding & Hallucination Prevention**: Prompts explicitly forbid inventing statistics, scores, or fake events. Requests for out-of-scope data (such as live injuries or transfer news) yield honest dataset boundary responses.
4. **LLM Provider Abstraction**:
   - `OpenAIProvider`: Uses OpenAI API if `OPENAI_API_KEY` is present.
   - `GeminiProvider`: Uses Google Gemini API if `GEMINI_API_KEY` is present.
   - `LocalMockProvider`: Offline fallback synthesizing answers directly from retrieved documents without requiring API keys.
5. **Extensible Ingestion Architecture**: Abstract `BaseIngestionSource` interface (`CSVIngestionSource`) allows future live news/RSS/web ingestion (`WebArticleIngestionSource`) to be plugged in without modifying retriever or vector database logic.

---

## 🛠️ Quick Start Guide

### 1. Build / Update the RAG Knowledge Base
Generate natural language documents, compute embeddings, and build the persistent vector store:
```bash
python rag/ingestion.py
```

### 2. Run the RAG System Evaluation Script
Test semantic retrieval, similarity scores, metadata filtering, and answer synthesis:
```bash
python rag/evaluate.py
```

### 3. Start the FastAPI Backend Server
```bash
python main.py
```
*(Runs on `http://127.0.0.1:8000` with interactive docs at `http://127.0.0.1:8000/docs`)*

### 4. Start the React Frontend UI
In a separate terminal:
```bash
npm run dev
```
*(Runs on `http://localhost:5173/`)*

---

## 📡 API Endpoints

- **`GET /teams`**: Returns sorted JSON list of all 195 valid international teams in the dataset.
- **`POST /predict`**:
  - *Payload*: `{ "home_team": "Spain", "away_team": "France" }`
  - *Response*: Predicted winner, probabilities, natural language reasons, and SHAP feature importances.
- **`POST /rag/chat`**:
  - *Payload*: `{ "question": "What is Spain's historical record against France?", "home_team": "Spain", "away_team": "France" }`
  - *Response*: Grounded answer, source document metadata (`team`, `opponent`, `date`, `tournament`), similarity scores, and retrieved document snippets.

---

## 💡 Example Chatbot Queries

- *"How has France performed recently?"*
- *"What is Spain's historical record against France?"*
- *"Tell me about Germany's recent results."*
- *"What tournaments are represented in the dataset?"*
- *"Who is injured for France today?"* *(Demonstrates out-of-scope handling)*

---

## 🔮 Future Architecture Plan for Live Football News / RSS Ingestion

The RAG system is designed with a pluggable ingestion pipeline:
```
Football News / RSS / Web
          │
          ▼
WebArticleIngestionSource (inherits from BaseIngestionSource)
          │
          ▼
  Same Chunking / Embedding / Local Vector Store / Retriever
```
This allows adding live injury updates, manager press conferences, and squad announcements without redesigning the retrieval system or ML prediction engine.
