# FootballPredictor

An end-to-end football match prediction system combining machine learning, feature engineering, RAG-based football data retrieval, FastAPI, and React.

## Overview

FootballPredictor predicts the outcome of international football matches using historical match data, Elo ratings, recent form, and other engineered features.

The project also includes a RAG-based question-answering system that allows users to query the historical football data through a chatbot interface.

The application is divided into two main components:

- ML prediction pipeline
- RAG question-answering pipeline

Both are exposed through a FastAPI backend and integrated into a React frontend.

## System Architecture

```text
                         FootballPredictor
                                |
                +---------------+---------------+
                |                               |
                v                               v
        ML Prediction Engine              RAG Q&A Engine
                |                               |
        Historical FIFA Data             Historical FIFA Data
                |                               |
        Feature Engineering             Document Generation
        (Elo, Form, H2H, etc.)                 |
                |                         Embeddings
                v                               |
          Trained ML Model                      v
                |                         Vector Store
                |                               |
                |                          Retrieval
                |                               |
                |                         LLM Provider
                |                               |
                +---------------+---------------+
                                |
                                v
                         FastAPI Backend
                                |
                                v
                         React Frontend
