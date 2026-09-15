"""
embeddings.py
-------------
Local embedding engine using SentenceTransformers with single-load caching.
Includes a lightweight local fallback to guarantee robust operation.
"""

import numpy as np
from typing import List, Union
from rag.config import DEFAULT_EMBEDDING_MODEL

_EMBEDDING_MODEL_CACHE = None

class LocalEmbeddingEngine:
    """
    Wrapper for SentenceTransformers model with singleton caching.
    """
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.model_name = model_name
        self.model = None
        self.fallback = False
        self._init_model()

    def _init_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            print(f"Loading embedding model '{self.model_name}'...")
            self.model = SentenceTransformer(self.model_name)
            print("[OK] SentenceTransformer embedding model loaded successfully.")
        except Exception as e:
            print(f"[NOTE] SentenceTransformer fallback ({e}). Using deterministic HashingVectorizer engine.")
            self.fallback = True
            from sklearn.feature_extraction.text import HashingVectorizer
            self.vectorizer = HashingVectorizer(n_features=384, stop_words="english", alternate_sign=False)

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        if not self.fallback and self.model is not None:
            embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return embeddings / norms
        else:
            embeddings = self.vectorizer.transform(texts).toarray()
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return embeddings / norms


def get_embedding_engine(model_name: str = DEFAULT_EMBEDDING_MODEL) -> LocalEmbeddingEngine:
    """
    Singleton getter for embedding engine.
    """
    global _EMBEDDING_MODEL_CACHE
    if _EMBEDDING_MODEL_CACHE is None:
        _EMBEDDING_MODEL_CACHE = LocalEmbeddingEngine(model_name)
    return _EMBEDDING_MODEL_CACHE
