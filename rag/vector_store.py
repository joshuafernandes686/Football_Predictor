"""
vector_store.py
---------------
Local, disk-persisted vector store supporting embedding storage,
metadata filtering (by team/tournament/doc_type), similarity search, and disk persistence.
"""

import os
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from rag.config import VECTOR_STORE_DIR
from rag.document_builder import FootballDocument


class LocalVectorStore:
    """
    Disk-persisted local vector database for semantic search and metadata filtering.
    """
    def __init__(self, store_dir: Path = VECTOR_STORE_DIR):
        self.store_dir = store_dir
        self.doc_ids: List[str] = []
        self.documents: List[FootballDocument] = []
        self.embeddings: Optional[np.ndarray] = None

    def save(self):
        """Persists embeddings matrix and metadata payload to disk."""
        self.store_dir.mkdir(parents=True, exist_ok=True)
        index_file = self.store_dir / "index.npz"
        docs_file = self.store_dir / "documents.pkl"

        if self.embeddings is not None and len(self.doc_ids) > 0:
            np.savez_compressed(index_file, embeddings=self.embeddings, doc_ids=np.array(self.doc_ids))
            
            # Serialize documents list
            doc_data = [doc.to_dict() for doc in self.documents]
            with open(docs_file, "wb") as f:
                pickle.dump(doc_data, f)
            print(f"[OK] Vector store persisted ({len(self.doc_ids)} docs) to {self.store_dir}")

    def load(self) -> bool:
        """Loads index and metadata from disk if available."""
        index_file = self.store_dir / "index.npz"
        docs_file = self.store_dir / "documents.pkl"

        if not index_file.exists() or not docs_file.exists():
            return False

        try:
            npz = np.load(index_file)
            self.embeddings = npz["embeddings"]
            self.doc_ids = list(npz["doc_ids"])

            with open(docs_file, "rb") as f:
                doc_dicts = pickle.load(f)

            self.documents = [
                FootballDocument(d["doc_id"], d["content"], d["metadata"])
                for d in doc_dicts
            ]
            print(f"[OK] Vector store loaded successfully ({len(self.doc_ids)} docs).")
            return True
        except Exception as e:
            print(f"[NOTE] Error loading vector store: {e}")
            return False

    def add_documents(self, documents: List[FootballDocument], embeddings: np.ndarray):
        """Adds documents and their embedding matrix to the store."""
        new_ids = [doc.doc_id for doc in documents]
        self.doc_ids.extend(new_ids)
        self.documents.extend(documents)

        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        team_filter: Optional[str] = None,
        opponent_filter: Optional[str] = None,
    ) -> List[Tuple[FootballDocument, float]]:
        """
        Performs cosine similarity search with optional team and opponent metadata filtering.
        """
        if self.embeddings is None or len(self.documents) == 0:
            return []

        # Cosine similarity for normalized vectors: dot product
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        sim_scores = np.dot(self.embeddings, query_embedding.T).flatten()

        # Apply metadata boosting/filtering if team_filter or opponent_filter is set
        adjusted_scores = sim_scores.copy()
        if team_filter:
            team_filter_lower = team_filter.lower().strip()
            for i, doc in enumerate(self.documents):
                doc_team = doc.metadata.get("team", "")
                doc_opp = doc.metadata.get("opponent", "")
                if doc_team == team_filter_lower or doc_opp == team_filter_lower:
                    adjusted_scores[i] += 0.35  # Boost relevant team documents

        if opponent_filter:
            opp_filter_lower = opponent_filter.lower().strip()
            for i, doc in enumerate(self.documents):
                doc_team = doc.metadata.get("team", "")
                doc_opp = doc.metadata.get("opponent", "")
                if (doc_team == team_filter_lower and doc_opp == opp_filter_lower) or \
                   (doc_team == opp_filter_lower and doc_opp == team_filter_lower):
                    adjusted_scores[i] += 0.40  # Extra boost for H2H match records

        # Sort indices by adjusted similarity score descending
        top_indices = np.argsort(adjusted_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            raw_score = float(sim_scores[idx])
            results.append((self.documents[idx], raw_score))

        return results
