"""
ingestion.py
------------
Ingestion pipeline for building and updating the RAG vector store index.
Designed with an extensible ingestion interface (CSVIngestionSource)
to easily accommodate future web/RSS/news sources without refactoring.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from abc import ABC, abstractmethod
from typing import List
from rag.document_builder import FootballDocument, build_all_documents
from rag.embeddings import get_embedding_engine
from rag.vector_store import LocalVectorStore
from rag.config import VECTOR_STORE_DIR


class BaseIngestionSource(ABC):
    """
    Base interface for RAG document ingestion sources.
    Future sources (e.g. WebArticleIngestionSource, RSSIngestionSource) will inherit from this class.
    """
    @abstractmethod
    def fetch_documents(self) -> List[FootballDocument]:
        pass


class CSVIngestionSource(BaseIngestionSource):
    """
    Ingestion source converting processed CSV datasets into natural language documents.
    """
    def fetch_documents(self) -> List[FootballDocument]:
        return build_all_documents()


class WebArticleIngestionSource(BaseIngestionSource):
    """
    Placeholder interface for future live news/web article ingestion.
    """
    def fetch_documents(self) -> List[FootballDocument]:
        print("[Future Feature] Web article ingestion source placeholder.")
        return []


def run_ingestion(rebuild: bool = True) -> LocalVectorStore:
    """
    Master ingestion workflow: Loads documents, embeds them, and persists vector store.
    """
    print("=" * 60)
    print("FOOTBALL PREDICTOR RAG INGESTION PIPELINE")
    print("=" * 60)

    # 1. Fetch documents from active ingestion sources
    csv_source = CSVIngestionSource()
    documents = csv_source.fetch_documents()

    if not documents:
        print("[WARNING] No documents generated for ingestion.")
        return LocalVectorStore()

    # 2. Extract contents and compute embeddings
    print(f"\nComputing embeddings for {len(documents)} documents...")
    embedding_engine = get_embedding_engine()
    contents = [doc.content for doc in documents]
    embeddings = embedding_engine.encode(contents)

    # 3. Populate vector store and persist
    print("\nPopulating local vector store...")
    vector_store = LocalVectorStore(VECTOR_STORE_DIR)
    vector_store.add_documents(documents, embeddings)
    vector_store.save()

    print("\n" + "=" * 60)
    print("RAG INGESTION COMPLETE")
    print("=" * 60)

    return vector_store


if __name__ == "__main__":
    run_ingestion()
