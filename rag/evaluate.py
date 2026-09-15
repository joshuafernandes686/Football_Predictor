"""
evaluate.py
-----------
Evaluation script for the RAG system.
Executes representative test queries, printing similarity scores, retrieved context,
source attribution, and LLM generated answers.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.config import VECTOR_STORE_DIR
from rag.vector_store import LocalVectorStore
from rag.rag_pipeline import RAGPipeline


def evaluate_rag():
    print("=" * 70)
    print("RAG SYSTEM EVALUATION & DEMONSTRATION SCRIPT")
    print("=" * 70)

    # Load vector store
    vector_store = LocalVectorStore(VECTOR_STORE_DIR)
    if not vector_store.load():
        print("Vector store not found. Running ingestion first...")
        from rag.ingestion import run_ingestion
        vector_store = run_ingestion()

    pipeline = RAGPipeline(vector_store)

    test_queries = [
        {
            "question": "Compare Argentina's results against Brazil, England, Germany, France, Italy and Spain.",
            "home": "Argentina",
            "away": None,
        },
        {
            "question": "How has Argentina performed against historically strong footballing nations such as England, Brazil, Germany, France, Italy and Spain, and does the available data suggest that Argentina performs differently against stronger opposition?",
            "home": "Argentina",
            "away": None,
        },
        {"question": "How has France performed recently?", "home": "France", "away": None},
        {"question": "What is Spain's historical record against France?", "home": "Spain", "away": "France"},
        {"question": "Who is injured for France today?", "home": "France", "away": None},
    ]

    for i, item in enumerate(test_queries, 1):
        q = item["question"]
        home = item["home"]
        away = item["away"]

        print("\n" + "-" * 70)
        print(f"QUERY {i}: \"{q}\"")
        if home or away:
            print(f"Context Filter: Home='{home}', Away='{away}'")
        print("-" * 70)

        result = pipeline.run(question=q, home_team=home, away_team=away, top_k=3)

        print("\n[RETRIEVED DOCUMENTS & SIMILARITY SCORES]")
        if result["retrieved_documents"]:
            for doc_text, score in zip(result["retrieved_documents"], result["similarity_scores"]):
                print(f"  - Score {score:.4f} | {doc_text}")
        else:
            print("  (No documents retrieved / Out of scope query)")

        print("\n[GENERATED ANSWER]")
        print(result["answer"])

        print("\n[SOURCES ATTRIBUTION]")
        for src in result["sources"]:
            print(f"  - {src['team'].title()} vs {src['opponent'].title()} | {src['tournament']} ({src['date']}) [{src['document_type']}]")

    print("\n" + "=" * 70)
    print("RAG EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    evaluate_rag()
