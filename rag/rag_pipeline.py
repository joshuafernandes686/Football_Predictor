"""
rag_pipeline.py
---------------
Master RAG Pipeline connecting query parsing, semantic document retrieval,
prompt grounding, hallucination prevention rules, and LLM generation.
"""

from typing import Dict, List, Any, Optional
from rag.document_builder import FootballDocument
from rag.retrieval import SemanticRetriever
from rag.vector_store import LocalVectorStore
from rag.llm_provider import get_llm_provider, LLMProvider


SYSTEM_GROUNDING_PROMPT = """
You are FootballPredictor AI, a factual and analytical football intelligence assistant.
Your job is to answer user questions strictly using the retrieved context provided below.

Strict Analytical & Grounding Guidelines:
1. Synthesize comparative questions across ALL named opponents in the retrieved documents rather than reciting isolated match snippets.
2. Group retrieved matches by opponent and summarize Head-to-Head aggregates: Matches Played, Wins, Draws, Losses, Goals Scored, and Goals Conceded.
3. Explicitly answer the user's analytical question based on the calculated evidence.
4. Distinguish between (a) retrieved raw facts, (b) calculated match aggregates, and (c) analytical inferences.
5. Do NOT invent a definition of "strong opposition". Clarify that "historically strong nations" refers to the specific teams named by the user rather than an automated rating threshold.
6. Do NOT invent statistics, scores, or fake historical events. Use ONLY retrieved documents.
7. If data for any named opponent is sparse or missing, state clearly that dataset context is limited for that team.
8. Do NOT alter or claim to change any ML prediction probabilities.
"""


class RAGPipeline:
    """
    Master RAG orchestration pipeline.
    """
    def __init__(self, vector_store: LocalVectorStore, llm_provider: Optional[LLMProvider] = None):
        self.vector_store = vector_store
        self.retriever = SemanticRetriever(vector_store)
        self.llm_provider = llm_provider or get_llm_provider()

    def run(
        self,
        question: str,
        home_team: Optional[str] = None,
        away_team: Optional[str] = None,
        prediction_context: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Executes the end-to-end RAG pipeline for a user question.
        """
        # Out-of-scope question check for live injuries / live news
        q_lower = question.lower()
        if any(term in q_lower for term in ["injury", "injured", "lineup", "starting 11", "transfer news", "suspended", "manager rumor"]):
            return {
                "answer": "The current FootballPredictor knowledge base contains historical FIFA match records and Elo statistics. It does not contain live injury, lineup, or transfer news data.",
                "sources": [],
                "retrieved_documents": [],
                "similarity_scores": [],
            }

        # 1. Semantic Retrieval
        retrieved_results = self.retriever.retrieve(
            question=question,
            home_team=home_team,
            away_team=away_team,
            top_k=top_k,
        )

        docs = [doc for doc, score in retrieved_results]
        scores = [score for doc, score in retrieved_results]

        # 2. Build Context String
        context_blocks = []
        for idx, doc in enumerate(docs, 1):
            context_blocks.append(f"[Document {idx}]: {doc.content}")

        context_str = "\n\n".join(context_blocks) if context_blocks else "No relevant context found in database."

        # Add optional prediction context if provided
        match_context_str = ""
        if prediction_context and "winner" in prediction_context:
            h = prediction_context.get("home_team", home_team or "Home")
            a = prediction_context.get("away_team", away_team or "Away")
            w = prediction_context.get("winner", "")
            probs = prediction_context.get("probabilities", {})
            match_context_str = (
                f"\nMatch ML Model Prediction Context:\n"
                f"- Match: {h} vs {a}\n"
                f"- Model Predicted Winner: {w}\n"
                f"- Probabilities: Home ({probs.get('home', 0)*100:.1f}%), Draw ({probs.get('draw', 0)*100:.1f}%), Away ({probs.get('away', 0)*100:.1f}%)\n"
            )

        # 3. Construct Prompt
        user_prompt = (
            f"User Question: {question}\n\n"
            f"{match_context_str}\n"
            f"Retrieved Historical Knowledge Context:\n"
            f"{context_str}\n\n"
            f"Please answer the user question factually using the retrieved context above."
        )

        # 4. LLM Generation
        answer = self.llm_provider.generate_response(
            prompt=user_prompt,
            retrieved_documents=docs,
            system_instruction=SYSTEM_GROUNDING_PROMPT,
        )

        # 5. Extract Source Metadata
        sources = []
        seen_keys = set()
        for doc in docs:
            src = {
                "document_type": doc.metadata.get("document_type", "record"),
                "team": doc.metadata.get("display_team", doc.metadata.get("team", "")),
                "opponent": doc.metadata.get("display_opponent", doc.metadata.get("opponent", "")),
                "date": doc.metadata.get("date", ""),
                "tournament": doc.metadata.get("tournament", "FIFA Match Dataset"),
                "source_dataset": doc.metadata.get("source_dataset", "results_clean.csv"),
            }
            key = (src["team"], src["opponent"], src["date"], src["document_type"])
            if key not in seen_keys:
                seen_keys.add(key)
                sources.append(src)

        return {
            "answer": answer,
            "sources": sources,
            "retrieved_documents": [doc.content for doc in docs],
            "similarity_scores": [round(float(s), 4) for s in scores],
        }
