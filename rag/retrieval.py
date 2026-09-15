"""
retrieval.py
------------
Semantic retrieval engine supporting team metadata filtering and similarity scoring.
"""

import re
from typing import List, Tuple, Optional
from rag.document_builder import FootballDocument
from rag.embeddings import get_embedding_engine
from rag.vector_store import LocalVectorStore


class SemanticRetriever:
    """
    Retrieval module bridging user queries and vector store search.
    """
    def __init__(self, vector_store: LocalVectorStore):
        self.vector_store = vector_store
        self.embedding_engine = get_embedding_engine()

    def _extract_all_teams_from_query(self, query: str) -> List[str]:
        """
        Extracts all international team names mentioned in the query.
        """
        query_lower = query.lower()
        # Dynamically retrieve available teams if possible, or use comprehensive list
        try:
            from predict import get_available_teams
            available_teams = get_available_teams()
        except Exception:
            available_teams = [
                "Argentina", "Brazil", "England", "Germany", "France", "Italy", "Spain",
                "Portugal", "Netherlands", "Uruguay", "Belgium", "Croatia", "Morocco",
                "Switzerland", "Japan", "South Korea", "Mexico", "Colombia", "Chile"
            ]

        found = []
        for team in available_teams:
            pattern = r'\b' + re.escape(team.lower()) + r'\b'
            if re.search(pattern, query_lower):
                found.append(team)

        # Preserve order of appearance in query
        found.sort(key=lambda t: query_lower.find(t.lower()))
        return found

    def retrieve(
        self,
        question: str,
        home_team: Optional[str] = None,
        away_team: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Tuple[FootballDocument, float]]:
        """
        Retrieves relevant documents for a given user question.
        Dynamically scales top_k and runs multi-pass grouped retrieval for multi-team comparisons.
        """
        query_vec = self.embedding_engine.encode(question)
        teams_in_query = self._extract_all_teams_from_query(question)

        # Primary focus team
        target_home = home_team if home_team else (teams_in_query[0] if teams_in_query else None)

        # Determine opponent teams
        opponents = []
        if away_team:
            opponents.append(away_team)
        elif len(teams_in_query) > 1:
            opponents = [t for t in teams_in_query if t.lower() != target_home.lower()]

        # Multi-opponent comparison retrieval logic
        if target_home and len(opponents) >= 2:
            print(f"[RAG Retrieval] Detected comparative analytical query for {target_home} vs {len(opponents)} opponents: {opponents}")
            combined_results: List[Tuple[FootballDocument, float]] = []
            seen_doc_ids = set()

            # Pass 1: Fetch Head-to-Head & Match records for EACH named opponent
            for opp in opponents:
                opp_results = self.vector_store.search(
                    query_embedding=query_vec,
                    top_k=4,
                    team_filter=target_home.lower(),
                    opponent_filter=opp.lower(),
                )
                for doc, score in opp_results:
                    if doc.doc_id not in seen_doc_ids:
                        seen_doc_ids.add(doc.doc_id)
                        combined_results.append((doc, score))

            # Pass 2: Fetch overall team summary for main team
            summary_results = self.vector_store.search(
                query_embedding=query_vec,
                top_k=2,
                team_filter=target_home.lower(),
            )
            for doc, score in summary_results:
                if doc.doc_id not in seen_doc_ids:
                    seen_doc_ids.add(doc.doc_id)
                    combined_results.append((doc, score))

            # Sort combined results by similarity score descending
            combined_results.sort(key=lambda item: item[1], reverse=True)
            return combined_results

        # Single team or pair retrieval
        target_away = opponents[0] if opponents else None
        effective_top_k = max(top_k, 12) if len(teams_in_query) > 1 else top_k

        results = self.vector_store.search(
            query_embedding=query_vec,
            top_k=effective_top_k,
            team_filter=target_home.lower() if target_home else None,
            opponent_filter=target_away.lower() if target_away else None,
        )

        return results
