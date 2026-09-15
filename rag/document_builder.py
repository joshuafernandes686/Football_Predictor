"""
document_builder.py
-------------------
Deterministic document builder that converts structured CSV datasets (results_clean.csv,
elo_clean.csv, features.csv) into meaningful, natural-language football documents
enriched with structured metadata.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from rag.config import RESULTS_FILE, ELO_FILE, FEATURES_FILE


class FootballDocument:
    """
    Represents a natural language document for the RAG knowledge base.
    """
    def __init__(self, doc_id: str, content: str, metadata: Dict[str, Any]):
        self.doc_id = doc_id
        self.content = content
        self.metadata = metadata

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "content": self.content,
            "metadata": self.metadata,
        }


def build_match_documents(results_df: pd.DataFrame) -> List[FootballDocument]:
    """
    Converts each match row into natural language match records for both home and away teams.
    """
    docs = []
    for idx, row in results_df.iterrows():
        match_date = str(row["date"])[:10]
        home_team = str(row["home_team"]).replace("\xa0", " ").strip()
        away_team = str(row["away_team"]).replace("\xa0", " ").strip()
        home_score = int(row["home_score"]) if pd.notnull(row["home_score"]) else 0
        away_score = int(row["away_score"]) if pd.notnull(row["away_score"]) else 0
        tournament = str(row["tournament"])
        city = str(row.get("city", ""))
        country = str(row.get("country", ""))
        neutral = bool(row.get("neutral", False))

        if neutral:
            venue_info = f"played on neutral ground in {city}, {country}" if city and country else "played on neutral ground"
        else:
            venue_info = f"with {home_team} as designated home team in {city}, {country}" if city and country else f"with {home_team} as designated home team"

        if home_score > away_score:
            outcome = f"{home_team} defeated {away_team} {home_score}-{away_score}"
            home_result_type = "Win"
            away_result_type = "Loss"
        elif away_score > home_score:
            outcome = f"{away_team} defeated {home_team} {away_score}-{home_score}"
            home_result_type = "Loss"
            away_result_type = "Win"
        else:
            outcome = f"{home_team} and {away_team} drew {home_score}-{away_score}"
            home_result_type = "Draw"
            away_result_type = "Draw"

        # Home team perspective
        home_content = (
            f"Match Record: On {match_date}, {home_team} played {away_team} "
            f"in {tournament} ({venue_info}). "
            f"Result: {outcome}. {home_team} scored {home_score} goals and conceded {away_score} goals."
        )
        home_metadata = {
            "team": home_team.lower(),
            "opponent": away_team.lower(),
            "display_team": home_team,
            "display_opponent": away_team,
            "date": match_date,
            "tournament": tournament,
            "result_type": home_result_type,
            "goals_for": home_score,
            "goals_against": away_score,
            "document_type": "match_record",
            "source_dataset": "results_clean.csv",
        }
        docs.append(FootballDocument(f"match_{idx}_home", home_content, home_metadata))

        # Away team perspective
        away_content = (
            f"Match Record: On {match_date}, {away_team} played {home_team} "
            f"in {tournament} ({venue_info}). "
            f"Result: {outcome}. {away_team} scored {away_score} goals and conceded {home_score} goals."
        )
        away_metadata = {
            "team": away_team.lower(),
            "opponent": home_team.lower(),
            "display_team": away_team,
            "display_opponent": home_team,
            "date": match_date,
            "tournament": tournament,
            "result_type": away_result_type,
            "goals_for": away_score,
            "goals_against": home_score,
            "document_type": "match_record",
            "source_dataset": "results_clean.csv",
        }
        docs.append(FootballDocument(f"match_{idx}_away", away_content, away_metadata))

    return docs


def build_team_summary_documents(features_df: pd.DataFrame) -> List[FootballDocument]:
    """
    Generates rolling statistical performance summary documents for each team across major matches.
    """
    docs = []
    # Take latest 5 matches per team to capture rolling form & state
    df_sorted = features_df.sort_values("date")

    team_records: Dict[str, List[pd.Series]] = {}
    for idx, row in df_sorted.iterrows():
        home = str(row["home_team"]).replace("\xa0", " ").strip()
        away = str(row["away_team"]).replace("\xa0", " ").strip()
        if home not in team_records:
            team_records[home] = []
        if away not in team_records:
            team_records[away] = []
        team_records[home].append(row)
        team_records[away].append(row)

    for team, rows in team_records.items():
        if not rows:
            continue
        latest_row = rows[-1]
        match_date = str(latest_row["date"])[:10]
        is_home = (str(latest_row["home_team"]).replace("\xa0", " ").strip() == team)

        elo = float(latest_row["home_elo"] if is_home else latest_row["away_elo"]) if pd.notnull(latest_row["home_elo"] if is_home else latest_row["away_elo"]) else 1500.0
        form = float(latest_row["home_form_last5"] if is_home else latest_row["away_form_last5"])
        goals_last5 = float(latest_row["home_goals_last5"] if is_home else latest_row["away_goals_last5"])
        conceded_last5 = float(latest_row["home_conceded_last5"] if is_home else latest_row["away_conceded_last5"])
        gd_last5 = float(latest_row["home_goal_difference_last5"] if is_home else latest_row["away_goal_difference_last5"])
        win_pct = float(latest_row["home_win_percentage_last5"] if is_home else latest_row["away_win_percentage_last5"]) * 100
        clean_sheets = float(latest_row["home_clean_sheets_last5"] if is_home else latest_row["away_clean_sheets_last5"])
        failed_score = float(latest_row["home_failed_to_score_last5"] if is_home else latest_row["away_failed_to_score_last5"])
        winning_streak = float(latest_row["home_winning_streak"] if is_home else latest_row["away_winning_streak"])
        unbeaten_streak = float(latest_row["home_unbeaten_streak"] if is_home else latest_row["away_unbeaten_streak"])
        opp_elo = float(latest_row["home_avg_opponent_elo_last5"] if is_home else latest_row["away_avg_opponent_elo_last5"])

        content = (
            f"Team Performance Summary for {team} as of {match_date}: "
            f"Elo Rating: {elo:.1f}. Recent 5-match form: {form:.0f} points out of 15 (win percentage: {win_pct:.1f}%). "
            f"Goals scored in last 5 matches: {goals_last5:.0f}, goals conceded: {conceded_last5:.0f} "
            f"(goal difference: {gd_last5:+.0f}). Clean sheets kept: {clean_sheets:.0f}. "
            f"Matches failed to score: {failed_score:.0f}. "
            f"Current winning streak: {winning_streak:.0f} matches. Current unbeaten streak: {unbeaten_streak:.0f} matches. "
            f"Average opponent Elo in recent matches: {opp_elo:.1f}."
        )

        metadata = {
            "team": team.lower(),
            "display_team": team,
            "date": match_date,
            "tournament": str(latest_row["tournament"]),
            "document_type": "team_summary",
            "source_dataset": "features.csv",
        }
        docs.append(FootballDocument(f"summary_{team.lower()}", content, metadata))

    return docs


def build_h2h_summary_documents(results_df: pd.DataFrame) -> List[FootballDocument]:
    """
    Builds head-to-head rivalry summary documents for pairs of teams that played 2 or more matches.
    """
    docs = []
    h2h_pairs: Dict[Tuple[str, str], List[pd.Series]] = {}

    for idx, row in results_df.iterrows():
        t1 = str(row["home_team"]).replace("\xa0", " ").strip()
        t2 = str(row["away_team"]).replace("\xa0", " ").strip()
        pair_key = (t1, t2) if t1 < t2 else (t2, t1)
        if pair_key not in h2h_pairs:
            h2h_pairs[pair_key] = []
        h2h_pairs[pair_key].append(row)

    for (t1, t2), matches in h2h_pairs.items():
        if len(matches) < 2:
            continue

        t1_wins = 0
        t2_wins = 0
        draws = 0
        t1_goals = 0
        t2_goals = 0

        for m in matches:
            home = str(m["home_team"]).replace("\xa0", " ").strip()
            h_score = int(m["home_score"]) if pd.notnull(m["home_score"]) else 0
            a_score = int(m["away_score"]) if pd.notnull(m["away_score"]) else 0

            if home == t1:
                t1_goals += h_score
                t2_goals += a_score
                if h_score > a_score:
                    t1_wins += 1
                elif a_score > h_score:
                    t2_wins += 1
                else:
                    draws += 1
            else:
                t2_goals += h_score
                t1_goals += a_score
                if h_score > a_score:
                    t2_wins += 1
                elif a_score > h_score:
                    t1_wins += 1
                else:
                    draws += 1

        last_match = matches[-1]
        last_date = str(last_match["date"])[:10]
        last_tournament = str(last_match["tournament"])

        content = (
            f"Head-to-Head History between {t1} and {t2}: "
            f"Total encounters played: {len(matches)}. "
            f"{t1} victories: {t1_wins}, {t2} victories: {t2_wins}, Draws: {draws}. "
            f"Total goals scored: {t1} {t1_goals}, {t2} {t2_goals}. "
            f"Most recent match date: {last_date} in {last_tournament}."
        )

        metadata_1 = {
            "team": t1.lower(),
            "opponent": t2.lower(),
            "display_team": t1,
            "display_opponent": t2,
            "date": last_date,
            "tournament": last_tournament,
            "document_type": "h2h_summary",
            "source_dataset": "results_clean.csv",
        }
        docs.append(FootballDocument(f"h2h_{t1.lower()}_{t2.lower()}", content, metadata_1))

        metadata_2 = {
            "team": t2.lower(),
            "opponent": t1.lower(),
            "display_team": t2,
            "display_opponent": t1,
            "date": last_date,
            "tournament": last_tournament,
            "document_type": "h2h_summary",
            "source_dataset": "results_clean.csv",
        }
        docs.append(FootballDocument(f"h2h_{t2.lower()}_{t1.lower()}", content, metadata_2))

    return docs


def build_all_documents() -> List[FootballDocument]:
    """
    Master pipeline loading cleaned datasets and generating all RAG knowledge documents.
    """
    print("Building natural language documents from FIFA datasets...")
    results_df = pd.read_csv(RESULTS_FILE)
    features_df = pd.read_csv(FEATURES_FILE)

    match_docs = build_match_documents(results_df)
    summary_docs = build_team_summary_documents(features_df)
    h2h_docs = build_h2h_summary_documents(results_df)

    all_docs = match_docs + summary_docs + h2h_docs
    print(f"[OK] Generated {len(all_docs)} total documents:")
    print(f"  - Match Records: {len(match_docs)}")
    print(f"  - Team Summaries: {len(summary_docs)}")
    print(f"  - Head-to-Head Summaries: {len(h2h_docs)}")

    return all_docs
