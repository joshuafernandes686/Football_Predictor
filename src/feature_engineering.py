"""
feature_engineering.py
----------------------
Production-grade feature engineering pipeline for Football Match Predictor.

Key Software Engineering Principles Implemented:
1. Performance Optimization: O(N) single-pass chronological state tracking replacing O(N^2) dataset scanning.
2. Binary Search Elo Lookup: O(log M) lookup via EloTracker using pre-sorted team rating indexes.
3. Chronological Integrity: Strict temporal evaluation preventing data leakage.
4. Rich Feature Engineering: Adds H2H, streaks, rest days, clean sheets, failed-to-score, opponent strength, Elo trends, and vector differences.
5. Reusable Modular Design: Exportable trackers for real-time inference in predict.py.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from bisect import bisect_right
from typing import Dict, List, Tuple, Optional

# ==========================================================
# Configuration & Constants
# ==========================================================

PROCESSED_DATA = Path("Data/processed")
RESULTS_FILE = PROCESSED_DATA / "results_clean.csv"
ELO_FILE = PROCESSED_DATA / "elo_clean.csv"
FEATURES_FILE = PROCESSED_DATA / "features.csv"

DEFAULT_REST_DAYS = 30  # Default rest days when no prior match exists
DEFAULT_LOOKBACK_WINDOW = 5  # Rolling window size for form statistics

TOURNAMENT_MAPPING = {
    "Friendly": 0,
    "FIFA World Cup": 1,
    "FIFA World Cup qualification": 2,
    "UEFA Euro": 3,
    "UEFA Euro qualification": 4,
    "UEFA Nations League": 5,
    "Copa América": 6,
    "Copa América qualification": 7,
    "AFC Asian Cup": 8,
    "AFC Asian Cup qualification": 9,
    "African Cup of Nations": 10,
    "African Cup of Nations qualification": 11,
    "CONCACAF Gold Cup": 12,
    "CONCACAF Nations League": 13,
    "Oceania Nations Cup": 14,
}

# ==========================================================
# Load Data
# ==========================================================

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load preprocessed results and Elo datasets sorted by date.
    """
    print("=" * 60)
    print("LOADING CLEAN DATASETS")
    print("=" * 60)

    results = pd.read_csv(RESULTS_FILE)
    elo = pd.read_csv(ELO_FILE)

    results["date"] = pd.to_datetime(results["date"])
    elo["date"] = pd.to_datetime(elo["date"])

    results = results.sort_values("date").reset_index(drop=True)
    elo = elo.sort_values("date").reset_index(drop=True)

    print(f"Results Shape: {results.shape}")
    print(f"Elo Shape    : {elo.shape}")

    return results, elo

# ==========================================================
# Optimized Elo Tracker (Binary Search)
# ==========================================================

class EloTracker:
    """
    Efficient Elo rating lookup system using per-team pre-sorted date/rating arrays.
    Enables O(log M) binary search queries for team rating prior to any given date.
    """
    def __init__(self, elo_df: pd.DataFrame):
        self.team_dates: Dict[str, List[pd.Timestamp]] = {}
        self.team_ratings: Dict[str, List[float]] = {}
        self._build_index(elo_df)

    def _build_index(self, elo_df: pd.DataFrame):
        df_sorted = elo_df.sort_values("date")
        self.norm_map: Dict[str, str] = {}
        for team, group in df_sorted.groupby("team"):
            self.team_dates[team] = group["date"].tolist()
            self.team_ratings[team] = group["rating"].tolist()
            normalized = str(team).replace("\xa0", " ").strip()
            self.norm_map[normalized.lower()] = team

    def _resolve_team(self, team: str) -> str:
        if team in self.team_ratings:
            return team
        normalized = str(team).replace("\xa0", " ").strip().lower()
        return self.norm_map.get(normalized, team)

    def get_latest_elo(self, team: str, match_date: pd.Timestamp) -> Optional[float]:
        """
        Returns the latest Elo rating for a team strictly BEFORE match_date.
        """
        team_key = self._resolve_team(team)
        if team_key not in self.team_dates:
            return None
        dates = self.team_dates[team_key]
        idx = bisect_right(dates, match_date - pd.Timedelta(nanoseconds=1))
        if idx == 0:
            return None
        return self.team_ratings[team_key][idx - 1]

    def get_most_recent_elo(self, team: str) -> Optional[float]:
        """
        Returns the most recent Elo rating available in the dataset for a team.
        """
        team_key = self._resolve_team(team)
        if team_key not in self.team_ratings or not self.team_ratings[team_key]:
            return None
        return self.team_ratings[team_key][-1]

# ==========================================================
# Chronological Team State Tracker
# ==========================================================

class TeamMatchRecord:
    """
    Immutable representation of a single historical match performance for a team.
    """
    def __init__(
        self,
        date: pd.Timestamp,
        goals_for: float,
        goals_against: float,
        opponent: str,
        opponent_elo: Optional[float],
        elo: Optional[float],
    ):
        self.date = date
        self.goals_for = goals_for
        self.goals_against = goals_against
        self.opponent = opponent
        self.opponent_elo = opponent_elo
        self.elo = elo

        if goals_for > goals_against:
            self.points = 3
            self.outcome = "W"
        elif goals_for == goals_against:
            self.points = 1
            self.outcome = "D"
        else:
            self.points = 0
            self.outcome = "L"


class TeamStateTracker:
    """
    Chronological state tracker for all teams. Maintains match history
    and calculates rolling statistics strictly as matches occur to prevent data leakage.
    """
    def __init__(self, window_size: int = DEFAULT_LOOKBACK_WINDOW):
        self.window_size = window_size
        self.history: Dict[str, List[TeamMatchRecord]] = {}

    def get_team_stats(self, team: str, match_date: pd.Timestamp, current_elo: Optional[float]) -> Dict[str, float]:
        """
        Computes rolling statistics for a team strictly using matches prior to match_date.
        """
        records = self.history.get(team, [])

        if not records:
            default_elo = current_elo if current_elo is not None else 1500.0
            return {
                "form_last5": 0.0,
                "goals_last5": 0.0,
                "conceded_last5": 0.0,
                "goal_difference_last5": 0.0,
                "win_percentage_last5": 0.0,
                "draw_percentage_last5": 0.0,
                "matches_last5": 0.0,
                "average_goals_last5": 0.0,
                "average_conceded_last5": 0.0,
                "clean_sheets_last5": 0.0,
                "failed_to_score_last5": 0.0,
                "winning_streak": 0.0,
                "unbeaten_streak": 0.0,
                "days_since_last_match": float(DEFAULT_REST_DAYS),
                "avg_opponent_elo_last5": float(default_elo),
                "elo_trend": 0.0,
            }

        last_n = records[-self.window_size:]
        n_matches = len(last_n)

        form_last5 = float(sum(r.points for r in last_n))
        goals_last5 = float(sum(r.goals_for for r in last_n))
        conceded_last5 = float(sum(r.goals_against for r in last_n))
        goal_difference_last5 = goals_last5 - conceded_last5
        wins_last5 = sum(1 for r in last_n if r.outcome == "W")
        draws_last5 = sum(1 for r in last_n if r.outcome == "D")

        win_percentage_last5 = wins_last5 / n_matches if n_matches > 0 else 0.0
        draw_percentage_last5 = draws_last5 / n_matches if n_matches > 0 else 0.0

        average_goals_last5 = goals_last5 / n_matches if n_matches > 0 else 0.0
        average_conceded_last5 = conceded_last5 / n_matches if n_matches > 0 else 0.0

        clean_sheets_last5 = float(sum(1 for r in last_n if r.goals_against == 0))
        failed_to_score_last5 = float(sum(1 for r in last_n if r.goals_for == 0))

        # Streaks (working backwards from most recent match)
        winning_streak = 0
        for r in reversed(records):
            if r.outcome == "W":
                winning_streak += 1
            else:
                break

        unbeaten_streak = 0
        for r in reversed(records):
            if r.outcome in ("W", "D"):
                unbeaten_streak += 1
            else:
                break

        days_since_last_match = float((match_date - records[-1].date).days)

        opp_elos = [r.opponent_elo for r in last_n if r.opponent_elo is not None]
        default_elo = current_elo if current_elo is not None else 1500.0
        avg_opponent_elo = float(np.mean(opp_elos)) if opp_elos else float(default_elo)

        # Elo trend: difference between current Elo and Elo at start of rolling window
        oldest_window_elo = last_n[0].elo if last_n[0].elo is not None else current_elo
        elo_trend = float(current_elo - oldest_window_elo) if (current_elo is not None and oldest_window_elo is not None) else 0.0

        return {
            "form_last5": form_last5,
            "goals_last5": goals_last5,
            "conceded_last5": conceded_last5,
            "goal_difference_last5": goal_difference_last5,
            "win_percentage_last5": win_percentage_last5,
            "draw_percentage_last5": draw_percentage_last5,
            "matches_last5": float(n_matches),
            "average_goals_last5": average_goals_last5,
            "average_conceded_last5": average_conceded_last5,
            "clean_sheets_last5": clean_sheets_last5,
            "failed_to_score_last5": failed_to_score_last5,
            "winning_streak": float(winning_streak),
            "unbeaten_streak": float(unbeaten_streak),
            "days_since_last_match": days_since_last_match,
            "avg_opponent_elo_last5": avg_opponent_elo,
            "elo_trend": elo_trend,
        }

    def add_match(self, team: str, record: TeamMatchRecord):
        """
        Record a completed match into a team's history.
        """
        if team not in self.history:
            self.history[team] = []
        self.history[team].append(record)

# ==========================================================
# Head-to-Head (H2H) Tracker
# ==========================================================

class H2HTracker:
    """
    Chronological tracker for head-to-head match history between pairs of teams.
    """
    def __init__(self):
        self.h2h_history: Dict[Tuple[str, str], List[Tuple[float, float, bool]]] = {}

    def _get_key(self, team1: str, team2: str) -> Tuple[str, str]:
        return (team1, team2) if team1 < team2 else (team2, team1)

    def get_h2h_stats(self, home_team: str, away_team: str) -> Dict[str, float]:
        """
        Calculates head-to-head statistics prior to match date.
        """
        key = self._get_key(home_team, away_team)
        matches = self.h2h_history.get(key, [])

        if not matches:
            return {
                "h2h_home_wins": 0.0,
                "h2h_away_wins": 0.0,
                "h2h_draws": 0.0,
                "h2h_matches_played": 0.0,
                "h2h_home_win_percentage": 0.0,
                "h2h_home_goal_diff": 0.0,
            }

        home_wins = 0
        away_wins = 0
        draws = 0
        home_goal_diff = 0.0

        for t1_score, t2_score, t1_is_home in matches:
            # Reconstruct home/away relative to current home_team query
            query_is_t1 = (home_team == key[0])
            if query_is_t1:
                h_score, a_score = t1_score, t2_score
            else:
                h_score, a_score = t2_score, t1_score

            home_goal_diff += (h_score - a_score)

            if h_score > a_score:
                home_wins += 1
            elif a_score > h_score:
                away_wins += 1
            else:
                draws += 1

        total_played = float(len(matches))
        return {
            "h2h_home_wins": float(home_wins),
            "h2h_away_wins": float(away_wins),
            "h2h_draws": float(draws),
            "h2h_matches_played": total_played,
            "h2h_home_win_percentage": home_wins / total_played if total_played > 0 else 0.0,
            "h2h_home_goal_diff": home_goal_diff,
        }

    def add_match(self, home_team: str, away_team: str, home_score: float, away_score: float):
        """
        Record a completed match into H2H history.
        """
        key = self._get_key(home_team, away_team)
        if key not in self.h2h_history:
            self.h2h_history[key] = []
        t1_is_home = (home_team == key[0])
        if t1_is_home:
            self.h2h_history[key].append((home_score, away_score, True))
        else:
            self.h2h_history[key].append((away_score, home_score, False))

# ==========================================================
# Master Single-Pass Feature Generation Engine
# ==========================================================

def build_features_and_trackers(
    results_df: pd.DataFrame,
    elo_df: pd.DataFrame
) -> Tuple[pd.DataFrame, TeamStateTracker, EloTracker, H2HTracker]:
    """
    Computes all engineered features in an O(N) single-pass loop across results_df.
    Guarantees chronological correctness and populates reusable trackers for inference.
    """
    print("\nBuilding features in single-pass O(N) loop...")

    elo_tracker = EloTracker(elo_df)
    team_tracker = TeamStateTracker(window_size=DEFAULT_LOOKBACK_WINDOW)
    h2h_tracker = H2HTracker()

    feature_rows = []
    total_matches = len(results_df)

    for idx, row in results_df.iterrows():
        match_date = row["date"]
        home_team = row["home_team"]
        away_team = row["away_team"]

        # 1. Elo Ratings prior to match
        home_elo = elo_tracker.get_latest_elo(home_team, match_date)
        away_elo = elo_tracker.get_latest_elo(away_team, match_date)

        # 2. Team Rolling Stats prior to match
        home_stats = team_tracker.get_team_stats(home_team, match_date, home_elo)
        away_stats = team_tracker.get_team_stats(away_team, match_date, away_elo)

        # 3. Head-to-Head Stats prior to match
        h2h_stats = h2h_tracker.get_h2h_stats(home_team, away_team)

        # Build feature dict for current match
        match_features = {
            # Metadata & Identifiers
            "date": match_date,
            "home_team": home_team,
            "away_team": away_team,
            "home_score": row["home_score"],
            "away_score": row["away_score"],
            "tournament": row["tournament"],
            "city": row.get("city", ""),
            "country": row.get("country", ""),
            "neutral": int(row["neutral"]),
            "result": row["result"],

            # Elo features
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_difference": (home_elo - away_elo) if (home_elo is not None and away_elo is not None) else np.nan,
            "match_year": match_date.year,

            # Original Home Rolling Stats
            "home_form_last5": home_stats["form_last5"],
            "home_goals_last5": home_stats["goals_last5"],
            "home_conceded_last5": home_stats["conceded_last5"],
            "home_goal_difference_last5": home_stats["goal_difference_last5"],
            "home_win_percentage_last5": home_stats["win_percentage_last5"],
            "home_draw_percentage_last5": home_stats["draw_percentage_last5"],
            "home_matches_last5": home_stats["matches_last5"],
            "home_average_goals_last5": home_stats["average_goals_last5"],
            "home_average_conceded_last5": home_stats["average_conceded_last5"],

            # Original Away Rolling Stats
            "away_form_last5": away_stats["form_last5"],
            "away_goals_last5": away_stats["goals_last5"],
            "away_conceded_last5": away_stats["conceded_last5"],
            "away_goal_difference_last5": away_stats["goal_difference_last5"],
            "away_win_percentage_last5": away_stats["win_percentage_last5"],
            "away_draw_percentage_last5": away_stats["draw_percentage_last5"],
            "away_matches_last5": away_stats["matches_last5"],
            "away_average_goals_last5": away_stats["average_goals_last5"],
            "away_average_conceded_last5": away_stats["average_conceded_last5"],

            # NEW Home Metrics
            "home_clean_sheets_last5": home_stats["clean_sheets_last5"],
            "home_failed_to_score_last5": home_stats["failed_to_score_last5"],
            "home_winning_streak": home_stats["winning_streak"],
            "home_unbeaten_streak": home_stats["unbeaten_streak"],
            "home_days_since_last_match": home_stats["days_since_last_match"],
            "home_avg_opponent_elo_last5": home_stats["avg_opponent_elo_last5"],
            "home_elo_trend": home_stats["elo_trend"],

            # NEW Away Metrics
            "away_clean_sheets_last5": away_stats["clean_sheets_last5"],
            "away_failed_to_score_last5": away_stats["failed_to_score_last5"],
            "away_winning_streak": away_stats["winning_streak"],
            "away_unbeaten_streak": away_stats["unbeaten_streak"],
            "away_days_since_last_match": away_stats["days_since_last_match"],
            "away_avg_opponent_elo_last5": away_stats["avg_opponent_elo_last5"],
            "away_elo_trend": away_stats["elo_trend"],

            # H2H Features
            "h2h_home_wins": h2h_stats["h2h_home_wins"],
            "h2h_away_wins": h2h_stats["h2h_away_wins"],
            "h2h_draws": h2h_stats["h2h_draws"],
            "h2h_matches_played": h2h_stats["h2h_matches_played"],
            "h2h_home_win_percentage": h2h_stats["h2h_home_win_percentage"],
            "h2h_home_goal_diff": h2h_stats["h2h_home_goal_diff"],

            # Categorical / Match Context
            "home_advantage": 1 - int(row["neutral"]),
            "is_world_cup": int(row["tournament"] == "FIFA World Cup"),
            "tournament_type": TOURNAMENT_MAPPING.get(row["tournament"], 99),
        }

        feature_rows.append(match_features)

        # 4. Update team and H2H state AFTER feature computation (zero leakage)
        home_score = float(row["home_score"])
        away_score = float(row["away_score"])

        team_tracker.add_match(
            home_team,
            TeamMatchRecord(match_date, home_score, away_score, away_team, away_elo, home_elo)
        )
        team_tracker.add_match(
            away_team,
            TeamMatchRecord(match_date, away_score, home_score, home_team, home_elo, away_elo)
        )
        h2h_tracker.add_match(home_team, away_team, home_score, away_score)

    features_df = pd.DataFrame(feature_rows)

    # 5. Compute Vectorized Difference Features
    print("Computing vectorized difference features...")
    features_df["form_difference"] = features_df["home_form_last5"] - features_df["away_form_last5"]
    features_df["goal_difference_difference"] = features_df["home_goal_difference_last5"] - features_df["away_goal_difference_last5"]
    features_df["win_percentage_difference"] = features_df["home_win_percentage_last5"] - features_df["away_win_percentage_last5"]
    features_df["draw_percentage_difference"] = features_df["home_draw_percentage_last5"] - features_df["away_draw_percentage_last5"]
    features_df["avg_goals_difference"] = features_df["home_average_goals_last5"] - features_df["away_average_goals_last5"]
    features_df["avg_conceded_difference"] = features_df["home_average_conceded_last5"] - features_df["away_average_conceded_last5"]
    features_df["clean_sheets_difference"] = features_df["home_clean_sheets_last5"] - features_df["away_clean_sheets_last5"]
    features_df["failed_to_score_difference"] = features_df["home_failed_to_score_last5"] - features_df["away_failed_to_score_last5"]
    features_df["winning_streak_difference"] = features_df["home_winning_streak"] - features_df["away_winning_streak"]
    features_df["unbeaten_streak_difference"] = features_df["home_unbeaten_streak"] - features_df["away_unbeaten_streak"]
    features_df["days_since_last_match_difference"] = features_df["home_days_since_last_match"] - features_df["away_days_since_last_match"]
    features_df["opponent_elo_difference"] = features_df["home_avg_opponent_elo_last5"] - features_df["away_avg_opponent_elo_last5"]
    features_df["elo_trend_difference"] = features_df["home_elo_trend"] - features_df["away_elo_trend"]

    print("[OK] Single-pass feature engineering complete.")
    return features_df, team_tracker, elo_tracker, h2h_tracker

# ==========================================================
# Save Features
# ==========================================================

def save_features(df: pd.DataFrame):
    """
    Save engineered features to CSV.
    """
    FEATURES_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FEATURES_FILE, index=False)
    print(f"\nFeatures saved to:\n{FEATURES_FILE}")

# ==========================================================
# Main Execution
# ==========================================================

def main():
    results, elo = load_data()
    features_df, team_tracker, elo_tracker, h2h_tracker = build_features_and_trackers(results, elo)

    print("\nFiltering matches missing historical Elo ratings...")
    before = len(features_df)
    features_df = features_df.dropna(subset=["home_elo", "away_elo"]).reset_index(drop=True)
    after = len(features_df)
    print(f"Removed {before - after} matches lacking initial Elo ratings.")

    save_features(features_df)

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 60)
    print(f"Final Dataset Shape: {features_df.shape}")
    print("\nEngineered Feature Columns:")
    for col in features_df.columns:
        print(f"  - {col}")

    print("\nTarget Class Distribution:")
    print(features_df["result"].value_counts())

if __name__ == "__main__":
    main()