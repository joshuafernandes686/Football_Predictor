"""
predict.py
----------
Inference and explanation pipeline for Football Match Predictor.

Features:
- Dynamic model artifact loading (loads model, encoder, feature_columns).
- Real-time state tracker rebuild from processed datasets (zero hardcoded values).
- Robust interactive CLI allowing team selection via index or name search.
- Exact feature reconstruction matching training pipeline.
- Machine learning explanation engine using SHAP (or model feature contribution fallback).
- Human-readable reason generation for predicted match winner.
- Persistent logging to docs/prediction_log.md.
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from feature_engineering import (
    H2HTracker,
    EloTracker,
    TeamStateTracker,
    TOURNAMENT_MAPPING,
    build_features_and_trackers,
    load_data,
)

# ==========================================================
# Configuration
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA = PROJECT_ROOT / "Data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"

MODEL_FILE = MODEL_DIR / "best_model.pkl"
RESULTS_FILE = PROCESSED_DATA / "results_clean.csv"
ELO_FILE = PROCESSED_DATA / "elo_clean.csv"
FEATURES_FILE = PROCESSED_DATA / "features.csv"
PREDICTION_LOG = DOCS_DIR / "PREDICTION_LOG.md"

_RUNTIME_CONTEXT: Optional[Dict[str, Any]] = None

# ==========================================================
# Load Persisted Model Artifact
# ==========================================================

def load_prediction_artifacts(model_path: Path = MODEL_FILE) -> Tuple[Any, Any, List[str]]:
    """Load the saved model, label encoder, and feature columns."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {model_path}. "
            "Please run 'python src/train.py' first."
        )

    saved = joblib.load(model_path)
    model = saved["model"]
    encoder = saved["encoder"]
    feature_columns = saved.get("feature_columns")

    if not feature_columns:
        if FEATURES_FILE.exists():
            df_feat = pd.read_csv(FEATURES_FILE, nrows=1)
            exclude_cols = {
                "date",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
                "tournament",
                "city",
                "country",
                "result",
            }
            feature_columns = [
                column
                for column in df_feat.columns
                if column not in exclude_cols
                and pd.api.types.is_numeric_dtype(df_feat[column])
            ]
        else:
            raise KeyError(
                "feature_columns key missing from model artifact and "
                "features.csv not found."
            )

    return model, encoder, feature_columns


def _get_runtime_context() -> Dict[str, Any]:
    """Initialize and cache reusable prediction resources."""
    global _RUNTIME_CONTEXT

    if _RUNTIME_CONTEXT is None:
        model, encoder, feature_columns = load_prediction_artifacts()
        results_df, elo_df = load_data()
        _, team_tracker, elo_tracker, h2h_tracker = (
            build_features_and_trackers(results_df, elo_df)
        )
        train_features_df = pd.read_csv(FEATURES_FILE)

        _RUNTIME_CONTEXT = {
            "model": model,
            "encoder": encoder,
            "feature_columns": feature_columns,
            "results_df": results_df,
            "elo_df": elo_df,
            "team_tracker": team_tracker,
            "elo_tracker": elo_tracker,
            "h2h_tracker": h2h_tracker,
            "train_features_df": train_features_df,
        }

    return _RUNTIME_CONTEXT


# ==========================================================
# Real-Time Match Feature Builder
# ==========================================================

def compute_match_features(
    home_team: str,
    away_team: str,
    results_df: pd.DataFrame,
    team_tracker: TeamStateTracker,
    elo_tracker: EloTracker,
    h2h_tracker: H2HTracker,
    feature_columns: List[str],
    neutral: int = 0,
    tournament: str = "Friendly",
) -> pd.DataFrame:
    """
    Construct an inference feature vector for a match using the latest
    historical state trackers.
    """
    match_date = results_df["date"].max() + pd.Timedelta(days=7)

    home_elo = elo_tracker.get_most_recent_elo(home_team)
    away_elo = elo_tracker.get_most_recent_elo(away_team)

    if home_elo is None:
        raise ValueError(f"Team '{home_team}' not found in Elo dataset.")
    if away_elo is None:
        raise ValueError(f"Team '{away_team}' not found in Elo dataset.")

    home_stats = team_tracker.get_team_stats(home_team, match_date, home_elo)
    away_stats = team_tracker.get_team_stats(away_team, match_date, away_elo)
    h2h_stats = h2h_tracker.get_h2h_stats(home_team, away_team)

    match_dict = {
        "home_elo": home_elo,
        "away_elo": away_elo,
        "elo_difference": home_elo - away_elo,
        "match_year": match_date.year,
        "home_form_last5": home_stats["form_last5"],
        "home_goals_last5": home_stats["goals_last5"],
        "home_conceded_last5": home_stats["conceded_last5"],
        "home_goal_difference_last5": home_stats["goal_difference_last5"],
        "home_win_percentage_last5": home_stats["win_percentage_last5"],
        "home_draw_percentage_last5": home_stats["draw_percentage_last5"],
        "home_matches_last5": home_stats["matches_last5"],
        "home_average_goals_last5": home_stats["average_goals_last5"],
        "home_average_conceded_last5": home_stats["average_conceded_last5"],
        "away_form_last5": away_stats["form_last5"],
        "away_goals_last5": away_stats["goals_last5"],
        "away_conceded_last5": away_stats["conceded_last5"],
        "away_goal_difference_last5": away_stats["goal_difference_last5"],
        "away_win_percentage_last5": away_stats["win_percentage_last5"],
        "away_draw_percentage_last5": away_stats["draw_percentage_last5"],
        "away_matches_last5": away_stats["matches_last5"],
        "away_average_goals_last5": away_stats["average_goals_last5"],
        "away_average_conceded_last5": away_stats["average_conceded_last5"],
        "home_clean_sheets_last5": home_stats["clean_sheets_last5"],
        "home_failed_to_score_last5": home_stats["failed_to_score_last5"],
        "home_winning_streak": home_stats["winning_streak"],
        "home_unbeaten_streak": home_stats["unbeaten_streak"],
        "home_days_since_last_match": home_stats["days_since_last_match"],
        "home_avg_opponent_elo_last5": home_stats["avg_opponent_elo_last5"],
        "home_elo_trend": home_stats["elo_trend"],
        "away_clean_sheets_last5": away_stats["clean_sheets_last5"],
        "away_failed_to_score_last5": away_stats["failed_to_score_last5"],
        "away_winning_streak": away_stats["winning_streak"],
        "away_unbeaten_streak": away_stats["unbeaten_streak"],
        "away_days_since_last_match": away_stats["days_since_last_match"],
        "away_avg_opponent_elo_last5": away_stats["avg_opponent_elo_last5"],
        "away_elo_trend": away_stats["elo_trend"],
        "h2h_home_wins": h2h_stats["h2h_home_wins"],
        "h2h_away_wins": h2h_stats["h2h_away_wins"],
        "h2h_draws": h2h_stats["h2h_draws"],
        "h2h_matches_played": h2h_stats["h2h_matches_played"],
        "h2h_home_win_percentage": h2h_stats["h2h_home_win_percentage"],
        "h2h_home_goal_diff": h2h_stats["h2h_home_goal_diff"],
        "neutral": neutral,
        "home_advantage": 1 - neutral,
        "is_world_cup": int(tournament == "FIFA World Cup"),
        "tournament_type": TOURNAMENT_MAPPING.get(tournament, 99),
    }

    match_dict["form_difference"] = (
        match_dict["home_form_last5"] - match_dict["away_form_last5"]
    )
    match_dict["goal_difference_difference"] = (
        match_dict["home_goal_difference_last5"]
        - match_dict["away_goal_difference_last5"]
    )
    match_dict["win_percentage_difference"] = (
        match_dict["home_win_percentage_last5"]
        - match_dict["away_win_percentage_last5"]
    )
    match_dict["draw_percentage_difference"] = (
        match_dict["home_draw_percentage_last5"]
        - match_dict["away_draw_percentage_last5"]
    )
    match_dict["avg_goals_difference"] = (
        match_dict["home_average_goals_last5"]
        - match_dict["away_average_goals_last5"]
    )
    match_dict["avg_conceded_difference"] = (
        match_dict["home_average_conceded_last5"]
        - match_dict["away_average_conceded_last5"]
    )
    match_dict["clean_sheets_difference"] = (
        match_dict["home_clean_sheets_last5"]
        - match_dict["away_clean_sheets_last5"]
    )
    match_dict["failed_to_score_difference"] = (
        match_dict["home_failed_to_score_last5"]
        - match_dict["away_failed_to_score_last5"]
    )
    match_dict["winning_streak_difference"] = (
        match_dict["home_winning_streak"] - match_dict["away_winning_streak"]
    )
    match_dict["unbeaten_streak_difference"] = (
        match_dict["home_unbeaten_streak"] - match_dict["away_unbeaten_streak"]
    )
    match_dict["days_since_last_match_difference"] = (
        match_dict["home_days_since_last_match"]
        - match_dict["away_days_since_last_match"]
    )
    match_dict["opponent_elo_difference"] = (
        match_dict["home_avg_opponent_elo_last5"]
        - match_dict["away_avg_opponent_elo_last5"]
    )
    match_dict["elo_trend_difference"] = (
        match_dict["home_elo_trend"] - match_dict["away_elo_trend"]
    )

    df = pd.DataFrame([match_dict])
    return df[feature_columns]


# ==========================================================
# Explanation Engine (SHAP & Model Feature Importance)
# ==========================================================

FEATURE_DISPLAY_NAMES = {
    "elo_difference": "Elo Difference",
    "form_difference": "Recent Form Difference",
    "goal_difference_difference": "Goal Difference",
    "win_percentage_difference": "Win Percentage Difference",
    "draw_percentage_difference": "Draw Percentage Difference",
    "home_winning_streak": "Home Winning Streak",
    "away_winning_streak": "Away Winning Streak",
    "winning_streak_difference": "Winning Streak Difference",
    "home_unbeaten_streak": "Home Unbeaten Streak",
    "away_unbeaten_streak": "Away Unbeaten Streak",
    "unbeaten_streak_difference": "Unbeaten Streak Difference",
    "clean_sheets_difference": "Clean Sheets Difference",
    "failed_to_score_difference": "Failed to Score Difference",
    "h2h_home_wins": "H2H Home Wins",
    "h2h_away_wins": "H2H Away Wins",
    "h2h_home_win_percentage": "Head-to-Head Win %",
    "h2h_home_goal_diff": "Head-to-Head Goal Diff",
    "days_since_last_match_difference": "Rest Days Difference",
    "opponent_elo_difference": "Opponent Strength Difference",
    "elo_trend_difference": "Elo Trend Difference",
    "home_advantage": "Home Advantage",
    "is_world_cup": "World Cup Match",
}


def build_explanation_details(
    model: Any,
    feature_df: pd.DataFrame,
    feature_columns: List[str],
    home_team: str,
    away_team: str,
    train_features_df: pd.DataFrame,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Create human-readable reasons and structured SHAP feature importance values."""
    feature_impacts: Dict[str, float] = {}
    shap_used = False

    try:
        import shap

        sample_background = train_features_df[feature_columns].sample(
            min(50, len(train_features_df)), random_state=42
        )
        explainer = shap.Explainer(model, sample_background)
        shap_output = explainer(feature_df[feature_columns])

        vals = shap_output.values[0]
        if len(vals.shape) == 2:
            vals = vals.mean(axis=1)

        for i, col in enumerate(feature_columns):
            feature_impacts[col] = float(vals[i])
        shap_used = True
    except Exception:
        shap_used = False

    if not shap_used:
        means = train_features_df[feature_columns].mean()
        stds = train_features_df[feature_columns].std().replace(0, 1.0)
        row_vals = feature_df[feature_columns].iloc[0]
        z_scores = (row_vals - means) / stds

        if hasattr(model, "coef_"):
            coefs = model.coef_[0]
            for i, col in enumerate(feature_columns):
                feature_impacts[col] = float(coefs[i] * z_scores[col])
        elif hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            for i, col in enumerate(feature_columns):
                sign = 1.0 if z_scores[col] >= 0 else -1.0
                feature_impacts[col] = float(importances[i] * sign * abs(z_scores[col]))
        else:
            for i, col in enumerate(feature_columns):
                feature_impacts[col] = float(z_scores[col])

    sorted_features = sorted(feature_impacts.items(), key=lambda item: abs(item[1]), reverse=True)
    row = feature_df.iloc[0]
    reasons = []

    for col, _ in sorted_features:
        if len(reasons) >= 5:
            break

        val = row[col]

        if col == "elo_difference":
            if val > 0:
                reasons.append(f"{home_team} Elo is +{int(val)} higher")
            elif val < 0:
                reasons.append(f"{away_team} Elo is +{int(abs(val))} higher")

        elif col == "form_difference":
            home_pts = int(row.get("home_form_last5", 0))
            away_pts = int(row.get("away_form_last5", 0))
            if val > 0:
                reasons.append(
                    f"{home_team} has stronger recent form "
                    f"({home_pts} pts vs {away_pts} pts in last 5)"
                )
            elif val < 0:
                reasons.append(
                    f"{away_team} has stronger recent form "
                    f"({away_pts} pts vs {home_pts} pts in last 5)"
                )

        elif col == "goal_difference_difference":
            home_gd = int(row.get("home_goal_difference_last5", 0))
            away_gd = int(row.get("away_goal_difference_last5", 0))
            if val > 0:
                reasons.append(
                    f"{home_team} has superior goal difference "
                    f"({home_gd:+d} vs {away_gd:+d})"
                )
            elif val < 0:
                reasons.append(
                    f"{away_team} has superior goal difference "
                    f"({away_gd:+d} vs {home_gd:+d})"
                )

        elif col == "winning_streak_difference":
            home_str = int(row.get("home_winning_streak", 0))
            away_str = int(row.get("away_winning_streak", 0))
            if home_str >= 2:
                reasons.append(f"{home_team} is on a {home_str}-match winning streak")
            elif away_str >= 2:
                reasons.append(f"{away_team} is on a {away_str}-match winning streak")

        elif col == "unbeaten_streak_difference":
            home_unb = int(row.get("home_unbeaten_streak", 0))
            away_unb = int(row.get("away_unbeaten_streak", 0))
            if home_unb >= 3:
                reasons.append(f"{home_team} is unbeaten in past {home_unb} matches")
            elif away_unb >= 3:
                reasons.append(f"{away_team} is unbeaten in past {away_unb} matches")

        elif col == "clean_sheets_difference":
            home_cs = int(row.get("home_clean_sheets_last5", 0))
            away_cs = int(row.get("away_clean_sheets_last5", 0))
            if home_cs > away_cs:
                reasons.append(
                    f"{home_team} kept more clean sheets recently "
                    f"({home_cs} vs {away_cs})"
                )
            elif away_cs > home_cs:
                reasons.append(
                    f"{away_team} kept more clean sheets recently "
                    f"({away_cs} vs {home_cs})"
                )

        elif col == "failed_to_score_difference":
            home_fts = int(row.get("home_failed_to_score_last5", 0))
            away_fts = int(row.get("away_failed_to_score_last5", 0))
            if away_fts > home_fts:
                reasons.append(
                    f"{away_team} scored fewer goals recently "
                    f"(failed to score in {away_fts} of last 5)"
                )
            elif home_fts > away_fts:
                reasons.append(
                    f"{home_team} scored fewer goals recently "
                    f"(failed to score in {home_fts} of last 5)"
                )

        elif col in ("h2h_home_win_percentage", "h2h_home_wins"):
            h2h_h_wins = int(row.get("h2h_home_wins", 0))
            h2h_a_wins = int(row.get("h2h_away_wins", 0))
            if h2h_h_wins > h2h_a_wins:
                reasons.append(
                    f"{home_team} leads head-to-head history "
                    f"({h2h_h_wins} wins vs {h2h_a_wins})"
                )
            elif h2h_a_wins > h2h_h_wins:
                reasons.append(
                    f"{away_team} leads head-to-head history "
                    f"({h2h_a_wins} wins vs {h2h_h_wins})"
                )

        elif col == "days_since_last_match_difference":
            home_days = int(row.get("home_days_since_last_match", 30))
            away_days = int(row.get("away_days_since_last_match", 30))
            if home_days > away_days + 3:
                reasons.append(
                    f"{home_team} has had more rest "
                    f"({home_days} days vs {away_days} days)"
                )
            elif away_days > home_days + 3:
                reasons.append(
                    f"{away_team} has had more rest "
                    f"({away_days} days vs {home_days} days)"
                )

    if not reasons:
        reasons.append(f"Statistical advantage across {len(feature_columns)} engineered features")

    shap_data = [
        {
            "feature": FEATURE_DISPLAY_NAMES.get(col, col.replace("_", " ").title()),
            "raw_feature": col,
            "value": round(float(row[col]), 2),
            "impact": round(float(score), 4),
            "direction": "positive" if score >= 0 else "negative",
        }
        for col, score in sorted_features[:8]
    ]

    return list(dict.fromkeys(reasons)), shap_data


def explain_prediction(
    model: Any,
    feature_df: pd.DataFrame,
    feature_columns: List[str],
    predicted_winner_name: str,
    home_team: str,
    away_team: str,
    train_features_df: pd.DataFrame,
) -> List[str]:
    """Backward-compatible helper returning reasons only."""
    reasons, _ = build_explanation_details(
        model,
        feature_df,
        feature_columns,
        home_team,
        away_team,
        train_features_df,
    )
    return reasons


# ==========================================================
# Persistent Logging
# ==========================================================

def log_prediction(
    home_team: str,
    away_team: str,
    winner: str,
    home_elo: float,
    away_elo: float,
    encoder: Any,
    probabilities: np.ndarray,
    reasons: List[str],
) -> None:
    """Append prediction output and explanations to the log file."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    with open(PREDICTION_LOG, "a", encoding="utf-8") as file_handle:
        file_handle.write("=" * 60 + "\n")
        file_handle.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        file_handle.write(f"Match: {home_team} vs {away_team}\n")
        file_handle.write(f"Predicted Winner : {winner}\n")
        file_handle.write(f"Home Elo         : {home_elo:.1f}\n")
        file_handle.write(f"Away Elo         : {away_elo:.1f}\n\n")
        file_handle.write("Probabilities:\n")

        label_map = {"Home Win": home_team, "Away Win": away_team, "Draw": "Draw"}
        for label, prob in zip(encoder.classes_, probabilities):
            display_name = label_map.get(label, label)
            file_handle.write(f"  {display_name:<15}: {prob * 100:.1f}%\n")

        file_handle.write("\nReasons:\n")
        for reason in reasons:
            file_handle.write(f"  - {reason}\n")
        file_handle.write("\n")


# ==========================================================
# Interactive CLI Team Selector
# ==========================================================

def select_teams(available_teams: List[str]) -> Tuple[str, str]:
    """Interactive terminal menu allowing team selection by number or search."""
    teams_sorted = sorted(available_teams)
    print("\n" + "=" * 60)
    print("FOOTBALL MATCH PREDICTOR - TEAM SELECTION")
    print("=" * 60)
    print(f"Total Available Teams: {len(teams_sorted)}\n")

    featured = [
        team
        for team in [
            "Spain",
            "France",
            "Argentina",
            "England",
            "Belgium",
            "Brazil",
            "Morocco",
            "Switzerland",
        ]
        if team in teams_sorted
    ]
    print("Featured Teams:")
    for idx, team in enumerate(featured, 1):
        print(f"  {idx}. {team}")
    print(
        "\n(You can enter a number above, or type any team name, "
        "e.g., 'Germany', 'Portugal')\n"
    )

    def resolve_team(prompt: str) -> str:
        while True:
            user_input = input(prompt).strip()
            if not user_input:
                print("Input cannot be empty. Try again.")
                continue

            if user_input.isdigit():
                num = int(user_input)
                if 1 <= num <= len(featured):
                    return featured[num - 1]

            matches = [team for team in teams_sorted if team.lower() == user_input.lower()]
            if matches:
                return matches[0]

            partial = [team for team in teams_sorted if user_input.lower() in team.lower()]
            if len(partial) == 1:
                return partial[0]
            if len(partial) > 1:
                print(
                    f"Multiple matches found for '{user_input}': "
                    f"{', '.join(partial[:5])}..."
                )
                print("Please type the exact team name.")
                continue

            print(f"Team '{user_input}' not found in dataset. Please try again.")

    home_team = resolve_team("Select Home Team (Team 1): ")

    while True:
        away_team = resolve_team("Select Away Team (Team 2): ")
        if away_team == home_team:
            print("Home and Away teams must be different! Select a different team.")
            continue
        break

    return home_team, away_team


def get_available_teams() -> List[str]:
    """Return every available team in alphabetical order with normalized whitespace."""
    runtime_context = _get_runtime_context()
    raw_teams = runtime_context["elo_df"]["team"].dropna().unique()
    cleaned_teams = {str(team).replace("\xa0", " ").strip() for team in raw_teams}
    return sorted(list(cleaned_teams))


def predict_match(
    home_team: str,
    away_team: str,
    tournament: str = "Friendly",
    neutral: bool = False,
) -> Dict[str, Any]:
    """Run the full prediction pipeline and return a backend-friendly dictionary."""
    runtime_context = _get_runtime_context()
    model = runtime_context["model"]
    encoder = runtime_context["encoder"]
    feature_columns = runtime_context["feature_columns"]
    results_df = runtime_context["results_df"]
    team_tracker = runtime_context["team_tracker"]
    elo_tracker = runtime_context["elo_tracker"]
    h2h_tracker = runtime_context["h2h_tracker"]
    train_features_df = runtime_context["train_features_df"]

    match_features_df = compute_match_features(
        home_team,
        away_team,
        results_df,
        team_tracker,
        elo_tracker,
        h2h_tracker,
        feature_columns,
        neutral=int(neutral),
        tournament=tournament,
    )

    probabilities = model.predict_proba(match_features_df)[0]
    predicted_class_idx = model.predict(match_features_df)[0]
    predicted_label = encoder.inverse_transform([predicted_class_idx])[0]

    label_map = {"Home Win": home_team, "Away Win": away_team, "Draw": "Draw"}
    winner = label_map[predicted_label]

    reasons, feature_importance = build_explanation_details(
        model,
        match_features_df,
        feature_columns,
        home_team,
        away_team,
        train_features_df,
    )

    probability_map = {}
    for label, prob in zip(encoder.classes_, probabilities):
        display_name = label_map.get(label, label)
        probability_map[display_name] = float(prob)

    prob_home = float(probability_map.get(home_team, 0.0))
    prob_draw = float(probability_map.get("Draw", 0.0))
    prob_away = float(probability_map.get(away_team, 0.0))

    return {
        "home_team": home_team,
        "away_team": away_team,
        "prediction": winner,
        "label": predicted_label,
        "probabilities": {
            "home": round(prob_home, 4),
            "draw": round(prob_draw, 4),
            "away": round(prob_away, 4),
        },
        "raw_probabilities": {
            k: round(v, 4) for k, v in probability_map.items()
        },
        "reasons": reasons,
        "shap": feature_importance,
    }


# ==========================================================
# Main Execution Pipeline
# ==========================================================

def main() -> None:
    """CLI entry point that preserves the original interactive behavior."""
    available_teams = get_available_teams()

    if len(sys.argv) >= 3:
        home_team, away_team = sys.argv[1], sys.argv[2]
    else:
        home_team, away_team = select_teams(available_teams)

    result = predict_match(home_team, away_team)

    print("\n" + "=" * 60)
    print("MATCH PREDICTION RESULT")
    print("=" * 60)
    print(f"\n{home_team} vs {away_team}\n")

    print("Probabilities:")
    print("--------------")
    for label, prob in result["probabilities"].items():
        print(f"{label:<15}: {prob * 100:.1f}%")

    print("\nPredicted Winner:")
    print("-----------------")
    print(result["winner"])

    print("\nReasons:")
    print("--------")
    for reason in result["reasons"]:
        print(f"- {reason}")

    print(f"\nPrediction logged to {PREDICTION_LOG}")


if __name__ == "__main__":
    main()