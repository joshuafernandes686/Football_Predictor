import pandas as pd
from pathlib import Path

# ==========================================================
# Configuration
# ==========================================================

PROCESSED_DATA = Path("Data/processed")

RESULTS_FILE = PROCESSED_DATA / "results_clean.csv"
ELO_FILE = PROCESSED_DATA / "elo_clean.csv"

FEATURES_FILE = PROCESSED_DATA / "features.csv"

# ==========================================================
# Load Data
# ==========================================================

def load_data():
    print("=" * 60)
    print("LOADING CLEAN DATASETS")
    print("=" * 60)

    results = pd.read_csv(RESULTS_FILE)
    elo = pd.read_csv(ELO_FILE)

    results["date"] = pd.to_datetime(results["date"])
    elo["date"] = pd.to_datetime(elo["date"])

    results = results.sort_values("date").reset_index(drop=True)
    elo = elo.sort_values("date").reset_index(drop=True)

    return results, elo


# ==========================================================
# Elo Lookup
# ==========================================================

def get_latest_elo(team, match_date, elo_df):
    """
    Returns the latest Elo rating for a team BEFORE the match date.
    """

    team_history = elo_df[
        (elo_df["team"] == team)
        &
        (elo_df["date"] < match_date)
    ]

    if team_history.empty:
        return None

    return team_history.iloc[-1]["rating"]


# ==========================================================
# Feature Engineering
# ==========================================================

def add_elo_features(results_df, elo_df):

    print("\nAdding Elo features...")

    home_elo = []
    away_elo = []

    total_matches = len(results_df)

    for index, row in results_df.iterrows():

        if (index + 1) % 500 == 0:
            print(f"Processed {index + 1}/{total_matches}")

        home_rating = get_latest_elo(
            row["home_team"],
            row["date"],
            elo_df
        )

        away_rating = get_latest_elo(
            row["away_team"],
            row["date"],
            elo_df
        )

        home_elo.append(home_rating)
        away_elo.append(away_rating)

    results_df["home_elo"] = home_elo
    results_df["away_elo"] = away_elo

    results_df["elo_difference"] = (
        results_df["home_elo"]
        -
        results_df["away_elo"]
    )

    results_df["match_year"] = results_df["date"].dt.year

    print("✓ Elo features created.")

    return results_df


# ==========================================================
# Save Features
# ==========================================================

def save_features(df):

    FEATURES_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        FEATURES_FILE,
        index=False
    )

    print(f"\nFeatures saved to:\n{FEATURES_FILE}")


# ==========================================================
# Main
# ==========================================================

def main():

    results, elo = load_data()

    features = add_elo_features(
        results,
        elo
    )

    print("\nRemoving matches without historical Elo...")

    before = len(features)

    features = features.dropna(
        subset=[
            "home_elo",
            "away_elo"
        ]
    )

    after = len(features)

    print(f"Removed {before - after} matches.")

    save_features(features)

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 60)

    print(f"\nFinal Shape: {features.shape}")

    print("\nColumns:")

    for column in features.columns:
        print(column)

    print("\nTarget Distribution:")
    print(features["result"].value_counts())


if __name__ == "__main__":
    main()