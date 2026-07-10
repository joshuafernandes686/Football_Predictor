import pandas as pd
from pathlib import Path

# ==========================================================
# Configuration
# ==========================================================

START_YEAR = 2016

RAW_DATA = Path("data/raw")
PROCESSED_DATA = Path("data/processed")

RESULTS_FILE = RAW_DATA / "results.csv"
ELO_FILE = RAW_DATA / "eloratings.csv"

PROCESSED_RESULTS = PROCESSED_DATA / "results_clean.csv"
PROCESSED_ELO = PROCESSED_DATA / "elo_clean.csv"

# ==========================================================
# Team Name Mapping
# ==========================================================

TEAM_NAME_MAP = {
    "Czech Republic": "Czechia",
    "DR Congo": "Democratic Republic of Congo",
    "Republic of Ireland": "Ireland",
    "Macau": "Macao",
    "Réunion": "Reunion",
    "São Tomé and Príncipe": "Sao Tome and Principe",
    "Saint Barthélemy": "Saint Barthelemy",
    "United States Virgin Islands": "US Virgin Islands",
    "Vatican City": "Vatican",
}

# ==========================================================
# Helper Functions
# ==========================================================

def standardize_team_names(df, columns):
    """
    Standardize team names across datasets.
    """

    for column in columns:
        df[column] = df[column].replace(TEAM_NAME_MAP)

    return df


def create_result(row):
    """
    Create target label.
    """

    if row["home_score"] > row["away_score"]:
        return "Home Win"

    elif row["home_score"] < row["away_score"]:
        return "Away Win"

    else:
        return "Draw"


# ==========================================================
# Load Data
# ==========================================================

print("=" * 60)
print("LOADING DATASETS")
print("=" * 60)

results = pd.read_csv(RESULTS_FILE)
elo = pd.read_csv(ELO_FILE)

# ==========================================================
# Convert Dates
# ==========================================================

print("\nConverting dates...")

results["date"] = pd.to_datetime(results["date"])

elo["date"] = pd.to_datetime(
    elo["date"],
    format="mixed"
)

print("✓ Dates converted.")

# ==========================================================
# Remove Missing Values
# ==========================================================

print("\nRemoving incomplete rows...")

results = results.dropna(
    subset=[
        "home_score",
        "away_score"
    ]
)

elo = elo.dropna(
    subset=[
        "rating"
    ]
)

print("✓ Missing values removed.")

# ==========================================================
# Filter Modern Football
# ==========================================================

print(f"\nKeeping matches from {START_YEAR} onwards...")

results = results[
    results["date"] >= f"{START_YEAR}-01-01"
].copy()

elo = elo[
    elo["date"] >= f"{START_YEAR}-01-01"
].copy()

print("✓ Date filtering complete.")

# ==========================================================
# Remove Duplicate Rows
# ==========================================================

results = results.drop_duplicates()

elo = elo.drop_duplicates()

# ==========================================================
# Standardize Team Names
# ==========================================================

print("\nStandardizing team names...")

results = standardize_team_names(
    results,
    ["home_team", "away_team"]
)

elo = standardize_team_names(
    elo,
    ["team"]
)

print("✓ Team names standardized.")

# ==========================================================
# Remove Teams Without Elo Ratings
# ==========================================================

print("\nRemoving matches without Elo ratings...")

valid_teams = set(elo["team"])

before = len(results)

results = results[
    results["home_team"].isin(valid_teams)
    &
    results["away_team"].isin(valid_teams)
].copy()

after = len(results)

print(f"Removed {before-after} matches.")

# ==========================================================
# Create Target Variable
# ==========================================================

print("\nCreating target variable...")

results["result"] = results.apply(
    create_result,
    axis=1
)

print("✓ Target variable created.")

# ==========================================================
# Sort Data
# ==========================================================

results = results.sort_values(
    "date"
).reset_index(drop=True)

elo = elo.sort_values(
    "date"
).reset_index(drop=True)

# ==========================================================
# Save Processed Data
# ==========================================================

PROCESSED_DATA.mkdir(
    parents=True,
    exist_ok=True
)

results.to_csv(
    PROCESSED_RESULTS,
    index=False
)

elo.to_csv(
    PROCESSED_ELO,
    index=False
)

# ==========================================================
# Summary
# ==========================================================

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE")
print("=" * 60)

print(f"\nResults saved to:")
print(PROCESSED_RESULTS)

print(f"\nElo saved to:")
print(PROCESSED_ELO)

print("\nFinal Results Shape:", results.shape)
print("Final Elo Shape    :", elo.shape)

print("\nTarget Distribution:")
print(results["result"].value_counts())

print("\nUnique Teams:", len(valid_teams))

print("\nDone.")