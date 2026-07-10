import joblib
import pandas as pd
from pathlib import Path

# ==========================================================
# Configuration
# ==========================================================

PROCESSED_DATA = Path("Data/processed")
MODEL_DIR = Path("models")
DOCS_DIR = Path("docs")
PREDICTION_LOG = DOCS_DIR / "PREDICTION_LOG.md"

MODEL_FILE = MODEL_DIR / "baseline_logistic_regression.pkl"
ELO_FILE = PROCESSED_DATA / "elo_clean.csv"

# ==========================================================
# Load Model
# ==========================================================

saved = joblib.load(MODEL_FILE)

model = saved["model"]
encoder = saved["encoder"]

# ==========================================================
# Load Elo Dataset
# ==========================================================

elo = pd.read_csv(ELO_FILE)

# ==========================================================
# User Input
# ==========================================================

# ==========================================================
# Team Selection Menu
# ==========================================================

teams = {
    1: "Belgium",
    2: "Spain",
    3: "France",
    4: "Morocco",
    5: "Norway",
    6: "England",
    7: "Argentina",
    8: "Switzerland"
}

print("\n" + "=" * 60)
print("WORLD CUP QUARTERFINALISTS")
print("=" * 60)

for number, team in teams.items():
    print(f"{number}. {team}")

print()

while True:
    try:
        home_choice = int(input("Select Team 1 (number): "))
        away_choice = int(input("Select Team 2 (number): "))

        if home_choice not in teams or away_choice not in teams:
            print("\nInvalid selection. Try again.\n")
            continue

        if home_choice == away_choice:
            print("\nPlease select two different teams.\n")
            continue

        break

    except ValueError:
        print("\nPlease enter numbers only.\n")

home_team = teams[home_choice]
away_team = teams[away_choice]
# ==========================================================
# Get Latest Elo Ratings
# ==========================================================

def latest_elo(team):

    team_data = elo[elo["team"] == team]

    if team_data.empty:
        raise ValueError(f"{team} not found in Elo dataset.")

    return team_data.iloc[-1]["rating"]


home_elo = latest_elo(home_team)
away_elo = latest_elo(away_team)

elo_difference = home_elo - away_elo

# ==========================================================
# Prepare Input
# ==========================================================

match = pd.DataFrame(
    {
        "home_elo": [home_elo],
        "away_elo": [away_elo],
        "elo_difference": [elo_difference],
    }
)

# ==========================================================
# Prediction
# ==========================================================

prediction = model.predict(match)[0]

probabilities = model.predict_proba(match)[0]

predicted_result = encoder.inverse_transform([prediction])[0]

# ==========================================================
# Convert Prediction to Team Name
# ==========================================================

if predicted_result == "Home Win":
    winner = home_team
elif predicted_result == "Away Win":
    winner = away_team
else:
    winner = "Draw"

# ==========================================================
# Save Prediction
# ==========================================================

def save_prediction():

    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    with open(PREDICTION_LOG, "a", encoding="utf-8") as file:

        file.write("=" * 60 + "\n")
        file.write(f"{home_team} vs {away_team}\n")
        file.write(f"Predicted Winner : {winner}\n")
        file.write(f"Home Elo         : {home_elo}\n")
        file.write(f"Away Elo         : {away_elo}\n\n")

        file.write("Probabilities\n")

        for label, probability in zip(encoder.classes_, probabilities):

            team = {
                "Home Win": home_team,
                "Away Win": away_team,
                "Draw": "Draw"
            }[label]

            file.write(f"{team:<15}: {probability*100:.2f}%\n")

        file.write("\n")
# ==========================================================
# Display
# ==========================================================

print("\n" + "=" * 60)
print("MATCH PREDICTION")
print("=" * 60)

print(f"\n{home_team} vs {away_team}")

print(f"\nHome Elo : {home_elo}")
print(f"Away Elo : {away_elo}")
print(f"Elo Difference : {elo_difference}")

print("\nPredicted Winner")
print("----------------")

if winner == "Draw":
    print("Draw")
else:
    print(winner)

print("\nProbabilities")
print("-------------")

label_map = {
    "Home Win": home_team,
    "Away Win": away_team,
    "Draw": "Draw"
}

for label, probability in zip(encoder.classes_, probabilities):
    print(f"{label_map[label]:<15}: {probability*100:.2f}%")

save_prediction()