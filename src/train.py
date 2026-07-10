import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ==========================================================
# Configuration
# ==========================================================

PROCESSED_DATA = Path("Data/processed")
MODEL_DIR = Path("models")
DOCS_DIR = Path("docs")

FEATURES_FILE = PROCESSED_DATA / "features.csv"
MODEL_FILE = MODEL_DIR / "baseline_logistic_regression.pkl"
REPORT_FILE = DOCS_DIR / "MODEL_REPORT.md"

# ==========================================================
# Load Data
# ==========================================================

def load_data():

    print("=" * 60)
    print("LOADING FEATURES DATASET")
    print("=" * 60)

    df = pd.read_csv(FEATURES_FILE)

    print(f"\nDataset Shape : {df.shape}")

    return df


# ==========================================================
# Prepare Data
# ==========================================================

def prepare_data(df):

    print("\nPreparing data...")

    feature_columns = [
        "home_elo",
        "away_elo",
        "elo_difference"
    ]

    X = df[feature_columns]

    y = df["result"]

    encoder = LabelEncoder()

    y = encoder.fit_transform(y)

    print("\nLabel Mapping")

    for index, label in enumerate(encoder.classes_):
        print(f"{label} --> {index}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"\nTraining Samples : {len(X_train)}")
    print(f"Testing Samples  : {len(X_test)}")

    return X_train, X_test, y_train, y_test, feature_columns


# ==========================================================
# Train Model
# ==========================================================

def train_model(X_train, y_train):

    print("\nTraining Logistic Regression...")

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train)

    print("✓ Training Complete")

    return model


# ==========================================================
# Evaluate Model
# ==========================================================

def evaluate_model(model, X_test, y_test):

    print("\nEvaluating Model...")

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions)

    print(f"\nAccuracy : {accuracy:.4f}")

    print("\nConfusion Matrix")
    print(cm)

    print("\nClassification Report")
    print(report)

    return accuracy, cm, report


# ==========================================================
# Save Model
# ==========================================================

def save_model(model, encoder):

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
    {
        "model": model,
        "encoder": encoder
    },
    MODEL_FILE
)

    print(f"\nModel saved to:\n{MODEL_FILE}")


# ==========================================================
# Save Model Report
# ==========================================================

def save_model_report(
    model_name,
    model,
    features,
    accuracy,
    cm,
    report
):

    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    with open(REPORT_FILE, "a", encoding="utf-8") as file:

        file.write(f"\n# {model_name}\n\n")

        file.write(f"**Date:** {datetime.now()}\n\n")

        file.write("## Features\n\n")

        for feature in features:
            file.write(f"- {feature}\n")

        file.write("\n## Parameters\n\n")
        file.write("```\n")
        file.write(str(model.get_params()))
        file.write("\n```\n\n")

        file.write("## Accuracy\n\n")
        file.write(f"{accuracy:.4f}\n\n")

        file.write("## Confusion Matrix\n\n")
        file.write("```\n")
        file.write(str(cm))
        file.write("\n```\n\n")

        file.write("## Classification Report\n\n")
        file.write("```\n")
        file.write(report)
        file.write("\n```\n\n")

        file.write("---\n\n")

    print(f"\nModel report updated:\n{REPORT_FILE}")


# ==========================================================
# Run Pipeline
# ==========================================================

df = load_data()

X_train, X_test, y_train, y_test, feature_columns = prepare_data(df)

model = train_model(X_train, y_train)

accuracy, cm, report = evaluate_model(
    model,
    X_test,
    y_test
)

save_model(model, encoder=LabelEncoder())

save_model_report(
    model_name="Logistic Regression",
    model=model,
    features=feature_columns,
    accuracy=accuracy,
    cm=cm,
    report=report
)

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)