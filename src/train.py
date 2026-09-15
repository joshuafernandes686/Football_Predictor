"""
train.py
--------
Training pipeline for Football Match Predictor.

Features:
- Dynamic feature selection (no hardcoded feature column arrays).
- Multi-model evaluation: Logistic Regression, Decision Tree, Random Forest, XGBoost.
- Automated hyperparameter tuning with GridSearchCV.
- Artifact persistence containing trained model, label encoder, and feature_columns schema.
- Automatic Markdown model evaluation report generation.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict, Any, Optional

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("[NOTE] xgboost package not found. Falling back to sklearn GradientBoostingClassifier.")
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==========================================================
# Configuration
# ==========================================================

PROCESSED_DATA = Path("Data/processed")
MODEL_DIR = Path("models")
DOCS_DIR = Path("docs")

FEATURES_FILE = PROCESSED_DATA / "features.csv"
REPORT_FILE = DOCS_DIR / "MODEL_REPORT.md"

METADATA_COLUMNS = {
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

# ==========================================================
# Load Data
# ==========================================================

def load_data() -> pd.DataFrame:
    """
    Load engineered features dataset from CSV.
    """
    print("=" * 60)
    print("LOADING FEATURES DATASET")
    print("=" * 60)

    df = pd.read_csv(FEATURES_FILE)
    print(f"Dataset Shape: {df.shape}")
    return df

# ==========================================================
# Dynamic Feature Selection & Data Preparation
# ==========================================================

def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, List[str], LabelEncoder]:
    """
    Dynamically extracts numeric feature columns and encodes target labels.
    """
    print("\nPreparing training data...")

    # Dynamically select feature columns (all numeric columns excluding metadata/target)
    feature_columns = [
        col for col in df.columns
        if col not in METADATA_COLUMNS and pd.api.types.is_numeric_dtype(df[col])
    ]

    print(f"\nDynamically identified {len(feature_columns)} feature columns:")
    for col in feature_columns:
        print(f"  - {col}")

    X = df[feature_columns].copy()
    y_raw = df["result"]

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    print("\nTarget Label Encoding:")
    for index, label in enumerate(encoder.classes_):
        print(f"  {label} --> {index}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"\nTraining Samples : {len(X_train)}")
    print(f"Testing Samples  : {len(X_test)}")

    return X_train, X_test, y_train, y_test, feature_columns, encoder

# ==========================================================
# Train & Hyperparameter Tuning
# ==========================================================

def train_model(
    model: Any,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    params: Optional[Dict[str, List[Any]]] = None
) -> Any:
    """
    Train a model with optional GridSearchCV hyperparameter tuning.
    """
    print("\n" + "=" * 60)
    print(f"TRAINING {model_name.upper()}")
    print("=" * 60)

    if params:
        grid = GridSearchCV(
            estimator=model,
            param_grid=params,
            cv=3,
            scoring="accuracy",
            n_jobs=1
        )
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        print("Best Parameters:")
        print(grid.best_params_)
        return best_model
    else:
        model.fit(X_train, y_train)
        print("[OK] Training Complete")
        return model

# ==========================================================
# Evaluate Model
# ==========================================================

def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: np.ndarray) -> Tuple[float, np.ndarray, str]:
    """
    Evaluate trained model performance on test set.
    """
    print("\nEvaluating Model...")
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions)

    print(f"Accuracy : {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    return accuracy, cm, report

# ==========================================================
# Save Model Artifact
# ==========================================================

def save_model(model: Any, encoder: LabelEncoder, feature_columns: List[str], filename: str):
    """
    Persists model artifact along with encoder and feature column metadata.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "encoder": encoder,
            "feature_columns": feature_columns,
        },
        MODEL_DIR / filename
    )
    print(f"Model saved to: {MODEL_DIR / filename}")

# ==========================================================
# Save Markdown Report
# ==========================================================

def save_model_report(
    model_name: str,
    model: Any,
    features: List[str],
    accuracy: float,
    cm: np.ndarray,
    report: str
):
    """
    Appends model metrics and parameter configuration to MODEL_REPORT.md.
    """
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_FILE, "a", encoding="utf-8") as file:
        file.write(f"\n# {model_name}\n\n")
        file.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        file.write("## Features\n\n")
        for feature in features:
            file.write(f"- {feature}\n")

        file.write("\n## Parameters\n\n```\n")
        file.write(str(model.get_params()))
        file.write("\n```\n\n")

        file.write("## Accuracy\n\n")
        file.write(f"{accuracy:.4f}\n\n")

        file.write("## Confusion Matrix\n\n```\n")
        file.write(str(cm))
        file.write("\n```\n\n")

        file.write("## Classification Report\n\n```\n")
        file.write(report)
        file.write("\n```\n\n---\n\n")

# ==========================================================
# Main Execution Pipeline
# ==========================================================

def main():
    df = load_data()
    X_train, X_test, y_train, y_test, feature_columns, encoder = prepare_data(df)

    model_results = {}
    trained_models = {}

    # 1. Logistic Regression
    logistic_model = train_model(
        LogisticRegression(random_state=42, max_iter=1000),
        "Logistic Regression",
        X_train,
        y_train,
        params=None
    )
    acc, cm, rep = evaluate_model(logistic_model, X_test, y_test)
    model_results["Logistic Regression"] = acc
    trained_models["Logistic Regression"] = logistic_model
    save_model(logistic_model, encoder, feature_columns, "baseline_logistic_regression.pkl")
    save_model_report("Logistic Regression", logistic_model, feature_columns, acc, cm, rep)

    # 2. Decision Tree
    tree_model = train_model(
        DecisionTreeClassifier(random_state=42, max_depth=10),
        "Decision Tree",
        X_train,
        y_train,
        params=None
    )
    acc, cm, rep = evaluate_model(tree_model, X_test, y_test)
    model_results["Decision Tree"] = acc
    trained_models["Decision Tree"] = tree_model
    save_model(tree_model, encoder, feature_columns, "decision_tree.pkl")
    save_model_report("Decision Tree", tree_model, feature_columns, acc, cm, rep)

    # 3. Random Forest
    rf_model = train_model(
        RandomForestClassifier(random_state=42, n_estimators=100, max_depth=15),
        "Random Forest",
        X_train,
        y_train,
        params=None
    )
    acc, cm, rep = evaluate_model(rf_model, X_test, y_test)
    model_results["Random Forest"] = acc
    trained_models["Random Forest"] = rf_model
    save_model(rf_model, encoder, feature_columns, "random_forest.pkl")
    save_model_report("Random Forest", rf_model, feature_columns, acc, cm, rep)

    # 4. XGBoost / Gradient Boosting
    if HAS_XGBOOST:
        gb_estimator = XGBClassifier(random_state=42, n_estimators=50, max_depth=3, eval_metric="mlogloss")
        model_name = "XGBoost"
        save_name = "xgboost.pkl"
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier
        gb_estimator = HistGradientBoostingClassifier(random_state=42, max_iter=50)
        model_name = "Gradient Boosting"
        save_name = "xgboost.pkl"

    xgb_model = train_model(
        gb_estimator,
        model_name,
        X_train,
        y_train,
        params=None
    )
    acc, cm, rep = evaluate_model(xgb_model, X_test, y_test)
    model_results[model_name] = acc
    trained_models[model_name] = xgb_model
    save_model(xgb_model, encoder, feature_columns, save_name)
    save_model_report(model_name, xgb_model, feature_columns, acc, cm, rep)

    # Best Model Selection
    best_name = max(model_results, key=model_results.get)
    best_acc = model_results[best_name]
    best_model_obj = trained_models[best_name]

    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    for name, score in model_results.items():
        print(f"{name:<25} : {score:.4f}")

    print(f"\nSelected Best Model: {best_name} ({best_acc:.4f})")
    save_model(best_model_obj, encoder, feature_columns, "best_model.pkl")

if __name__ == "__main__":
    main()