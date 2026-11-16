"""
Training script for the Risk Assessment Agent tabular model.
Compares XGBoost and Random Forest on Arshi's 100k construction risk dataset
(`construction_risk_dataset_100k.csv`) using strict train/test split and LabelEncoder.
Saves the best-performing model pipeline to `models/risk_model.joblib` and metrics to
`models/risk_100k_metrics.json`.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

try:
    import xgboost as xgb

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

DATA_PATH = Path("construction_risk_dataset_100k.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def train_and_evaluate():
    print(f"Loading 100k risk dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    features = [
        "violation_type",
        "zone_type",
        "weather",
        "workers_involved",
        "height_meters",
        "duration_minutes",
    ]
    target = "severity_label"

    X = df[features]
    y_raw = df[target]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    categorical_cols = ["violation_type", "zone_type", "weather"]
    numeric_cols = ["workers_involved", "height_meters", "duration_minutes"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=80, max_depth=14, random_state=42, n_jobs=-1
        )
    }

    if HAS_XGB:
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        )

    best_model_name = None
    best_score = 0.0
    best_pipeline = None
    metrics_report = {"dataset_rows": len(df), "models": {}}

    for name, clf in models.items():
        print(f"\n--- Training {name} on {len(X_train):,} train rows ---")
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", clf),
            ]
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, average="weighted"))
        macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
        print(f"{name} Test Accuracy: {acc * 100:.2f}% | Weighted F1: {f1:.4f} | Macro F1: {macro_f1:.4f}")

        metrics_report["models"][name] = {
            "accuracy_percent": round(acc * 100, 2),
            "weighted_f1": round(f1, 4),
            "macro_f1": round(macro_f1, 4),
        }

        if acc > best_score:
            best_score = acc
            best_model_name = name
            best_pipeline = pipeline

    print("\n==========================================")
    print(f"Best Model: {best_model_name} with {best_score * 100:.2f}% Accuracy!")
    print("==========================================")

    save_path = MODEL_DIR / "risk_model.joblib"
    joblib.dump({"pipeline": best_pipeline, "label_encoder": label_encoder}, save_path)
    metrics_report["best_model"] = best_model_name
    metrics_report["best_accuracy_percent"] = round(best_score * 100, 2)
    (MODEL_DIR / "risk_100k_metrics.json").write_text(
        json.dumps(metrics_report, indent=2), encoding="utf-8"
    )
    print(f"Saved best model pipeline to {save_path}")
    return metrics_report


if __name__ == "__main__":
    train_and_evaluate()
