from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict

import joblib
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from model.feature_selection import build_feature_report
from model.preprocess import build_preprocessor
from utils.config import ARTIFACT_FILES, DATASET_PATH, INPUT_FEATURES, RANDOM_STATE, ensure_directories


def _model_candidates() -> Dict[str, Any]:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1800, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=240,
            max_depth=18,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=280,
            max_depth=20,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=115,
            learning_rate=0.055,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
    }


def train_role_model(dataset_path=DATASET_PATH) -> Dict[str, Any]:
    ensure_directories()
    df = pd.read_csv(dataset_path)
    X = df[INPUT_FEATURES]
    y = df["recommended_role"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.22,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    comparison: Dict[str, Dict[str, float]] = {}
    fitted_pipelines: Dict[str, Pipeline] = {}
    k_features = min(48, len(INPUT_FEATURES) + 38)

    for name, estimator in _model_candidates().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("selector", SelectKBest(score_func=f_classif, k=k_features)),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        comparison[name] = {
            "accuracy": round(float(accuracy), 4),
            "f1_macro": round(float(f1_macro), 4),
            "f1_weighted": round(float(f1_weighted), 4),
            "selection_score": round(float(0.45 * accuracy + 0.40 * f1_macro + 0.15 * f1_weighted), 4),
        }
        fitted_pipelines[name] = pipeline

    best_model_name = max(comparison, key=lambda model_name: comparison[model_name]["selection_score"])
    best_pipeline = fitted_pipelines[best_model_name]
    feature_report = build_feature_report(best_pipeline, X_train, y_train, top_n=45, random_state=RANDOM_STATE)

    joblib.dump(best_pipeline, ARTIFACT_FILES["role_model"])
    joblib.dump(best_pipeline.named_steps["preprocessor"], ARTIFACT_FILES["role_preprocessor"])
    joblib.dump(feature_report, ARTIFACT_FILES["selected_features_role"])

    role_distribution = df["recommended_role"].value_counts().to_dict()
    payload: Dict[str, Any] = {
        "task": "role_recommendation",
        "best_model": best_model_name,
        "comparison": comparison,
        "test_size": len(X_test),
        "train_size": len(X_train),
        "role_distribution": role_distribution,
        "selected_feature_count": feature_report["selected_count"],
        "dropped_feature_count": feature_report["dropped_count"],
        "top_features": feature_report["top_features"][:18],
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }

    with open(ARTIFACT_FILES["role_metrics"], "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    return payload


if __name__ == "__main__":
    train_role_model()
