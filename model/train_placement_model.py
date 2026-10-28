from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict

import joblib
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from model.feature_selection import build_feature_report
from model.preprocess import build_preprocessor
from utils.config import ARTIFACT_FILES, DATASET_PATH, INPUT_FEATURES, RANDOM_STATE, ensure_directories


def _rounded_metrics(y_true, y_pred, y_prob) -> Dict[str, float]:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
    }


def _model_candidates() -> Dict[str, Any]:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced", random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, min_samples_leaf=18, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=220,
            max_depth=14,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=260,
            max_depth=16,
            min_samples_leaf=6,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.055,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
    }


def train_placement_model(dataset_path=DATASET_PATH) -> Dict[str, Any]:
    ensure_directories()
    df = pd.read_csv(dataset_path)
    X = df[INPUT_FEATURES]
    y = df["placement_status"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.22,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    comparison: Dict[str, Dict[str, float]] = {}
    fitted_pipelines: Dict[str, Pipeline] = {}
    k_features = min(42, len(INPUT_FEATURES) + 30)

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
        y_prob = pipeline.predict_proba(X_test)[:, list(pipeline.classes_).index(1)]
        metrics = _rounded_metrics(y_test, y_pred, y_prob)
        metrics["selection_score"] = round(0.45 * metrics["f1"] + 0.45 * metrics["roc_auc"] + 0.10 * metrics["accuracy"], 4)
        comparison[name] = metrics
        fitted_pipelines[name] = pipeline

    best_model_name = max(comparison, key=lambda model_name: comparison[model_name]["selection_score"])
    best_pipeline = fitted_pipelines[best_model_name]
    feature_report = build_feature_report(best_pipeline, X_train, y_train, top_n=40, random_state=RANDOM_STATE)

    joblib.dump(best_pipeline, ARTIFACT_FILES["placement_model"])
    joblib.dump(best_pipeline.named_steps["preprocessor"], ARTIFACT_FILES["placement_preprocessor"])
    joblib.dump(feature_report, ARTIFACT_FILES["selected_features_placement"])

    payload: Dict[str, Any] = {
        "task": "placement_prediction",
        "best_model": best_model_name,
        "comparison": comparison,
        "test_size": len(X_test),
        "train_size": len(X_train),
        "positive_rate": round(float(y.mean()), 4),
        "selected_feature_count": feature_report["selected_count"],
        "dropped_feature_count": feature_report["dropped_count"],
        "top_features": feature_report["top_features"][:18],
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }

    with open(ARTIFACT_FILES["placement_metrics"], "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    return payload


if __name__ == "__main__":
    train_placement_model()
