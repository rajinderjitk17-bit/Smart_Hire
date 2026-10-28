from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd

from model.preprocess import coerce_input_frame, coerce_profile
from model.readiness import compute_readiness_score, explanation_text, strengths_and_weaknesses
from utils.config import ARTIFACT_FILES, INPUT_FEATURES


@lru_cache(maxsize=1)
def load_artifacts() -> Dict[str, Any]:
    return {
        "placement_model": joblib.load(ARTIFACT_FILES["placement_model"]),
        "role_model": joblib.load(ARTIFACT_FILES["role_model"]),
        "placement_features": joblib.load(ARTIFACT_FILES["selected_features_placement"]),
        "role_features": joblib.load(ARTIFACT_FILES["selected_features_role"]),
    }


def _positive_probability(pipeline: Any, frame: pd.DataFrame) -> np.ndarray:
    probabilities = pipeline.predict_proba(frame)
    classes = list(pipeline.classes_)
    if 1 in classes:
        positive_index = classes.index(1)
    else:
        positive_index = 1
    return probabilities[:, positive_index]


def _top_roles(role_pipeline: Any, frame: pd.DataFrame, top_n: int = 3) -> List[List[Dict[str, Any]]]:
    probabilities = role_pipeline.predict_proba(frame)
    classes = list(role_pipeline.classes_)
    top_roles: List[List[Dict[str, Any]]] = []
    for row_probabilities in probabilities:
        top_indices = np.argsort(row_probabilities)[::-1][:top_n]
        top_roles.append(
            [
                {
                    "role": classes[index],
                    "probability": round(float(row_probabilities[index] * 100), 1),
                }
                for index in top_indices
            ]
        )
    return top_roles


def _top_factor_notes(profile: Dict[str, Any]) -> List[str]:
    checks = [
        ("technical_test_score", "Technical test", "strong", "needs lift"),
        ("programming_skills", "Programming", "strong", "needs lift"),
        ("aptitude_score", "Aptitude", "strong", "needs lift"),
        ("communication_skills", "Communication", "strong", "needs lift"),
        ("resume_score", "Resume", "strong", "needs lift"),
        ("mock_interview_score", "Mock interview", "strong", "needs lift"),
    ]
    notes: List[str] = []
    for key, label, high_text, low_text in checks:
        value = float(profile.get(key, 0))
        if value >= 75:
            notes.append(f"{label}: {high_text} ({value:.0f})")
        elif value < 58:
            notes.append(f"{label}: {low_text} ({value:.0f})")
    if float(profile.get("internships_count", 0)) >= 2:
        notes.append("Internships: strong exposure")
    if float(profile.get("projects_count", 0)) >= 4:
        notes.append("Projects: portfolio strength")
    if float(profile.get("backlogs", 0)) > 0:
        notes.append("Backlogs: placement risk")
    return notes[:6] or ["Signals are mixed, with no single dominant factor."]


def predict_student(profile: Dict[str, Any]) -> Dict[str, Any]:
    artifacts = load_artifacts()
    clean_profile = coerce_profile(profile)
    frame = coerce_input_frame(clean_profile)

    placement_model = artifacts["placement_model"]
    role_model = artifacts["role_model"]

    placement_pred = int(placement_model.predict(frame)[0])
    placement_probability = float(_positive_probability(placement_model, frame)[0] * 100)
    confidence = placement_probability if placement_pred == 1 else 100 - placement_probability
    placement_label = "Placed" if placement_pred == 1 else "Not Placed"

    top_roles = _top_roles(role_model, frame, top_n=3)[0]
    readiness = compute_readiness_score(clean_profile)
    strengths, weak_areas = strengths_and_weaknesses(clean_profile)
    explanation = explanation_text(clean_profile, placement_label, placement_probability, readiness, top_roles)

    return {
        "placement_label": placement_label,
        "placement_status": placement_pred,
        "placement_probability": round(placement_probability, 1),
        "confidence": round(confidence, 1),
        "top_roles": top_roles,
        "primary_role": top_roles[0]["role"],
        "readiness": readiness,
        "strengths": strengths,
        "weak_areas": weak_areas,
        "explanation": explanation,
        "top_factors": _top_factor_notes(clean_profile),
        "profile": clean_profile,
    }


def predict_batch(input_frame: pd.DataFrame) -> pd.DataFrame:
    artifacts = load_artifacts()
    clean_frame = coerce_input_frame(input_frame)
    placement_model = artifacts["placement_model"]
    role_model = artifacts["role_model"]

    placement_predictions = placement_model.predict(clean_frame).astype(int)
    placement_probabilities = _positive_probability(placement_model, clean_frame) * 100
    role_predictions = role_model.predict(clean_frame)
    top_roles = _top_roles(role_model, clean_frame, top_n=3)

    output = input_frame.copy()
    for column in INPUT_FEATURES:
        if column not in output.columns:
            output[column] = clean_frame[column]

    readiness_scores = []
    readiness_levels = []
    strengths_list = []
    weak_areas_list = []
    for record in clean_frame.to_dict(orient="records"):
        readiness = compute_readiness_score(record)
        strengths, weak_areas = strengths_and_weaknesses(record)
        readiness_scores.append(readiness["score"])
        readiness_levels.append(readiness["level"])
        strengths_list.append("; ".join(strengths[:4]))
        weak_areas_list.append("; ".join(weak_areas[:4]))

    output["placement_prediction"] = np.where(placement_predictions == 1, "Placed", "Not Placed")
    output["placement_probability"] = np.round(placement_probabilities, 1)
    output["prediction_confidence"] = np.round(
        np.where(placement_predictions == 1, placement_probabilities, 100 - placement_probabilities),
        1,
    )
    output["recommended_role"] = role_predictions
    output["top_3_roles"] = ["; ".join(f"{item['role']} ({item['probability']}%)" for item in roles) for roles in top_roles]
    output["readiness_score"] = readiness_scores
    output["readiness_level"] = readiness_levels
    output["strengths"] = strengths_list
    output["weak_areas"] = weak_areas_list
    return output
