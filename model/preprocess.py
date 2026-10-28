from __future__ import annotations

from typing import Any, Dict

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils.config import (
    CATEGORICAL_FEATURES,
    COLLEGE_TIER_OPTIONS,
    DEFAULT_PROFILE,
    DEGREE_OPTIONS,
    DOMAIN_OPTIONS,
    GENDER_OPTIONS,
    INPUT_FEATURES,
    NUMERIC_FEATURES,
    SPECIALIZATION_OPTIONS,
)


VALID_OPTIONS = {
    "gender": GENDER_OPTIONS,
    "degree_type": DEGREE_OPTIONS,
    "specialization": SPECIALIZATION_OPTIONS,
    "college_tier": COLLEGE_TIER_OPTIONS,
    "preferred_domain": DOMAIN_OPTIONS,
}


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", make_one_hot_encoder(), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def coerce_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    clean = DEFAULT_PROFILE.copy()
    clean.update({k: v for k, v in profile.items() if k in INPUT_FEATURES})

    for column in NUMERIC_FEATURES:
        try:
            value = clean.get(column, DEFAULT_PROFILE[column])
            if value in (None, ""):
                value = DEFAULT_PROFILE[column]
            clean[column] = float(value)
        except (TypeError, ValueError):
            clean[column] = float(DEFAULT_PROFILE[column])

    for column in CATEGORICAL_FEATURES:
        value = str(clean.get(column, DEFAULT_PROFILE[column])).strip()
        if value not in VALID_OPTIONS[column]:
            value = DEFAULT_PROFILE[column]
        clean[column] = value

    return clean


def coerce_input_frame(data: Any) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        frame = data.copy()
    elif isinstance(data, list):
        frame = pd.DataFrame(data)
    else:
        frame = pd.DataFrame([data])

    for column, default in DEFAULT_PROFILE.items():
        if column not in frame.columns:
            frame[column] = default

    frame = frame[INPUT_FEATURES].copy()

    for column in NUMERIC_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(DEFAULT_PROFILE[column])

    for column in CATEGORICAL_FEATURES:
        frame[column] = frame[column].astype(str).str.strip()
        frame.loc[~frame[column].isin(VALID_OPTIONS[column]), column] = DEFAULT_PROFILE[column]

    return frame
