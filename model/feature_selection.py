from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder

from utils.config import CATEGORICAL_FEATURES


def clean_feature_name(name: str) -> str:
    if name.startswith("num__"):
        return name.replace("num__", "")
    if name.startswith("cat__"):
        raw_name = name.replace("cat__", "")
        for column in sorted(CATEGORICAL_FEATURES, key=len, reverse=True):
            prefix = f"{column}_"
            if raw_name.startswith(prefix):
                return f"{column} = {raw_name[len(prefix):]}"
        return raw_name
    return name


def get_feature_names(preprocessor: Any) -> List[str]:
    try:
        return [clean_feature_name(name) for name in preprocessor.get_feature_names_out()]
    except Exception:
        names: List[str] = []
        for transformer_name, transformer, columns in preprocessor.transformers_:
            if transformer_name == "remainder":
                continue
            if hasattr(transformer, "get_feature_names_out"):
                raw = transformer.get_feature_names_out(columns)
                names.extend([clean_feature_name(f"{transformer_name}__{name}") for name in raw])
            else:
                names.extend(columns)
        return names


def _scale(values: Iterable[float]) -> np.ndarray:
    array = np.array(list(values), dtype=float)
    if array.size == 0:
        return array
    max_value = np.nanmax(array)
    min_value = np.nanmin(array)
    if not np.isfinite(max_value) or max_value == min_value:
        return np.zeros_like(array)
    return (array - min_value) / (max_value - min_value)


def build_feature_report(
    pipeline: Any,
    X_train: pd.DataFrame,
    y_train: Iterable[Any],
    top_n: int = 35,
    random_state: int = 42,
) -> Dict[str, Any]:
    preprocessor = pipeline.named_steps["preprocessor"]
    selector = pipeline.named_steps["selector"]

    encoded = preprocessor.transform(X_train)
    feature_names = get_feature_names(preprocessor)
    selector_scores = np.nan_to_num(selector.scores_, nan=0.0, posinf=0.0, neginf=0.0)
    selected_mask = selector.get_support()

    y_values = np.array(list(y_train))
    if y_values.dtype.kind not in {"i", "u", "f", "b"}:
        y_values = LabelEncoder().fit_transform(y_values)

    tree = ExtraTreesClassifier(
        n_estimators=220,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=1,
    )
    tree.fit(encoded, y_values)
    tree_importance = tree.feature_importances_

    combined_score = 0.55 * _scale(selector_scores) + 0.45 * _scale(tree_importance)
    rows = []
    for name, selected, kbest, importance, combined in zip(
        feature_names,
        selected_mask,
        selector_scores,
        tree_importance,
        combined_score,
    ):
        rows.append(
            {
                "feature": name,
                "selected": bool(selected),
                "kbest_score": round(float(kbest), 5),
                "tree_importance": round(float(importance), 6),
                "combined_score": round(float(combined), 6),
            }
        )

    rankings = sorted(rows, key=lambda item: item["combined_score"], reverse=True)
    selected_features = [row["feature"] for row in rows if row["selected"]]
    dropped_features = [row["feature"] for row in rows if not row["selected"]]

    return {
        "selected_features": selected_features,
        "dropped_features": dropped_features,
        "rankings": rankings,
        "top_features": rankings[:top_n],
        "selection_method": "SelectKBest(f_classif) plus ExtraTrees feature importance",
        "selected_count": len(selected_features),
        "dropped_count": len(dropped_features),
    }
