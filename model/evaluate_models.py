from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict

import pandas as pd

from model.generate_data import create_hero_asset, save_dataset
from model.train_placement_model import train_placement_model
from model.train_role_model import train_role_model
from utils.config import ARTIFACT_FILES, DATASET_PATH, DATASET_ROWS, RANDOM_STATE, ensure_directories


def artifacts_exist() -> bool:
    required = [
        ARTIFACT_FILES["placement_model"],
        ARTIFACT_FILES["role_model"],
        ARTIFACT_FILES["selected_features_placement"],
        ARTIFACT_FILES["selected_features_role"],
        ARTIFACT_FILES["placement_metrics"],
        ARTIFACT_FILES["role_metrics"],
    ]
    return DATASET_PATH.exists() and all(path.exists() for path in required)


def ensure_project_ready(force: bool = False, n_rows: int = DATASET_ROWS) -> Dict[str, Any]:
    ensure_directories()

    if force or not DATASET_PATH.exists():
        df = save_dataset(n_rows=n_rows, random_state=RANDOM_STATE)
    else:
        df = pd.read_csv(DATASET_PATH)

    create_hero_asset()

    if force or not artifacts_exist():
        placement_metrics = train_placement_model(DATASET_PATH)
        role_metrics = train_role_model(DATASET_PATH)
    else:
        with open(ARTIFACT_FILES["placement_metrics"], "r", encoding="utf-8") as file:
            placement_metrics = json.load(file)
        with open(ARTIFACT_FILES["role_metrics"], "r", encoding="utf-8") as file:
            role_metrics = json.load(file)

    metadata = {
        "project": "SMART HIRE - Student Placement Prediction System",
        "dataset_path": str(DATASET_PATH),
        "dataset_rows": int(len(df)),
        "placement_positive_rate": round(float(df["placement_status"].mean()), 4),
        "roles": sorted(df["recommended_role"].unique().tolist()),
        "placement_best_model": placement_metrics["best_model"],
        "role_best_model": role_metrics["best_model"],
        "last_verified_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(ARTIFACT_FILES["metadata"], "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    return {
        "metadata": metadata,
        "placement_metrics": placement_metrics,
        "role_metrics": role_metrics,
    }


if __name__ == "__main__":
    ensure_project_ready(force=True)
