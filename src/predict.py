"""Load the trained LightGBM + calibrator and score one patient-hour.

Per-alert drivers use LightGBM's exact `pred_contrib` values (the same SHAP
values the training audit checked for additivity), so the explanation matches
the deployed model.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import ARTIFACTS


class ModelNotConfigured(RuntimeError):
    """The trained model artefacts are absent or incomplete."""


FACT_FEATURES = {
    "HR_last": "HR",
    "O2Sat_last": "O2Sat",
    "Temp_last": "Temp",
    "SBP_last": "SBP",
    "MAP_last": "MAP",
    "DBP_last": "DBP",
    "Resp_last": "Resp",
    "EtCO2_last": "EtCO2",
    "Lactate_last": "lactate",
    "Creatinine_last": "creatinine",
    "WBC_last": "WBC",
    "Platelets_last": "platelets",
    "FiO2_last": "FiO2",
    "BUN_last": "BUN",
}


def features_to_matrix(features: dict, columns: list[str]) -> pd.DataFrame:
    """Build a float32 matrix, turning None/missing labs into NaN.

    The store cleans NaN to None for JSON; without this coercion pandas infers
    object dtype for all-missing columns and LightGBM refuses the matrix. A
    NumPy array is much faster than a per-column `to_numeric` on the hot path.
    """
    values = np.asarray(
        [features.get(column, np.nan) for column in columns], dtype=np.float32
    ).reshape(1, -1)
    return pd.DataFrame(values, columns=columns)


def _safe(value):
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return round(value, 3)
    return value


class ModelPredictor:
    def __init__(self, artifacts_dir: Path = ARTIFACTS) -> None:
        artifacts_dir = Path(artifacts_dir)
        paths = {
            "model": artifacts_dir / "model.txt",
            "calibrator": artifacts_dir / "calibrator.joblib",
            "meta": artifacts_dir / "model_meta.json",
        }
        missing = [str(path) for path in paths.values() if not path.exists()]
        if missing:
            raise ModelNotConfigured(
                f"missing model artefacts: {missing}. Run `python -m src.train`."
            )
        meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
        self.features: list[str] = meta["features"]
        self.threshold: float = float(meta["threshold"])
        self.best_iteration = meta.get("best_iteration")
        self.booster = lgb.Booster(model_file=str(paths["model"]))
        self.calibrator = joblib.load(paths["calibrator"])

    def predict(self, features: dict) -> dict:
        missing = [name for name in self.features if name not in features]
        if missing:
            raise ModelNotConfigured(
                f"patient state is missing {len(missing)} model features, e.g. {missing[:5]}"
            )
        matrix = features_to_matrix(features, self.features)
        # One call gives both the explanation and the raw margin: the last
        # contribution is the base value, the rest sum to the logit.
        full = self.booster.predict(matrix, pred_contrib=True)[0]
        contributions = full[:-1]
        margin = float(full[-1] + contributions.sum())
        # Contributions are log-odds; the calibrator was fitted on probabilities.
        probability = 1.0 / (1.0 + math.exp(-margin))
        risk = float(self.calibrator.predict(np.array([probability]))[0])
        order = np.argsort(np.abs(contributions))[::-1][:5]
        drivers = [
            {
                "feature": self.features[index],
                "contribution": round(float(contributions[index]), 6),
                "direction": "increases" if contributions[index] >= 0 else "decreases",
                "value": _safe(features.get(self.features[index])),
            }
            for index in order
        ]
        return {
            "risk": round(risk, 6),
            "threshold": self.threshold,
            "top_drivers": drivers,
            "model": "lightgbm",
            "best_iteration": self.best_iteration,
            "facts": {
                label: _safe(features.get(name))
                for name, label in FACT_FEATURES.items()
                if features.get(name) is not None
            },
        }
