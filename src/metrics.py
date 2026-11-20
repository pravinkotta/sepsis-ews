"""Shared metric helpers for the LightGBM and GRU comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

EPS = 1e-6


def pre_onset_mask(frame: pd.DataFrame) -> np.ndarray:
    non_sepsis = frame["is_sepsis_patient"].to_numpy() == 0
    hours = frame["hours_to_onset"].fillna(1).to_numpy()
    return non_sepsis | (hours > 0)


def calibration_slope(p: np.ndarray, y: np.ndarray) -> float:
    clipped = np.clip(p, EPS, 1 - EPS)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    model.fit(logit, y)
    return float(model.coef_[0, 0])


def ranking_metrics(frame: pd.DataFrame, p: np.ndarray) -> dict:
    mask = pre_onset_mask(frame)
    y = frame["y_label"].to_numpy()[mask]
    probabilities = np.asarray(p)[mask]
    if len(np.unique(y)) < 2:
        raise ValueError("pre-onset subset has only one class")
    return {
        "auroc": round(float(roc_auc_score(y, probabilities)), 6),
        "auprc": round(float(average_precision_score(y, probabilities)), 6),
        "prevalence": round(float(y.mean()), 6),
        "n_hours": int(len(y)),
    }
