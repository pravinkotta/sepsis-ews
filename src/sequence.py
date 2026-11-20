"""Sequence tensors for the GRU challenger.

Temporal inputs are the 34 `_last` values already computed causally in Silver.
Static inputs are age/sex/unit/admit-time plus the current ICU hour.

Imputation and scaling statistics are fitted on the **training split only** and
stored with the model. Windows are left-padded by repeating the first available
hour, which is past data — never a future row.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from .config import LABS, VITALS

WINDOW = 24
TEMPORAL_FEATURES = [f"{v}_last" for v in VITALS] + [f"{lab}_last" for lab in LABS]
STATIC_FEATURES = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime"]
STATIC_WITH_HOUR = STATIC_FEATURES + ["ICULOS"]
META_COLUMNS = [
    "patient_id",
    "icu_hour",
    "y_label",
    "t_sepsis_est",
    "hours_to_onset",
    "is_sepsis_patient",
    "onset_censored",
]


class SequenceError(ValueError):
    """Raised when a split cannot be turned into windows."""


@dataclass
class SequenceBatch:
    X: np.ndarray          # (n, window, 34)
    S: np.ndarray          # (n, 6)
    meta: pd.DataFrame     # labels + identifying columns

    def __len__(self) -> int:
        return len(self.meta)


@dataclass
class SequenceStats:
    temporal_median: np.ndarray
    temporal_mean: np.ndarray
    temporal_std: np.ndarray
    static_median: np.ndarray
    static_mean: np.ndarray
    static_std: np.ndarray
    window: int = WINDOW

    def as_dict(self) -> dict:
        return {
            "window": self.window,
            "temporal_features": TEMPORAL_FEATURES,
            "static_features": STATIC_WITH_HOUR,
            "temporal_median": self.temporal_median.tolist(),
            "temporal_mean": self.temporal_mean.tolist(),
            "temporal_std": self.temporal_std.tolist(),
            "static_median": self.static_median.tolist(),
            "static_mean": self.static_mean.tolist(),
            "static_std": self.static_std.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "SequenceStats":
        return cls(
            temporal_median=np.asarray(payload["temporal_median"], dtype=np.float64),
            temporal_mean=np.asarray(payload["temporal_mean"], dtype=np.float64),
            temporal_std=np.asarray(payload["temporal_std"], dtype=np.float64),
            static_median=np.asarray(payload["static_median"], dtype=np.float64),
            static_mean=np.asarray(payload["static_mean"], dtype=np.float64),
            static_std=np.asarray(payload["static_std"], dtype=np.float64),
            window=int(payload.get("window", WINDOW)),
        )


def _fit_matrix(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    median = np.nanmedian(values, axis=0)
    median = np.where(np.isnan(median), 0.0, median)
    filled = np.where(np.isnan(values), median, values)
    mean = filled.mean(axis=0)
    std = filled.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return median, mean, std


def fit_stats(gold: pd.DataFrame) -> SequenceStats:
    temporal = gold[TEMPORAL_FEATURES].to_numpy(dtype=np.float64)
    static = gold[STATIC_WITH_HOUR].to_numpy(dtype=np.float64)
    tm, tmean, tstd = _fit_matrix(temporal)
    sm, smean, sstd = _fit_matrix(static)
    return SequenceStats(tm, tmean, tstd, sm, smean, sstd)


def _scale(values: np.ndarray, median, mean, std) -> np.ndarray:
    values = np.where(np.isnan(values), median, values)
    return np.clip((values - mean) / std, -5.0, 5.0)


def build_sequences(
    gold: pd.DataFrame,
    split: str,
    *,
    stats: SequenceStats,
    window: int = WINDOW,
    max_per_patient: int | None = None,
    seed: int = 42,
) -> SequenceBatch:
    subset = gold[gold["split"] == split]
    if subset.empty:
        raise SequenceError(f"split {split!r} is empty")
    missing = [c for c in set(TEMPORAL_FEATURES + STATIC_WITH_HOUR + META_COLUMNS) if c not in subset.columns]
    if missing:
        raise SequenceError(f"Gold table is missing sequence columns: {sorted(missing)}")

    rng = np.random.default_rng(seed)
    x_parts: list[np.ndarray] = []
    s_parts: list[np.ndarray] = []
    meta_parts: list[pd.DataFrame] = []

    for _, block in subset.sort_values(["patient_id", "icu_hour"]).groupby(
        "patient_id", sort=False
    ):
        n = len(block)
        if n == 0:
            continue
        temporal = _scale(
            block[TEMPORAL_FEATURES].to_numpy(dtype=np.float64),
            stats.temporal_median,
            stats.temporal_mean,
            stats.temporal_std,
        )
        static = _scale(
            block[STATIC_WITH_HOUR].to_numpy(dtype=np.float64),
            stats.static_median,
            stats.static_mean,
            stats.static_std,
        )
        padded = np.vstack([np.repeat(temporal[:1], window - 1, axis=0), temporal])
        windows = sliding_window_view(padded, window, axis=0).transpose(0, 2, 1)

        if max_per_patient is not None and n > max_per_patient:
            indices = np.sort(rng.choice(n, size=max_per_patient, replace=False))
        else:
            indices = np.arange(n)

        x_parts.append(np.ascontiguousarray(windows[indices], dtype=np.float32))
        s_parts.append(np.ascontiguousarray(static[indices], dtype=np.float32))
        meta_parts.append(block.iloc[indices][META_COLUMNS].reset_index(drop=True))

    if not x_parts:
        raise SequenceError(f"no windows built for split {split!r}")

    return SequenceBatch(
        X=np.concatenate(x_parts, axis=0),
        S=np.concatenate(s_parts, axis=0),
        meta=pd.concat(meta_parts, ignore_index=True),
    )
