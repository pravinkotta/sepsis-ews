"""Synthetic Gold tables for pipeline tests only — never a measured result."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import DEMOGRAPHICS
from src.schema import GOLD_FEATURE_COLUMNS, assign_patient_splits


def make_gold_frame(
    n_patients: int = 60,
    n_hours: int = 20,
    n_sepsis: int = 12,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ids = [f"p{i:03d}" for i in range(n_patients)]
    flags = {pid: i < n_sepsis for i, pid in enumerate(ids)}
    splits = assign_patient_splits(ids, sepsis_by_patient=flags, seed=seed + 1)

    rows = []
    for pid in ids:
        onset = 16 if flags[pid] else None
        for hour in range(1, n_hours + 1):
            row = {column: float(rng.normal()) for column in GOLD_FEATURE_COLUMNS}
            row.update({"patient_id": pid, "icu_hour": hour, "split": splits[pid]})
            row["y_label"] = int(bool(flags[pid]) and hour >= onset - 6)
            row["is_sepsis_patient"] = int(flags[pid])
            row["t_sepsis_est"] = onset
            row["hours_to_onset"] = (onset - hour) if flags[pid] else None
            row["onset_censored"] = 0
            for column in DEMOGRAPHICS:
                row[column] = float(rng.normal())
            row["Age"] = 65.0
            row["Gender"] = 1.0
            row["Unit1"] = 1.0
            row["Unit2"] = 0.0
            row["HospAdmTime"] = -2.0
            row["ICULOS"] = float(hour)

            signal = 1.0 if (flags[pid] and onset - 12 <= hour < onset) else 0.0
            row["HR_last"] = 80 + 30 * signal + rng.normal(0, 3)
            row["HR_mean_24h"] = row["HR_last"] + rng.normal(0, 1)
            row["Lactate_last"] = 1 + 5 * signal + rng.normal(0, 0.4)
            row["Lactate_hours_since"] = float(rng.integers(0, 3))
            rows.append(row)

    frame = pd.DataFrame(rows)
    frame["t_sepsis_est"] = pd.array(frame["t_sepsis_est"], dtype="Int32")
    frame["hours_to_onset"] = pd.array(frame["hours_to_onset"], dtype="Int32")
    return frame
