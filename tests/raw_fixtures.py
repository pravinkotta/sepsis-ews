"""Clearly-synthetic raw records for parser/feature tests only.

These are never committed as data and never used for training (C1). They exist
so the verifier and feature builder can be tested offline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import EXPECTED_SETS, RAW_COLUMNS


def make_raw(hours: int = 8, sepsis_from: int | None = None) -> pd.DataFrame:
    rows = []
    for hour in range(1, hours + 1):
        row = {column: np.nan for column in RAW_COLUMNS}
        row.update(
            {
                "HR": 80 + 10 * hour,
                "O2Sat": 97.0,
                "Temp": 37.0,
                "SBP": 120.0,
                "MAP": 80.0,
                "DBP": 70.0,
                "Resp": 18.0,
                "Age": 65.0,
                "Gender": 1,
                "Unit1": 1,
                "Unit2": 0,
                "HospAdmTime": -2.0,
                "ICULOS": float(hour),
                "SepsisLabel": int(sepsis_from is not None and hour >= sepsis_from),
            }
        )
        if hour in (1, 4):
            row["Lactate"] = 2.0 + hour
        rows.append(row)
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def write_raw_cohort(root, specs) -> None:
    """specs: iterable of (set_name, patient_id, hours, sepsis_from)."""
    for set_name, patient_id, hours, sepsis_from in specs:
        directory = root / EXPECTED_SETS[set_name]
        directory.mkdir(parents=True, exist_ok=True)
        make_raw(hours, sepsis_from).to_csv(
            directory / f"{patient_id}.psv", sep="|", index=False, na_rep="NaN"
        )
