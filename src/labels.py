"""Sepsis-3 onset derivation from the released PhysioNet 2019 labels.

The dataset ships ``SepsisLabel``, not ``t_sepsis``. The challenge defines:

    SepsisLabel = 1  if t >= t_sepsis - 6
    SepsisLabel = 0  otherwise (and always 0 for non-sepsis patients)

So the first positive hour is an estimate of ``t_sepsis - 6``:

    t_sepsis_est = first_positive_hour + 6

If the record *begins* with a positive label, the true onset is left-censored:
the patient entered the record inside the 6-hour window. Such patients are
flagged ``onset_censored`` and excluded from lead-time claims (they are still
counted for alert burden and for training).

Nothing here invents a label. The Sepsis-3 clinical definition (suspected
infection + SOFA rise >= 2) is realised upstream by the challenge organisers;
this module only converts their released label into an onset time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .config import LABEL, LABEL_SHIFT_HOURS


class LabelError(ValueError):
    """Raised when a patient record violates the released label contract."""


@dataclass(frozen=True)
class Onset:
    sepsis_patient: bool
    t_sepsis_est: int | None
    first_positive_hour: int | None
    censored: bool


def derive_onset(sepsis_labels: Sequence[int], icu_hours: Sequence[int]) -> Onset:
    """Return the onset estimate for one patient.

    Labels must be non-decreasing (0*1*). A 1 -> 0 transition means the file is
    not a valid challenge record and must fail loudly rather than be repaired.
    """
    labels = np.asarray(sepsis_labels, dtype=np.int64)
    hours = np.asarray(icu_hours, dtype=np.int64)
    if labels.shape != hours.shape:
        raise LabelError("labels and icu_hours must have the same length")
    if labels.size == 0:
        raise LabelError("empty patient record")
    if not np.isin(labels, (0, 1)).all():
        raise LabelError("SepsisLabel must contain only 0 and 1")
    if np.any(np.diff(labels) < 0):
        raise LabelError("SepsisLabel is not monotonic (found a 1 -> 0 transition)")
    if np.any(np.diff(hours) != 1):
        raise LabelError("icu_hour must increase by exactly 1")

    positives = np.flatnonzero(labels == 1)
    if positives.size == 0:
        return Onset(False, None, None, False)

    first_idx = int(positives[0])
    first_hour = int(hours[first_idx])
    t_est = first_hour + LABEL_SHIFT_HOURS
    censored = first_idx == 0
    return Onset(True, t_est, first_hour, censored)


def add_label_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach derived label columns to a per-patient hourly frame.

    Required input columns: ``patient_id``, ``icu_hour``, ``SepsisLabel``.
    The output adds: ``is_sepsis_patient``, ``t_sepsis_est``,
    ``hours_to_onset``, ``onset_censored``.
    """
    required = {"patient_id", "icu_hour", "SepsisLabel"}
    missing = required - set(frame.columns)
    if missing:
        raise LabelError(f"missing required columns: {sorted(missing)}")

    out = frame.sort_values(["patient_id", "icu_hour"]).copy()
    n = len(out)

    # Fast path: the builder calls this once per patient. Avoid the groupby
    # machinery (and its iloc copies) for the overwhelmingly common case.
    if out["patient_id"].nunique(dropna=False) == 1:
        onset = derive_onset(
            out[LABEL].to_numpy(), out["icu_hour"].to_numpy()
        )
        out["is_sepsis_patient"] = np.int8(onset.sepsis_patient)
        out["onset_censored"] = np.int8(onset.censored)
        out["t_sepsis_est"] = pd.array(
            [onset.t_sepsis_est] * n if onset.sepsis_patient else [None] * n,
            dtype="Int64",
        )
        out["hours_to_onset"] = (
            out["t_sepsis_est"] - out["icu_hour"]
        ).astype("Int64")
        return out

    is_sepsis = np.zeros(n, dtype=np.int8)
    t_est = np.full(n, -1, dtype=np.int64)
    censored = np.zeros(n, dtype=np.int8)

    for pid, idx in out.groupby("patient_id", sort=False).indices.items():
        block = out.iloc[idx]
        onset = derive_onset(block["SepsisLabel"].to_numpy(), block["icu_hour"].to_numpy())
        is_sepsis[idx] = int(onset.sepsis_patient)
        censored[idx] = int(onset.censored)
        if onset.t_sepsis_est is not None:
            t_est[idx] = onset.t_sepsis_est

    out["is_sepsis_patient"] = is_sepsis
    out["onset_censored"] = censored
    out["t_sepsis_est"] = pd.array(t_est, dtype="Int64")
    out.loc[out["is_sepsis_patient"] == 0, "t_sepsis_est"] = pd.NA
    out["hours_to_onset"] = (out["t_sepsis_est"] - out["icu_hour"]).astype("Int64")
    return out


def build_patient_index(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per patient: length, onset, censoring, and prevalence inputs.

    ``frame`` must already carry the columns produced by ``add_label_columns``.
    """
    if "t_sepsis_est" not in frame.columns:
        frame = add_label_columns(frame)
    grouped = frame.groupby("patient_id", sort=True)
    index = grouped.agg(
        n_hours=("icu_hour", "size"),
        first_hour=("icu_hour", "min"),
        last_hour=("icu_hour", "max"),
        is_sepsis_patient=("is_sepsis_patient", "max"),
        onset_censored=("onset_censored", "max"),
        t_sepsis_est=("t_sepsis_est", "first"),
    ).reset_index()
    return index
