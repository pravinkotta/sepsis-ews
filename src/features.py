"""Causal hourly feature builder.

Every feature at hour ``t`` may use only rows at hours ``<= t``:

* forward fill looks backwards only;
* rolling windows are trailing;
* a lab's staleness is explicit via ``hours_since``;
* no future row can influence a past feature — asserted by a test that mutates
  the future and checks the past is byte-identical.

The rolling/ffill kernels are NumPy rather than pandas because the cohort is
40k tiny records; per-record pandas overhead dominated the build.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DEMOGRAPHICS, LABEL, LABS, VITALS
from .labels import add_label_columns
from .schema import GoldSchemaError, REQUIRED_GOLD_COLUMNS, validate_raw_frame


def _ffill(values: np.ndarray) -> np.ndarray:
    mask = ~np.isnan(values)
    index = np.where(mask, np.arange(len(values)), -1)
    np.maximum.accumulate(index, out=index)
    out = np.full_like(values, np.nan)
    valid = index >= 0
    out[valid] = values[index[valid]]
    return out


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    n = len(values)
    finite = ~np.isnan(values)
    sums = np.concatenate([[0.0], np.cumsum(np.where(finite, values, 0.0))])
    counts = np.concatenate([[0], np.cumsum(finite.astype(np.int64))])
    starts = np.maximum(np.arange(n) - window + 1, 0)
    ends = np.arange(n) + 1
    window_sums = sums[ends] - sums[starts]
    window_counts = counts[ends] - counts[starts]
    out = np.full(n, np.nan)
    ok = window_counts > 0
    out[ok] = window_sums[ok] / window_counts[ok]
    return out


def _rolling_missing_frac(values: np.ndarray, window: int) -> np.ndarray:
    n = len(values)
    missing = np.isnan(values).astype(np.int64)
    cumulative = np.concatenate([[0], np.cumsum(missing)])
    starts = np.maximum(np.arange(n) - window + 1, 0)
    ends = np.arange(n) + 1
    return (cumulative[ends] - cumulative[starts]) / (ends - starts)


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype="float64")


def build_patient_features(raw: pd.DataFrame, patient_id: str) -> pd.DataFrame:
    """Turn one validated raw patient record into Gold-contract rows.

    The returned frame carries a placeholder ``split`` (all ``train``); the
    builder overwrites it from the patient-level assignment. Unit tests use the
    placeholder directly.
    """
    frame = raw.copy()
    frame.columns = [str(c).strip() for c in frame.columns]
    validate_raw_frame(frame, patient_id=patient_id)

    frame["patient_id"] = patient_id
    frame["icu_hour"] = frame["ICULOS"].astype("int64")
    frame = add_label_columns(frame).sort_values("icu_hour").reset_index(drop=True)

    n = len(frame)
    data: dict[str, object] = {
        "patient_id": np.repeat(patient_id, n),
        "icu_hour": frame["icu_hour"].to_numpy(dtype="int32"),
        "split": np.repeat("train", n),
        "y_label": frame[LABEL].to_numpy(dtype="int8"),
        "is_sepsis_patient": frame["is_sepsis_patient"].to_numpy(dtype="int8"),
        "t_sepsis_est": frame["t_sepsis_est"].array,
        "hours_to_onset": frame["hours_to_onset"].array,
        "onset_censored": frame["onset_censored"].to_numpy(dtype="int8"),
    }

    for column in DEMOGRAPHICS:
        data[column] = _numeric(frame, column).astype("float32")

    hours = frame["icu_hour"].to_numpy(dtype="float64")
    for vital in VITALS:
        series = _numeric(frame, vital)
        data[f"{vital}_last"] = _ffill(series).astype("float32")
        data[f"{vital}_mean_6h"] = _rolling_mean(series, 6).astype("float32")
        data[f"{vital}_mean_24h"] = _rolling_mean(series, 24).astype("float32")
        data[f"{vital}_missing_frac_24h"] = _rolling_missing_frac(series, 24).astype(
            "float32"
        )

    for lab in LABS:
        series = _numeric(frame, lab)
        data[f"{lab}_last"] = _ffill(series).astype("float32")
        observed_hour = np.where(~np.isnan(series), hours, np.nan)
        last_observed = _ffill(observed_hour)
        data[f"{lab}_hours_since"] = (hours - last_observed).astype("float32")

    out = pd.DataFrame(data)
    missing = [c for c in REQUIRED_GOLD_COLUMNS if c not in out.columns]
    if missing:  # pragma: no cover - contract guard
        raise GoldSchemaError(f"{patient_id}: feature builder missed {missing}")
    return out[REQUIRED_GOLD_COLUMNS]


def assert_causal(raw: pd.DataFrame, patient_id: str, *, cutoff_hour: int) -> None:
    """Raise if mutating rows after ``cutoff_hour`` changes features at or before it.

    This is the leakage test in executable form: the future is replaced with
    absurd values, the features are rebuilt, and every cell for hours
    ``<= cutoff_hour`` must be identical. ``SepsisLabel`` is left untouched so
    the label contract stays valid.
    """
    before = build_patient_features(raw, patient_id)
    mutated = raw.copy()
    future = mutated["ICULOS"] > cutoff_hour
    if not future.any():
        return
    for column in VITALS + LABS:
        mutated.loc[future, column] = 9_999.0
    after = build_patient_features(mutated, patient_id)

    past_before = before[before["icu_hour"] <= cutoff_hour].reset_index(drop=True)
    past_after = after[after["icu_hour"] <= cutoff_hour].reset_index(drop=True)
    pd.testing.assert_frame_equal(past_before, past_after, check_exact=True)


def build_feature_matrix(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concat per-patient Gold frames, preserving dtypes and column order."""
    if not frames:
        raise GoldSchemaError("no patient frames to concatenate")
    combined = pd.concat(frames, ignore_index=True)
    # A concat of empty-slice frames can upcast ints; re-assert the contract.
    for column in ("icu_hour", "y_label", "is_sepsis_patient", "onset_censored"):
        combined[column] = combined[column].astype(
            "int32" if column == "icu_hour" else "int8"
        )
    for column in ("t_sepsis_est", "hours_to_onset"):
        combined[column] = combined[column].astype("Int32")
    return combined[REQUIRED_GOLD_COLUMNS]
