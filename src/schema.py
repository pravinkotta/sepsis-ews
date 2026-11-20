"""Data contracts: raw PhysioNet records, the Gold hourly table, and splits.

The rule this module enforces: **a label can never become a feature, and a
patient can never appear in two splits.** Everything else is a dtype or
coverage check that fails loudly with the offending patient id.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from .config import (
    DEMOGRAPHICS,
    FORBIDDEN_FEATURE_NAMES,
    LABEL,
    LABS,
    RAW_COLUMNS,
    SPLIT_RATIOS,
    SPLIT_SEED,
    SPLITS,
    VITALS,
)


class GoldSchemaError(ValueError):
    """Raised when the Gold table violates its contract."""


GOLD_KEY_COLUMNS = ["patient_id", "icu_hour", "split"]
GOLD_LABEL_COLUMNS = [
    "y_label",
    "is_sepsis_patient",
    "t_sepsis_est",
    "hours_to_onset",
    "onset_censored",
]


def _gold_feature_columns() -> list[str]:
    cols: list[str] = []
    for vital in VITALS:
        cols += [
            f"{vital}_last",
            f"{vital}_mean_6h",
            f"{vital}_mean_24h",
            f"{vital}_missing_frac_24h",
        ]
    for lab in LABS:
        cols += [f"{lab}_last", f"{lab}_hours_since"]
    return cols


GOLD_FEATURE_COLUMNS = _gold_feature_columns()
REQUIRED_GOLD_COLUMNS = (
    GOLD_KEY_COLUMNS + GOLD_LABEL_COLUMNS + DEMOGRAPHICS + GOLD_FEATURE_COLUMNS
)


def validate_raw_frame(frame: pd.DataFrame, *, patient_id: str) -> None:
    """Validate one raw ``.psv`` record before it enters Bronze."""
    missing = [c for c in RAW_COLUMNS if c not in frame.columns]
    if missing:
        raise GoldSchemaError(f"{patient_id}: missing raw columns {missing}")
    if len(frame) == 0:
        raise GoldSchemaError(f"{patient_id}: empty record")

    hours = frame["ICULOS"].to_numpy()
    if not np.issubdtype(hours.dtype, np.number):
        raise GoldSchemaError(f"{patient_id}: ICULOS is not numeric")
    if np.isnan(hours).any():
        raise GoldSchemaError(f"{patient_id}: ICULOS contains NaN")
    # Records may begin at ICU hour 2, 3, ... (the stay started before the
    # released record). Onset and labels are in ICULOS units, so we preserve
    # the starting hour and require contiguity, not a start at 1.
    if hours[0] < 1:
        raise GoldSchemaError(f"{patient_id}: ICULOS must be >= 1, got {hours[0]}")
    if np.any(np.diff(hours) != 1):
        raise GoldSchemaError(f"{patient_id}: ICULOS is not contiguous")

    labels = frame[LABEL].to_numpy()
    if not np.isin(labels, (0, 1)).all():
        raise GoldSchemaError(f"{patient_id}: {LABEL} must be 0/1")
    if np.any(np.diff(labels) < 0):
        raise GoldSchemaError(f"{patient_id}: {LABEL} is not monotonic")


def validate_gold_frame(frame: pd.DataFrame) -> None:
    """Validate the Gold hourly table. Raises on the first contract violation."""
    missing = [c for c in REQUIRED_GOLD_COLUMNS if c not in frame.columns]
    if missing:
        raise GoldSchemaError(f"missing Gold columns: {missing}")
    if len(frame) == 0:
        raise GoldSchemaError("Gold table is empty")
    if frame["patient_id"].isna().any():
        raise GoldSchemaError("patient_id contains nulls")
    if frame.duplicated(["patient_id", "icu_hour"]).any():
        dup = frame.loc[
            frame.duplicated(["patient_id", "icu_hour"], keep=False), "patient_id"
        ].unique()[:3]
        raise GoldSchemaError(f"duplicate (patient_id, icu_hour) rows, e.g. {list(dup)}")

    if not set(frame["split"].unique()).issubset(set(SPLITS)):
        raise GoldSchemaError(f"split must be one of {SPLITS}")
    n_splits = frame.groupby("patient_id")["split"].nunique()
    leaked = n_splits[n_splits > 1].index.tolist()[:3]
    if leaked:
        raise GoldSchemaError(f"patients in more than one split, e.g. {leaked}")

    for pid, block in frame.groupby("patient_id", sort=False):
        block = block.sort_values("icu_hour")
        hours = block["icu_hour"].to_numpy()
        if hours[0] < 1 or np.any(np.diff(hours) != 1):
            raise GoldSchemaError(f"{pid}: icu_hour must be contiguous from a positive hour")

        labels = block["y_label"].to_numpy()
        if not np.isin(labels, (0, 1)).all():
            raise GoldSchemaError(f"{pid}: y_label must be 0/1")
        if np.any(np.diff(labels) < 0):
            raise GoldSchemaError(f"{pid}: y_label is not monotonic")

        sepsis = int(block["is_sepsis_patient"].iloc[0])
        if block["is_sepsis_patient"].nunique() != 1:
            raise GoldSchemaError(f"{pid}: is_sepsis_patient varies within the stay")
        if block["onset_censored"].nunique() != 1:
            raise GoldSchemaError(f"{pid}: onset_censored varies within the stay")
        censored = int(block["onset_censored"].iloc[0])
        if censored not in (0, 1):
            raise GoldSchemaError(f"{pid}: onset_censored must be 0/1")

        t_values = block["t_sepsis_est"].dropna().unique()
        if sepsis == 0:
            if len(t_values) != 0:
                raise GoldSchemaError(f"{pid}: non-sepsis patient has t_sepsis_est")
            if labels.any():
                raise GoldSchemaError(f"{pid}: non-sepsis patient has positive y_label")
            if block["hours_to_onset"].notna().any():
                raise GoldSchemaError(f"{pid}: non-sepsis patient has hours_to_onset")
        else:
            if len(t_values) != 1:
                raise GoldSchemaError(f"{pid}: sepsis patient needs one t_sepsis_est")
            t_est = int(t_values[0])
            if t_est < 7:
                raise GoldSchemaError(f"{pid}: t_sepsis_est must be >= 7")
            expected = (hours >= t_est - 6).astype(int)
            if not np.array_equal(labels, expected):
                raise GoldSchemaError(f"{pid}: y_label disagrees with t_sepsis_est")
            expected_hours = t_est - hours
            observed = block["hours_to_onset"].to_numpy()
            if not np.array_equal(observed, expected_hours):
                raise GoldSchemaError(f"{pid}: hours_to_onset disagrees with t_sepsis_est")
            if censored and int(labels[0]) != 1:
                raise GoldSchemaError(f"{pid}: censored onset must start positive")

    feature_values = frame[GOLD_FEATURE_COLUMNS].to_numpy(dtype="float64")
    if np.isinf(feature_values).any():
        raise GoldSchemaError("feature columns contain infinite values")


def assign_patient_splits(
    patient_ids: Sequence[str],
    *,
    sepsis_by_patient: Mapping[str, bool] | None = None,
    ratios: tuple[float, float, float] = SPLIT_RATIOS,
    seed: int = SPLIT_SEED,
) -> dict[str, str]:
    """Assign every patient to exactly one split, optionally stratified.

    Stratifying on sepsis status keeps the ~7% positive class represented in
    validation and test. The function asserts coverage and disjointness before
    returning, so a silent hole in the split is impossible.
    """
    ids = sorted(set(patient_ids))
    if len(ids) != len(patient_ids):
        raise GoldSchemaError("patient_ids contains duplicates")
    if len(ids) < 3:
        raise GoldSchemaError("need at least 3 patients to build three splits")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise GoldSchemaError("split ratios must sum to 1")

    rng = np.random.default_rng(seed)

    def split_one(group: list[str], assignment: dict[str, str]) -> None:
        group = list(group)
        rng.shuffle(group)
        n = len(group)
        if n >= 3:
            n_train = max(1, int(round(n * ratios[0])))
            n_val = max(1, int(round(n * ratios[1])))
            if n_train + n_val >= n:
                n_train, n_val = n - 2, 1
        else:
            # A stratum too small to fill three cells: train + test only.
            n_train, n_val = 1, 0
        for i, pid in enumerate(group):
            if i < n_train:
                assignment[pid] = "train"
            elif i < n_train + n_val:
                assignment[pid] = "val"
            else:
                assignment[pid] = "test"

    assignment: dict[str, str] = {}
    if sepsis_by_patient is None:
        split_one(ids, assignment)
    else:
        missing = [pid for pid in ids if pid not in sepsis_by_patient]
        if missing:
            raise GoldSchemaError(f"missing sepsis flag for patients: {missing[:3]}")
        for flag in (False, True):
            group = [pid for pid in ids if bool(sepsis_by_patient[pid]) is flag]
            if group:
                split_one(group, assignment)

    if set(assignment) != set(ids):
        raise GoldSchemaError("split assignment does not cover every patient")

    # Tiny strata (or a tiny cohort) may leave a split empty. Rebalance from the
    # largest split rather than silently returning two splits.
    for missing in sorted(set(SPLITS) - set(assignment.values())):
        counts = {s: list(assignment.values()).count(s) for s in SPLITS}
        donor = max(counts, key=counts.get)
        if counts[donor] < 2:
            raise GoldSchemaError(
                f"cannot populate split {missing!r} from {len(ids)} patients"
            )
        candidate = next(pid for pid, value in assignment.items() if value == donor)
        assignment[candidate] = missing

    if set(assignment.values()) != set(SPLITS):
        raise GoldSchemaError("split assignment does not use all three splits")
    return assignment


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return the model feature list, refusing to leak labels or metadata."""
    forbidden = set(GOLD_KEY_COLUMNS) | set(GOLD_LABEL_COLUMNS) | set(FORBIDDEN_FEATURE_NAMES)
    cols = [c for c in frame.columns if c not in forbidden]
    leaked = sorted(set(cols) & forbidden)
    if leaked:
        raise GoldSchemaError(f"label/metadata columns leaked into features: {leaked}")
    if not cols:
        raise GoldSchemaError("no feature columns remain after exclusions")
    return cols
