import numpy as np
import pandas as pd
import pytest

from src.config import DEMOGRAPHICS
from src.schema import (
    GOLD_FEATURE_COLUMNS,
    GoldSchemaError,
    assign_patient_splits,
    feature_columns,
    validate_gold_frame,
)


def make_gold() -> pd.DataFrame:
    rows = []
    meta = [
        ("p1", 13, True, "train", 0),
        ("p2", None, False, "val", 0),
        ("p3", 11, True, "test", 0),
    ]
    for pid, t_est, sepsis, split, censored in meta:
        for hour in range(1, 9):
            row = {"patient_id": pid, "icu_hour": hour, "split": split}
            row["y_label"] = int(bool(sepsis) and hour >= (t_est - 6))
            row["is_sepsis_patient"] = int(sepsis)
            row["t_sepsis_est"] = t_est
            row["hours_to_onset"] = (t_est - hour) if sepsis else None
            row["onset_censored"] = censored
            for column in DEMOGRAPHICS:
                row[column] = 60.0 if column == "Age" else 0.0
            for column in GOLD_FEATURE_COLUMNS:
                row[column] = 0.0
            rows.append(row)
    frame = pd.DataFrame(rows)
    for column in ("t_sepsis_est", "hours_to_onset"):
        frame[column] = pd.array(frame[column], dtype="Int64")
    return frame


def test_valid_gold_passes():
    validate_gold_frame(make_gold())


def test_label_on_non_sepsis_patient_fails():
    frame = make_gold()
    mask = frame.patient_id == "p2"
    frame.loc[mask, "y_label"] = 1
    with pytest.raises(GoldSchemaError, match="non-sepsis patient has positive"):
        validate_gold_frame(frame)


def test_y_label_must_match_t_sepsis_est():
    frame = make_gold()
    mask = frame.patient_id == "p1"
    frame.loc[mask, "y_label"] = 0
    with pytest.raises(GoldSchemaError, match="disagrees with t_sepsis_est"):
        validate_gold_frame(frame)


def test_patient_cannot_span_two_splits():
    frame = make_gold()
    frame.loc[(frame.patient_id == "p1") & (frame.icu_hour == 1), "split"] = "test"
    with pytest.raises(GoldSchemaError, match="more than one split"):
        validate_gold_frame(frame)


def test_contiguous_hours_required():
    frame = make_gold()
    frame = frame[~((frame.patient_id == "p1") & (frame.icu_hour == 4))]
    with pytest.raises(GoldSchemaError, match="contiguous"):
        validate_gold_frame(frame)


def test_split_assignment_is_disjoint_and_complete():
    ids = [f"p{i:03d}" for i in range(60)]
    sepsis = {pid: i % 10 == 0 for i, pid in enumerate(ids)}
    assignment = assign_patient_splits(ids, sepsis_by_patient=sepsis)
    assert set(assignment) == set(ids)
    counts = {s: list(assignment.values()).count(s) for s in ("train", "val", "test")}
    assert all(counts[s] > 0 for s in counts)
    assert sum(counts.values()) == len(ids)


def test_feature_columns_never_include_labels():
    frame = make_gold()
    features = feature_columns(frame)
    forbidden = {
        "y_label",
        "is_sepsis_patient",
        "t_sepsis_est",
        "hours_to_onset",
        "onset_censored",
        "split",
        "patient_id",
        "icu_hour",
    }
    assert not (set(features) & forbidden)
    assert "HR_last" in features
    assert "Lactate_last" in features
