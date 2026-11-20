import pandas as pd
import pytest

from src.labels import LabelError, add_label_columns, build_patient_index, derive_onset


def test_onset_is_first_positive_plus_six():
    onset = derive_onset([0, 0, 0, 1, 1], [1, 2, 3, 4, 5])
    assert onset.sepsis_patient
    assert onset.first_positive_hour == 4
    assert onset.t_sepsis_est == 10
    assert onset.censored is False


def test_record_starting_positive_is_censored():
    onset = derive_onset([1, 1, 1], [1, 2, 3])
    assert onset.t_sepsis_est == 7
    assert onset.censored is True


def test_record_may_start_after_icu_hour_one():
    # Real 2019 records often begin at ICULOS 2 or 3. Onset stays in ICULOS
    # units: first positive hour 5 -> t_sepsis_est 11.
    onset = derive_onset([0, 0, 1], [3, 4, 5])
    assert onset.first_positive_hour == 5
    assert onset.t_sepsis_est == 11
    assert onset.censored is False


def test_non_sepsis_patient_has_no_onset():
    onset = derive_onset([0, 0, 0], [1, 2, 3])
    assert onset.sepsis_patient is False
    assert onset.t_sepsis_est is None
    assert onset.censored is False


def test_non_monotonic_label_fails_loudly():
    with pytest.raises(LabelError, match="not monotonic"):
        derive_onset([0, 1, 0], [1, 2, 3])


def test_gap_in_hours_fails_loudly():
    with pytest.raises(LabelError, match="increase by exactly 1"):
        derive_onset([0, 0, 0], [1, 2, 4])


def test_add_label_columns_and_index():
    frame = pd.DataFrame(
        {
            "patient_id": ["p1"] * 5 + ["p2"] * 3,
            "icu_hour": [1, 2, 3, 4, 5, 1, 2, 3],
            "SepsisLabel": [0, 0, 0, 1, 1, 0, 0, 0],
        }
    )
    out = add_label_columns(frame)
    p1 = out[out.patient_id == "p1"]
    p2 = out[out.patient_id == "p2"]
    assert p1["is_sepsis_patient"].unique().tolist() == [1]
    assert p1["t_sepsis_est"].dropna().unique().tolist() == [10]
    assert p2["is_sepsis_patient"].unique().tolist() == [0]
    assert p2["t_sepsis_est"].isna().all()
    assert p2["hours_to_onset"].isna().all()

    index = build_patient_index(out)
    assert set(index.patient_id) == {"p1", "p2"}
    assert int(index.loc[index.patient_id == "p1", "n_hours"].iloc[0]) == 5
