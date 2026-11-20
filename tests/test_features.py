import numpy as np
import pandas as pd
import pytest

from src.features import assert_causal, build_patient_features
from tests.raw_fixtures import make_raw


def test_rolling_windows_are_trailing():
    out = build_patient_features(make_raw(hours=6), "p1")
    # HR = 90, 100, 110, 120, 130, 140
    assert out.loc[5, "HR_mean_6h"] == pytest.approx(115.0)
    assert out.loc[5, "HR_mean_24h"] == pytest.approx(115.0)
    assert out.loc[1, "HR_mean_6h"] == pytest.approx(95.0)
    assert out.loc[5, "HR_last"] == pytest.approx(140.0)


def test_missing_fraction_is_explicit():
    raw = make_raw(hours=6)
    raw.loc[1:2, "HR"] = np.nan  # hours 2 and 3
    out = build_patient_features(raw, "p1")
    assert out.loc[5, "HR_missing_frac_24h"] == pytest.approx(2 / 6)
    assert out.loc[2, "HR_missing_frac_24h"] == pytest.approx(2 / 3)
    assert out.loc[0, "HR_missing_frac_24h"] == pytest.approx(0.0)
    # Last value is forward-filled from hour 1, never from the future.
    assert out.loc[2, "HR_last"] == pytest.approx(90.0)


def test_lab_staleness_is_measured_in_hours():
    out = build_patient_features(make_raw(hours=6), "p1")
    assert out.loc[0, "Lactate_hours_since"] == pytest.approx(0.0)
    assert out.loc[3, "Lactate_hours_since"] == pytest.approx(0.0)
    assert out.loc[5, "Lactate_hours_since"] == pytest.approx(2.0)
    assert out.loc[5, "Lactate_last"] == pytest.approx(6.0)
    assert out.loc[2, "Lactate_last"] == pytest.approx(3.0)
    assert out.loc[2, "Lactate_hours_since"] == pytest.approx(2.0)


def test_lab_before_first_measurement_is_missing():
    raw = make_raw(hours=6)
    raw["Lactate"] = np.nan
    raw.loc[3, "Lactate"] = 4.2  # only hour 4
    out = build_patient_features(raw, "p1")
    assert np.isnan(out.loc[1, "Lactate_last"])
    assert np.isnan(out.loc[1, "Lactate_hours_since"])
    assert out.loc[3, "Lactate_hours_since"] == pytest.approx(0.0)


def test_future_perturbation_cannot_change_the_past():
    assert_causal(make_raw(hours=12), "p1", cutoff_hour=5)


def test_label_columns_are_derived_from_the_shifted_label():
    out = build_patient_features(make_raw(hours=12, sepsis_from=4), "p1")
    assert out["t_sepsis_est"].dropna().unique().tolist() == [10]
    assert out.loc[0, "hours_to_onset"] == 9
    assert out.loc[0, "y_label"] == 0
    assert out.loc[3, "y_label"] == 1
