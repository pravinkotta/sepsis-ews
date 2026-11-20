import numpy as np
import pandas as pd
import pytest

from src.store import DataFrameBackend, PatientNotFound
from src.tools import BackendNotConfigured


@pytest.fixture()
def backend():
    gold = pd.DataFrame(
        {
            "patient_id": ["p1", "p1", "p1", "p2"],
            "icu_hour": [1, 2, 3, 1],
            "HR_last": [100.0, 110.0, np.nan, 80.0],
            "Lactate": [2.0, np.nan, 3.0, 1.0],
            "y_label": [0, 0, 1, 0],
            "t_sepsis_est": [3, 3, 3, None],
        }
    )
    alerts = pd.DataFrame({"patient_id": ["p1"], "alert_id": ["a1"]})
    return DataFrameBackend(gold, alerts)


def test_patient_state_hides_labels(backend):
    state = backend.patient_state("p1", 2)
    assert state["as_of_hour"] == 2
    assert state["features"]["HR_last"] == 110.0
    assert "y_label" not in state["features"]
    assert "t_sepsis_est" not in state["features"]


def test_state_before_first_hour_fails(backend):
    with pytest.raises(PatientNotFound):
        backend.patient_state("p1", 0)


def test_unknown_patient_fails(backend):
    with pytest.raises(PatientNotFound):
        backend.patient_state("nope", 1)


def test_recent_labs_window_and_nan_cleaning(backend):
    result = backend.recent_labs("p1", 3, hours=2)
    assert len(result["rows"]) == 2
    assert result["rows"][0]["Lactate"] is None


def test_alert_history(backend):
    assert backend.alert_history("p1")["alerts"][0]["alert_id"] == "a1"
    assert backend.alert_history("p2")["alerts"] == []


def test_unbuilt_model_fails_loudly(backend):
    with pytest.raises(BackendNotConfigured, match="Refusing to invent"):
        backend.score("p1", 2)
    with pytest.raises(BackendNotConfigured):
        backend.retrieve("lactate", 5)
