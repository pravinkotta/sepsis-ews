import numpy as np
import pandas as pd
import pytest

from src.threshold import (
    AlertPolicy,
    InfeasibleAlertBudget,
    _Stream,
    alert_hours,
    evaluate_policy,
    select_threshold,
)


def frame_from_probs(patient_id, probs, t_sepsis=None, sepsis=False, start=1):
    return pd.DataFrame(
        {
            "patient_id": [patient_id] * len(probs),
            "icu_hour": list(range(start, start + len(probs))),
            "p": probs,
            "t_sepsis_est": [t_sepsis] * len(probs),
            "is_sepsis_patient": [int(sepsis)] * len(probs),
            "onset_censored": [0] * len(probs),
        }
    )


def cohort():
    p1 = frame_from_probs(
        "p1",
        [0.05] * 9 + [0.80] * 5 + [0.05] * 16,
        t_sepsis=20,
        sepsis=True,
    )
    p2 = frame_from_probs("p2", [0.05] * 30, t_sepsis=30, sepsis=True)
    p3 = frame_from_probs("p3", [0.01] * 30)
    return pd.concat([p1, p2, p3], ignore_index=True)


def test_alert_hours_respects_refractory():
    stream = _Stream(
        hours=np.arange(1, 31),
        probs=np.full(30, 0.9),
        t_sepsis=None,
        eligible=False,
    )
    policy = AlertPolicy(threshold=0.5, refractory_hours=12, persistence_hours=1)
    assert alert_hours(stream, policy) == [1, 13, 25]


def test_persistence_suppresses_isolated_spikes():
    stream = _Stream(
        hours=np.array([1, 2, 3, 4, 5]),
        probs=np.array([0.1, 0.9, 0.1, 0.9, 0.1]),
        t_sepsis=None,
        eligible=False,
    )
    assert alert_hours(stream, AlertPolicy(0.5, 0, 2)) == []
    assert alert_hours(stream, AlertPolicy(0.5, 0, 1)) == [2, 4]


def test_policy_report_credits_early_detection():
    policy = AlertPolicy(threshold=0.5, refractory_hours=12, persistence_hours=1)
    report = evaluate_policy(cohort(), policy)
    assert report.n_early_detected == 1
    assert report.n_missed == 1
    assert report.median_lead_hours == 10.0
    assert report.sensitivity == 0.5
    assert report.n_alerts == 1


def test_select_threshold_respects_budget_and_maximises_sensitivity():
    policy, report = select_threshold(cohort(), budget=1.0)
    assert 0.5 <= policy.threshold <= 0.80
    assert report.sensitivity == 0.5
    assert report.alerts_per_patient_day <= 1.0
    assert report.n_alerts == 1


def test_lower_budget_raises_instead_of_violating_it():
    always = frame_from_probs("p1", [1.0] * 48)
    with pytest.raises(InfeasibleAlertBudget, match="Raise the budget"):
        select_threshold(always, budget=0.0)


def test_policy_report_serialises():
    _, report = select_threshold(cohort(), budget=1.0)
    payload = report.as_dict()
    assert payload["n_eligible"] == 2
    assert payload["n_alerts"] == 1
    assert 0.0 <= payload["sensitivity"] <= 1.0
