import numpy as np
import pandas as pd

from src.audit import audit, population_stability_index, wilson_interval
from src.train import train_model
from tests.gold_fixtures import make_gold_frame


def test_psi_is_small_for_same_distribution_and_large_after_shift():
    rng = np.random.default_rng(7)
    reference = pd.Series(rng.normal(0, 1, 3000))
    same = pd.Series(rng.normal(0, 1, 3000))
    shifted = pd.Series(rng.normal(3, 1, 3000))
    assert population_stability_index(reference, same) < 0.10
    assert population_stability_index(reference, shifted) > 0.25


def test_wilson_interval_bounds():
    low, high = wilson_interval(8, 10)
    assert 0.0 <= low < high <= 1.0
    assert low < 0.8 < high
    assert wilson_interval(0, 0) is None


def test_audit_writes_fairness_and_drift(tmp_path):
    gold = make_gold_frame()
    train_model(
        gold,
        tmp_path,
        num_boost_round=40,
        early_stopping=10,
        write_shap=False,
    )
    result = audit(gold, tmp_path)

    assert (tmp_path / "fairness.json").exists()
    assert (tmp_path / "drift_summary.json").exists()
    assert result["fairness"]["level"] == "patient-hour"
    assert result["fairness"]["suppressed"]
    assert any("recall_ci95" in row for row in result["fairness"]["groups"])
    assert "recommend_retrain" in result["drift"]
    assert len(result["drift"]["features"]) == 10
