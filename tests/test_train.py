import json

from src.train import train_model
from tests.gold_fixtures import make_gold_frame


def test_training_pipeline_writes_measured_artifacts(tmp_path):
    gold = make_gold_frame()
    metrics = train_model(
        gold,
        tmp_path,
        num_boost_round=80,
        early_stopping=10,
        write_shap=True,
    )

    assert metrics["auroc"] > 0.70
    assert metrics["policy"]["n_eligible"] == 2
    assert metrics["policy"]["alerts_per_patient_day"] <= 2.0
    assert metrics["calibration"]["brier"] < 0.25
    assert metrics["shap"]["additivity_max_diff"] < 1e-4

    for name in (
        "metrics.json",
        "model.txt",
        "calibrator.joblib",
        "model_meta.json",
        "global_shap.json",
    ):
        assert (tmp_path / name).exists(), name

    meta = json.loads((tmp_path / "model_meta.json").read_text())
    assert meta["features"]
    assert "y_label" not in meta["features"]
    assert "t_sepsis_est" not in meta["features"]
