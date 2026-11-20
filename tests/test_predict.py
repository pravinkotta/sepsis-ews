import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from sklearn.isotonic import IsotonicRegression

from src.predict import ModelPredictor, features_to_matrix


def test_missing_values_become_float_nan_not_object():
    features = {"HR_last": 112.0, "Lactate_last": None, "WBC_last": 12.3}
    matrix = features_to_matrix(features, ["HR_last", "Lactate_last", "WBC_last", "pH_last"])
    assert list(matrix.columns) == ["HR_last", "Lactate_last", "WBC_last", "pH_last"]
    assert all(str(dtype) == "float32" for dtype in matrix.dtypes)
    assert np.isnan(matrix.loc[0, "Lactate_last"])
    assert np.isnan(matrix.loc[0, "pH_last"])
    assert matrix.loc[0, "HR_last"] == 112.0


def test_predictor_matches_booster_plus_calibrator(tmp_path):
    rng = np.random.default_rng(3)
    columns = ["a", "b", "c"]
    frame = pd.DataFrame(rng.normal(size=(300, 3)), columns=columns)
    labels = (frame["a"] + frame["b"] > 0).astype(int)

    booster = lgb.train(
        {"objective": "binary", "verbose": -1, "seed": 1},
        lgb.Dataset(frame, label=labels),
        num_boost_round=15,
    )
    booster.save_model(str(tmp_path / "model.txt"))
    probabilities = booster.predict(frame)
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(probabilities, labels)
    joblib.dump(calibrator, tmp_path / "calibrator.joblib")
    (tmp_path / "model_meta.json").write_text(
        json.dumps({"features": columns, "threshold": 0.5, "best_iteration": 15}),
        encoding="utf-8",
    )

    predictor = ModelPredictor(tmp_path)
    row = {"a": 1.2, "b": 0.4, "c": -0.3}
    result = predictor.predict(row)
    expected_probability = float(
        booster.predict(pd.DataFrame([row])[columns])[0]
    )
    expected_risk = float(calibrator.predict([expected_probability])[0])
    assert result["risk"] == pytest.approx(expected_risk, abs=1e-6)
    assert result["top_drivers"]
