import numpy as np

from src.sequence import TEMPORAL_FEATURES, build_sequences, fit_stats
from tests.gold_fixtures import make_gold_frame


def test_sequences_have_the_contract_shape_and_are_finite():
    gold = make_gold_frame(n_patients=12, n_hours=8, n_sepsis=3)
    stats = fit_stats(gold[gold["split"] == "train"])
    batch = build_sequences(gold, "train", stats=stats, window=6)
    assert batch.X.shape[1:] == (6, len(TEMPORAL_FEATURES))
    assert batch.S.shape[1] == 6
    assert np.isfinite(batch.X).all()
    assert np.isfinite(batch.S).all()
    assert len(batch) == len(batch.meta)


def test_first_window_is_padded_with_the_first_hour():
    gold = make_gold_frame(n_patients=6, n_hours=8, n_sepsis=2)
    stats = fit_stats(gold[gold["split"] == "train"])
    batch = build_sequences(gold, "train", stats=stats, window=6)
    first = 0
    np.testing.assert_allclose(batch.X[first, 0], batch.X[first, -1], atol=1e-6)


def test_max_per_patient_limits_rows():
    gold = make_gold_frame(n_patients=12, n_hours=8, n_sepsis=3)
    stats = fit_stats(gold[gold["split"] == "train"])
    batch = build_sequences(
        gold, "train", stats=stats, window=6, max_per_patient=2, seed=1
    )
    counts = batch.meta.groupby("patient_id").size()
    assert counts.max() <= 2
