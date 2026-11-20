import pytest

pytest.importorskip("torch")

from src.gru import integrated_gradients, predict_logits, train_gru  # noqa: E402
from src.sequence import build_sequences, fit_stats  # noqa: E402
from tests.gold_fixtures import make_gold_frame  # noqa: E402


def test_gru_trains_and_explains_a_small_cohort():
    gold = make_gold_frame(n_patients=30, n_hours=20, n_sepsis=8)
    stats = fit_stats(gold[gold["split"] == "train"])
    train = build_sequences(
        gold, "train", stats=stats, window=12, max_per_patient=2, seed=1
    )
    val = build_sequences(
        gold, "val", stats=stats, window=12, max_per_patient=2, seed=2
    )
    model, history = train_gru(
        train, val, epochs=1, batch_size=64, hidden=8
    )
    assert len(history) == 1
    logits = predict_logits(model, val)
    assert len(logits) == len(val)

    ig = integrated_gradients(model, val, sample_n=8, steps=4)
    assert ig["method"] == "integrated_gradients"
    assert ig["top_features"]
    assert ig["completeness_max_diff"] < 0.5
