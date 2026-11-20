"""GRU challenger: 24-hour windows, 34 temporal inputs + 6 static inputs.

Trained on CPU, compared with LightGBM on the validation split. It is kept as
the champion only if it beats the baseline; either way the comparison, the
calibration and the Integrated Gradients audit are committed.

The GRU is not allowed to replace the deployed model by accident: `comparison.json`
records the decision rule and the champion.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

from .config import ARTIFACTS
from .metrics import calibration_slope, ranking_metrics
from .sequence import (
    STATIC_WITH_HOUR,
    TEMPORAL_FEATURES,
    WINDOW,
    SequenceBatch,
    SequenceStats,
    build_sequences,
    fit_stats,
)
from .threshold import evaluate_policy, select_threshold


class GRUError(RuntimeError):
    """Raised when the challenger cannot be trained or evaluated."""


class GRUNet(nn.Module):
    def __init__(
        self,
        n_temporal: int = len(TEMPORAL_FEATURES),
        n_static: int = len(STATIC_WITH_HOUR),
        hidden: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(n_temporal, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden + n_static, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(x)
        return self.head(torch.cat([output[:, -1, :], s], dim=1)).squeeze(1)


def _tensors(batch: SequenceBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.from_numpy(batch.X),
        torch.from_numpy(batch.S),
        torch.from_numpy(batch.meta["y_label"].to_numpy(dtype=np.float32)),
    )


@torch.no_grad()
def predict_logits(
    model: nn.Module, batch: SequenceBatch, batch_size: int = 2048
) -> np.ndarray:
    model.eval()
    outputs = []
    for start in range(0, len(batch), batch_size):
        x = torch.from_numpy(batch.X[start:start + batch_size])
        s = torch.from_numpy(batch.S[start:start + batch_size])
        outputs.append(model(x, s).cpu().numpy())
    return np.concatenate(outputs)


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def train_gru(
    train: SequenceBatch,
    val: SequenceBatch,
    *,
    epochs: int = 12,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    patience: int = 3,
    hidden: int = 64,
    dropout: float = 0.3,
    seed: int = 42,
) -> tuple[GRUNet, list[dict]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    x, s, y = _tensors(train)
    positives = int(y.sum())
    if positives == 0:
        raise GRUError("training sample has no positive windows")
    negatives = len(y) - positives
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negatives / positives], dtype=torch.float32)
    )
    model = GRUNet(hidden=hidden, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    val_y = val.meta["y_label"].to_numpy()
    best_ap = -1.0
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict] = []
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(x))
        total_loss = 0.0
        for start in range(0, len(x), batch_size):
            index = permutation[start:start + batch_size]
            optimizer.zero_grad()
            logits = model(x[index], s[index])
            loss = loss_fn(logits, y[index])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(index)

        val_p = sigmoid(predict_logits(model, val))
        val_auc = float(roc_auc_score(val_y, val_p))
        val_ap = float(average_precision_score(val_y, val_p))
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(total_loss / len(x), 6),
                "val_auroc": round(val_auc, 6),
                "val_auprc": round(val_ap, 6),
            }
        )
        print(
            f"  epoch {epoch:02d} loss={history[-1]['train_loss']:.4f} "
            f"val AUROC={val_auc:.4f} AUPRC={val_ap:.4f}"
        )
        if val_ap > best_ap + 1e-4:
            best_ap = val_ap
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"  early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return model, history


def integrated_gradients(
    model: nn.Module,
    batch: SequenceBatch,
    *,
    sample_n: int = 128,
    steps: int = 48,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    n = min(sample_n, len(batch))
    indices = rng.choice(len(batch), size=n, replace=False)
    x = torch.from_numpy(batch.X[indices])
    s = torch.from_numpy(batch.S[indices])
    baseline_x = torch.zeros_like(x)
    baseline_s = torch.zeros_like(s)

    model.eval()
    with torch.no_grad():
        fx = model(x, s)
        fb = model(baseline_x, baseline_s)

    alphas = torch.linspace(0.0, 1.0, steps)
    weights = torch.ones(steps)
    weights[0] = weights[-1] = 0.5
    weights = weights / (steps - 1)  # trapezoidal rule

    total_x = torch.zeros_like(x)
    total_s = torch.zeros_like(s)
    for alpha, weight in zip(alphas, weights):
        xi = baseline_x + alpha * (x - baseline_x)
        si = baseline_s + alpha * (s - baseline_s)
        xi.requires_grad_(True)
        si.requires_grad_(True)
        logits = model(xi, si)
        grads_x, grads_s = torch.autograd.grad(logits.sum(), [xi, si])
        total_x += weight * grads_x
        total_s += weight * grads_s

    ig_x = (x - baseline_x) * total_x
    ig_s = (s - baseline_s) * total_s
    reconstructed = ig_x.sum(dim=(1, 2)) + ig_s.sum(dim=1)
    delta = fx - fb
    differences = torch.abs(reconstructed - delta)
    max_diff = float(torch.max(differences))
    mean_diff = float(torch.mean(differences))

    temporal = ig_x.sum(dim=1).detach().numpy()
    mean_abs = np.abs(temporal).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:12]
    return {
        "method": "integrated_gradients",
        "n_sample": n,
        "steps": steps,
        "completeness_max_diff": round(max_diff, 8),
        "completeness_mean_diff": round(mean_diff, 8),
        "top_features": [
            {
                "feature": TEMPORAL_FEATURES[index],
                "mean_abs_ig": round(float(mean_abs[index]), 6),
            }
            for index in order
        ],
    }


def _val_test_frames(
    model: nn.Module,
    stats: SequenceStats,
    gold: pd.DataFrame,
    *,
    window: int,
    budget: float,
) -> tuple[dict, dict, IsotonicRegression, np.ndarray]:
    val_full = build_sequences(gold, "val", stats=stats, window=window)
    val_p_raw = sigmoid(predict_logits(model, val_full))
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(val_p_raw, val_full.meta["y_label"].to_numpy())
    val_p = calibrator.predict(val_p_raw)
    val_frame = val_full.meta.copy()
    val_frame["p"] = val_p

    policy, _ = select_threshold(val_frame, budget=budget)
    del val_full

    test_full = build_sequences(gold, "test", stats=stats, window=window)
    test_p = calibrator.predict(sigmoid(predict_logits(model, test_full)))
    test_frame = test_full.meta.copy()
    test_frame["p"] = test_p

    test_policy_frame = test_full.meta[
        ["patient_id", "icu_hour", "t_sepsis_est", "is_sepsis_patient", "onset_censored"]
    ].copy()
    test_policy_frame["p"] = test_p
    report = evaluate_policy(test_policy_frame, policy)

    val_metrics = ranking_metrics(val_frame, val_p)
    test_metrics = ranking_metrics(test_frame, test_p)
    brier = float(np.mean((test_p - test_full.meta["y_label"].to_numpy()) ** 2))
    slope = calibration_slope(test_p, test_frame["y_label"].to_numpy())
    metrics = {
        **test_metrics,
        "val_auroc": val_metrics["auroc"],
        "val_auprc": val_metrics["auprc"],
        "policy": report.as_dict(),
        "calibration": {"brier": round(brier, 6), "slope": round(slope, 6)},
        "n_test_windows": int(len(test_frame)),
        "n_test_patients": int(test_frame["patient_id"].nunique()),
    }
    # Keep a small slice for Integrated Gradients before freeing the big tensor.
    ig_batch = SequenceBatch(
        X=test_full.X[: min(256, len(test_full))],
        S=test_full.S[: min(256, len(test_full))],
        meta=test_full.meta.iloc[: min(256, len(test_full))].reset_index(drop=True),
    )
    del test_full
    return metrics, {"val_frame": val_frame, "policy": policy}, calibrator, ig_batch


def run_challenger(
    gold: pd.DataFrame,
    out_dir: Path = ARTIFACTS,
    *,
    epochs: int = 12,
    window: int = WINDOW,
    hidden: int = 64,
    max_per_patient: int = 2,
    budget: float = 2.0,
    write_ig: bool = True,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    stats = fit_stats(gold[gold["split"] == "train"])
    train = build_sequences(
        gold, "train", stats=stats, window=window,
        max_per_patient=max_per_patient, seed=42,
    )
    val = build_sequences(
        gold, "val", stats=stats, window=window,
        max_per_patient=max_per_patient, seed=43,
    )
    model, history = train_gru(
        train, val, epochs=epochs, hidden=hidden
    )
    del train, val

    metrics, extras, calibrator, ig_batch = _val_test_frames(
        model, stats, gold, window=window, budget=budget
    )
    metrics["model"] = "gru"
    metrics["hidden"] = hidden
    metrics["history"] = history
    metrics["duration_seconds"] = round(time.time() - started, 1)

    if write_ig:
        metrics["integrated_gradients"] = integrated_gradients(model, ig_batch)

    torch.save(model.state_dict(), out_dir / "gru.pt")
    joblib.dump(calibrator, out_dir / "gru_calibrator.joblib")
    (out_dir / "gru_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out_dir / "gru_meta.json").write_text(
        json.dumps(
            {
                "window": window,
                "hidden": hidden,
                "temporal_features": TEMPORAL_FEATURES,
                "static_features": STATIC_WITH_HOUR,
                "threshold": extras["policy"].threshold,
                "policy": {
                    "refractory_hours": extras["policy"].refractory_hours,
                    "persistence_hours": extras["policy"].persistence_hours,
                },
                "stats": stats.as_dict(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return metrics


def lightgbm_validation(gold: pd.DataFrame, out_dir: Path = ARTIFACTS) -> dict:
    import lightgbm as lgb

    meta = json.loads((out_dir / "model_meta.json").read_text(encoding="utf-8"))
    booster = lgb.Booster(model_file=str(out_dir / "model.txt"))
    val = gold[gold["split"] == "val"]
    p = booster.predict(val[meta["features"]])
    return ranking_metrics(val, p)


def compare(out_dir: Path = ARTIFACTS) -> dict:
    import lightgbm as lgb  # noqa: F401 - ensures LightGBM artefacts are usable
    import pandas as pd

    gold = pd.read_parquet(out_dir / "gold_hourly.parquet")
    lgbm_val = lightgbm_validation(gold, out_dir)
    lgbm_test = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    gru = json.loads((out_dir / "gru_metrics.json").read_text(encoding="utf-8"))

    gru_wins = (
        gru["val_auprc"] > lgbm_val["auprc"] + 0.002
        and gru["val_auroc"] >= lgbm_val["auroc"] - 0.002
    )
    payload = {
        "lightgbm": {
            "val_auroc": lgbm_val["auroc"],
            "val_auprc": lgbm_val["auprc"],
            "test_auroc": lgbm_test["auroc"],
            "test_auprc": lgbm_test["auprc"],
        },
        "gru": {
            "val_auroc": gru["val_auroc"],
            "val_auprc": gru["val_auprc"],
            "test_auroc": gru["auroc"],
            "test_auprc": gru["auprc"],
        },
        "champion": "gru" if gru_wins else "lightgbm",
        "decision_rule": (
            "GRU is champion only if validation AUPRC improves by > 0.002 and "
            "validation AUROC is not worse by more than 0.002."
        ),
        "note": (
            "Test metrics are reported for both models; the champion decision "
            "uses validation only."
        ),
    }
    (out_dir / "comparison.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload
