"""Train the M1 baseline: LightGBM + isotonic calibration + alert-budget policy.

    python -m src.train --gold artifacts/gold_hourly.parquet

The test split is touched exactly once, after the threshold is chosen on
validation. Every number written to ``metrics.json`` is measured on test rows;
no value is estimated. A failing metric is reported, not hidden.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import ALERT_BUDGET_PER_PATIENT_DAY, ARTIFACTS, SPLIT_SEED
from .schema import feature_columns
from .threshold import ThresholdError, evaluate_policy, select_threshold

EPS = 1e-6


class TrainingError(RuntimeError):
    """Raised when the cohort cannot support a defensible model."""


class ExplainabilityError(RuntimeError):
    """Raised when SHAP values fail their own additivity check."""


def _prepare(gold: pd.DataFrame) -> tuple[list[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for split in ("train", "val", "test"):
        if split not in set(gold["split"]):
            raise TrainingError(f"split {split!r} is missing from the Gold table")
    features = feature_columns(gold)
    train = gold[gold["split"] == "train"]
    val = gold[gold["split"] == "val"]
    test = gold[gold["split"] == "test"]
    for name, frame in (("train", train), ("val", val)):
        if frame["y_label"].nunique() < 2:
            raise TrainingError(f"{name} split has only one class")
        if frame[features].isna().all(axis=None):
            raise TrainingError(f"{name} split has no usable feature values")
    return features, train, val, test


def _fit_lightgbm(
    features: list[str],
    train: pd.DataFrame,
    val: pd.DataFrame,
    *,
    seed: int,
    num_boost_round: int,
    early_stopping: int,
) -> tuple[lgb.Booster, int]:
    positives = int(train["y_label"].sum())
    negatives = len(train) - positives
    if positives == 0:
        raise TrainingError("training split has zero positive hours")
    params = {
        "objective": "binary",
        "metric": "average_precision",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "scale_pos_weight": negatives / positives,
        "n_jobs": -1,
        "verbose": -1,
        "seed": seed,
        "deterministic": True,
    }
    train_set = lgb.Dataset(train[features], label=train["y_label"], free_raw_data=False)
    valid_set = lgb.Dataset(
        val[features], label=val["y_label"], reference=train_set, free_raw_data=False
    )
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[valid_set],
        callbacks=[
            lgb.early_stopping(early_stopping, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    best = int(booster.best_iteration or booster.current_iteration())
    return booster, best


def _calibrate(
    booster: lgb.Booster,
    features: list[str],
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, IsotonicRegression]:
    raw_val = booster.predict(val[features])
    raw_test = booster.predict(test[features])
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(raw_val, val["y_label"].to_numpy())
    return calibrator.predict(raw_val), calibrator.predict(raw_test), calibrator


def _calibration_slope(p: np.ndarray, y: np.ndarray) -> float:
    clipped = np.clip(p, EPS, 1 - EPS)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    model.fit(logit, y)
    return float(model.coef_[0, 0])


def _pre_onset_mask(frame: pd.DataFrame) -> np.ndarray:
    """Hours before onset (plus all hours of non-sepsis patients).

    Post-onset hours are excluded because the label is deterministically 1
    there; including them inflates discrimination without measuring early
    warning.
    """
    non_sepsis = frame["is_sepsis_patient"].to_numpy() == 0
    hours = frame["hours_to_onset"].fillna(1).to_numpy()
    return non_sepsis | (hours > 0)


def _shap_summary(
    booster: lgb.Booster,
    features: list[str],
    sample: pd.DataFrame,
    out_dir: Path,
) -> dict:
    import shap

    explainer = shap.TreeExplainer(booster)
    values = explainer.shap_values(sample[features])
    if isinstance(values, list):  # some SHAP versions return one array per class
        values = values[-1]
    values = np.asarray(values)
    if values.ndim != 2:
        raise ExplainabilityError(f"unexpected SHAP shape: {values.shape}")

    expected = explainer.expected_value
    if isinstance(expected, (list, np.ndarray)):
        expected = float(np.asarray(expected).ravel()[-1])
    raw = booster.predict(sample[features], raw_score=True)
    reconstructed = expected + values.sum(axis=1)
    max_diff = float(np.max(np.abs(reconstructed - raw)))
    if not np.allclose(reconstructed, raw, atol=1e-4):
        raise ExplainabilityError(
            f"SHAP additivity failed: max diff {max_diff:.6f} vs raw margin"
        )

    mean_abs = np.abs(values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:12]
    summary = {
        "additivity_max_diff": round(max_diff, 8),
        "n_sample": int(len(sample)),
        "top_features": [
            {"feature": features[i], "mean_abs_shap": round(float(mean_abs[i]), 6)}
            for i in order
        ],
    }
    (out_dir / "global_shap.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def train_model(
    gold: pd.DataFrame,
    out_dir: Path = ARTIFACTS,
    *,
    budget: float = ALERT_BUDGET_PER_PATIENT_DAY,
    seed: int = SPLIT_SEED,
    num_boost_round: int = 2000,
    early_stopping: int = 100,
    write_shap: bool = True,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    features, train, val, test = _prepare(gold)
    booster, best_iteration = _fit_lightgbm(
        features,
        train,
        val,
        seed=seed,
        num_boost_round=num_boost_round,
        early_stopping=early_stopping,
    )
    # Truncate to the early-stopped iteration so the deployed model, the
    # calibration, and SHAP all explain exactly the same trees.
    model_path = out_dir / "model.txt"
    booster.save_model(str(model_path), num_iteration=best_iteration)
    booster = lgb.Booster(model_file=str(model_path))

    val_p, test_p, calibrator = _calibrate(booster, features, val, test)

    val_frame = val[
        ["patient_id", "icu_hour", "t_sepsis_est", "is_sepsis_patient", "onset_censored"]
    ].copy()
    val_frame["p"] = val_p
    policy, _ = select_threshold(val_frame, budget=budget)

    test_frame = test[
        ["patient_id", "icu_hour", "t_sepsis_est", "is_sepsis_patient", "onset_censored"]
    ].copy()
    test_frame["p"] = test_p
    report = evaluate_policy(test_frame, policy)

    y_test = test["y_label"].to_numpy()
    mask = _pre_onset_mask(test)
    y_primary = y_test[mask]
    p_primary = test_p[mask]
    if len(np.unique(y_primary)) < 2:
        raise TrainingError("test pre-onset subset has only one class")

    prevalence = float(y_primary.mean())
    auroc = float(roc_auc_score(y_primary, p_primary))
    auprc = float(average_precision_score(y_primary, p_primary))

    auroc_all = float(roc_auc_score(y_test, test_p))
    patient_scores = (
        test.assign(_p=test_p)
        .groupby("patient_id")
        .agg(y=("is_sepsis_patient", "max"), p=("_p", "max"))
    )
    patient_auroc = (
        float(roc_auc_score(patient_scores["y"], patient_scores["p"]))
        if patient_scores["y"].nunique() == 2
        else None
    )

    brier = float(np.mean((test_p - y_test) ** 2))
    slope = _calibration_slope(test_p, y_test)

    joblib.dump(calibrator, out_dir / "calibrator.joblib")

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": "lightgbm",
        "best_iteration": best_iteration,
        "n_features": len(features),
        "cohort": {
            "train_patients": int(train["patient_id"].nunique()),
            "val_patients": int(val["patient_id"].nunique()),
            "test_patients": int(test["patient_id"].nunique()),
            "test_hours": int(len(test)),
            "pre_onset_hours": int(mask.sum()),
        },
        "auroc": round(auroc, 6),
        "auprc": round(auprc, 6),
        "prevalence": round(prevalence, 6),
        "auroc_all_hours": round(auroc_all, 6),
        "patient_level_auroc": None if patient_auroc is None else round(patient_auroc, 6),
        "policy": report.as_dict(),
        "calibration": {"brier": round(brier, 6), "slope": round(slope, 6)},
        "duration_seconds": round(time.time() - started, 1),
    }

    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out_dir / "model_meta.json").write_text(
        json.dumps(
            {
                "features": features,
                "threshold": policy.threshold,
                "refractory_hours": policy.refractory_hours,
                "persistence_hours": policy.persistence_hours,
                "best_iteration": metrics["best_iteration"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if write_shap:
        sample = train.sample(n=min(2000, len(train)), random_state=seed)
        metrics["shap"] = _shap_summary(booster, features, sample, out_dir)
        (out_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
        )

    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the sepsis baseline")
    parser.add_argument(
        "--gold", type=Path, default=ARTIFACTS / "gold_hourly.parquet"
    )
    parser.add_argument("--out", type=Path, default=ARTIFACTS)
    parser.add_argument("--budget", type=float, default=ALERT_BUDGET_PER_PATIENT_DAY)
    parser.add_argument("--rounds", type=int, default=2000)
    parser.add_argument("--early-stopping", type=int, default=100)
    parser.add_argument("--no-shap", action="store_true")
    args = parser.parse_args(argv)

    if not args.gold.exists():
        print(
            f"ERROR: Gold table not found: {args.gold}\n"
            f"Build it first: python -m src.build_gold --raw data/raw",
            file=sys.stderr,
        )
        return 2

    gold = pd.read_parquet(args.gold)
    try:
        metrics = train_model(
            gold,
            args.out,
            budget=args.budget,
            num_boost_round=args.rounds,
            early_stopping=args.early_stopping,
            write_shap=not args.no_shap,
        )
    except (TrainingError, ExplainabilityError, ThresholdError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        f"AUROC={metrics['auroc']:.4f} AUPRC={metrics['auprc']:.4f} "
        f"prevalence={metrics['prevalence']:.4f}"
    )
    print(
        f"policy: threshold={metrics['policy']['threshold']} "
        f"sensitivity={metrics['policy']['sensitivity']} "
        f"alerts/patient-day={metrics['policy']['alerts_per_patient_day']} "
        f"median lead={metrics['policy']['median_lead_hours']} h"
    )
    print(f"metrics -> {args.out / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
