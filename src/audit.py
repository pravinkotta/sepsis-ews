"""Subgroup audit and drift summary, measured on the test split only.

    python -m src.audit --gold artifacts/gold_hourly.parquet

Fairness: Fairlearn MetricFrame by age band, sex, and ICU unit. The released
2019 cohort has no race/ethnicity field, so no such claim is made. Groups with
fewer than 50 patients are suppressed rather than described.

Drift: population stability index (PSI) on the top features, train vs test.
A high PSI sets ``recommend_retrain``; nothing retrains automatically (C9).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame, false_positive_rate
from sklearn.metrics import recall_score

from .config import ARTIFACTS
from .schema import feature_columns

MIN_GROUP_PATIENTS = 50
PSI_DRIFT = 0.25
PSI_MODERATE = 0.10


def wilson_interval(successes: int, total: int, z: float = 1.96):
    """95% Wilson interval for a binomial proportion; None when total is 0."""
    if total <= 0:
        return None
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [
        round(max(0.0, centre - half), 4),
        round(min(1.0, centre + half), 4),
    ]


class AuditError(RuntimeError):
    """Raised when the model artefacts needed for an audit are absent."""


def add_subgroups(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["age_band"] = pd.cut(
        out["Age"],
        bins=[0, 50, 65, 80, 200],
        labels=["<50", "50-64", "65-79", "80+"],
        right=False,
    ).astype(str)
    out["sex"] = out["Gender"].map({0: "female", 1: "male"}).fillna("unknown")
    out["unit"] = np.select(
        [out["Unit1"] == 1, out["Unit2"] == 1],
        ["MICU", "SICU"],
        default="Other",
    )
    return out


def population_stability_index(
    reference: pd.Series, current: pd.Series, bins: int = 10
) -> float:
    ref = reference.dropna().to_numpy(dtype=float)
    cur = current.dropna().to_numpy(dtype=float)
    if len(ref) < 10 or len(cur) < 10:
        return 0.0
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_share = np.clip(ref_counts / max(ref_counts.sum(), 1), 1e-6, None)
    cur_share = np.clip(cur_counts / max(cur_counts.sum(), 1), 1e-6, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def _load_model(out_dir: Path):
    meta_path = out_dir / "model_meta.json"
    model_path = out_dir / "model.txt"
    calibrator_path = out_dir / "calibrator.joblib"
    metrics_path = out_dir / "metrics.json"
    missing = [
        str(path)
        for path in (meta_path, model_path, calibrator_path, metrics_path)
        if not path.exists()
    ]
    if missing:
        raise AuditError(
            f"missing model artefacts {missing}; run `python -m src.train` first"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    booster = lgb.Booster(model_file=str(model_path))
    calibrator = joblib.load(calibrator_path)
    return meta, metrics, booster, calibrator


def audit(
    gold: pd.DataFrame,
    out_dir: Path = ARTIFACTS,
    *,
    threshold: float | None = None,
) -> dict:
    out_dir = Path(out_dir)
    meta, metrics, booster, calibrator = _load_model(out_dir)
    features = meta["features"]
    threshold = float(
        threshold if threshold is not None else metrics["policy"]["threshold"]
    )

    train = gold[gold["split"] == "train"]
    test = gold[gold["split"] == "test"].copy()
    if test.empty:
        raise AuditError("test split is empty")
    test["p"] = calibrator.predict(booster.predict(test[features]))
    test["alert"] = (test["p"] >= threshold).astype(int)
    test = add_subgroups(test)

    allowed = {"age_band", "sex", "unit"}
    group_rows: list[dict] = []
    deltas: dict[str, dict] = {}
    suppressed: list[dict] = []

    y_true = test["y_label"].to_numpy()
    y_pred = test["alert"].to_numpy()

    for attribute in ("age_band", "sex", "unit"):
        frame_metric = MetricFrame(
            metrics={
                "recall": partial(recall_score, zero_division=0),
                "fpr": false_positive_rate,
            },
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=test[attribute],
        )
        rates = frame_metric.by_group
        kept: dict[str, tuple[float, float]] = {}
        for group, block in test.groupby(attribute, observed=True):
            n_patients = int(block["patient_id"].nunique())
            n_hours = int(len(block))
            positives = int(block["y_label"].sum())
            alerts = int(block["alert"].sum())
            tp = int(((block["y_label"] == 1) & (block["alert"] == 1)).sum())
            fp = int(((block["y_label"] == 0) & (block["alert"] == 1)).sum())
            row = {
                "attribute": attribute,
                "group": str(group),
                "n_patients": n_patients,
                "n_hours": n_hours,
                "positives": positives,
                "alerts": alerts,
                "tp": tp,
                "fp": fp,
                "recall": round(float(tp / positives), 4) if positives else None,
                "recall_ci95": wilson_interval(tp, positives),
                "fpr": round(float(fp / (n_hours - positives)), 4)
                if n_hours > positives
                else None,
                "fpr_ci95": wilson_interval(fp, n_hours - positives),
                "selection_rate": round(alerts / n_hours, 4) if n_hours else None,
                "selection_rate_ci95": wilson_interval(alerts, n_hours),
            }
            if n_patients < MIN_GROUP_PATIENTS:
                suppressed.append({"attribute": attribute, "group": str(group),
                                   "n_patients": n_patients})
                row["suppressed"] = True
            else:
                row["suppressed"] = False
                recall = rates.loc[group, "recall"] if group in rates.index else np.nan
                fpr = rates.loc[group, "fpr"] if group in rates.index else np.nan
                if not np.isnan(recall) and not np.isnan(fpr):
                    kept[str(group)] = (float(recall), float(fpr))
            group_rows.append(row)

        if kept:
            recalls = [v[0] for v in kept.values()]
            fprs = [v[1] for v in kept.values()]
            deltas[attribute] = {
                "recall_max_min": round(max(recalls) - min(recalls), 4),
                "fpr_max_min": round(max(fprs) - min(fprs), 4),
                "groups_compared": sorted(kept),
            }

    fairness = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "threshold": threshold,
        "level": "patient-hour",
        "min_group_patients": MIN_GROUP_PATIENTS,
        "attributes": sorted(allowed),
        "groups": group_rows,
        "deltas": deltas,
        "suppressed": suppressed,
        "caveat": (
            "Observations about this model on one retrospective cohort, not "
            "proof of discrimination. Hour-level rates weight long stays. "
            "95% intervals are Wilson binomial intervals, not causal claims."
        ),
    }
    (out_dir / "fairness.json").write_text(
        json.dumps(fairness, indent=2, sort_keys=True), encoding="utf-8"
    )

    shap_path = out_dir / "global_shap.json"
    if shap_path.exists():
        shap_summary = json.loads(shap_path.read_text(encoding="utf-8"))
        top = [item["feature"] for item in shap_summary.get("top_features", [])]
    else:
        top = feature_columns(gold)[:10]
    top = [name for name in top if name in train.columns][:10]

    drift_rows = []
    for name in top:
        score = population_stability_index(train[name], test[name])
        drift_rows.append(
            {
                "feature": name,
                "psi": round(score, 6),
                "verdict": "drift" if score >= PSI_DRIFT
                else "moderate" if score >= PSI_MODERATE
                else "stable",
            }
        )
    recommend = any(row["psi"] >= PSI_DRIFT for row in drift_rows)
    drift = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reference": "train",
        "current": "test",
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "features": drift_rows,
        "recommend_retrain": recommend,
        "note": (
            "Recorded, never executed. Automatic retraining is exactly what "
            "human oversight (EU AI Act Art. 14) is meant to prevent."
        ),
    }
    (out_dir / "drift_summary.json").write_text(
        json.dumps(drift, indent=2, sort_keys=True), encoding="utf-8"
    )

    return {"fairness": fairness, "drift": drift}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fairness + drift audit")
    parser.add_argument("--gold", type=Path, default=ARTIFACTS / "gold_hourly.parquet")
    parser.add_argument("--out", type=Path, default=ARTIFACTS)
    args = parser.parse_args(argv)

    if not args.gold.exists():
        print(f"ERROR: Gold table not found: {args.gold}", file=sys.stderr)
        return 2
    try:
        result = audit(pd.read_parquet(args.gold), args.out)
    except AuditError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"fairness -> {args.out / 'fairness.json'}")
    print(f"drift    -> {args.out / 'drift_summary.json'}")
    print(
        f"recommend_retrain={result['drift']['recommend_retrain']} "
        f"groups_reported={len(result['fairness']['groups'])}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
