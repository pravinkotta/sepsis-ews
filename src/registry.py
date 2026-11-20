"""MLflow tracking + model registry for the baseline and the challenger.

    python -m src.registry

Both runs log params, measured metrics and artefacts. The winner recorded in
`comparison.json` gets the `champion` alias; the other remains available for
comparison. No metric is logged that is not present in a committed artefact.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402
import mlflow.lightgbm  # noqa: E402
import mlflow.pytorch  # noqa: E402
import lightgbm as lgb  # noqa: E402
import torch  # noqa: E402

from .config import ARTIFACTS  # noqa: E402
from .gru import GRUNet  # noqa: E402


class RegistryError(RuntimeError):
    """Raised when tracking or registration cannot be completed."""


def _tracking_uri(out_dir: Path) -> str:
    database = (out_dir / "mlflow.db").resolve()
    return f"sqlite:///{database.as_posix()}"


def _numeric_metrics(payload: dict) -> dict:
    metrics = {}
    for key in ("auroc", "auprc", "prevalence", "val_auroc", "val_auprc",
                "auroc_all_hours", "patient_level_auroc"):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            metrics[key] = float(value)
    policy = payload.get("policy", {})
    for key in ("threshold", "sensitivity", "alerts_per_patient_day", "median_lead_hours"):
        value = policy.get(key)
        if isinstance(value, (int, float)):
            metrics[f"policy_{key}"] = float(value)
    calibration = payload.get("calibration", {})
    for key in ("brier", "slope"):
        value = calibration.get(key)
        if isinstance(value, (int, float)):
            metrics[f"calibration_{key}"] = float(value)
    return metrics


def _alias_champion(client, model_name: str) -> str:
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise RegistryError(f"no registered versions for {model_name}")
    latest = max(versions, key=lambda version: int(version.version))
    client.set_registered_model_alias(model_name, "champion", latest.version)
    return latest.version


def register(out_dir: Path = ARTIFACTS) -> dict:
    out_dir = Path(out_dir)
    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    gru = json.loads((out_dir / "gru_metrics.json").read_text(encoding="utf-8"))
    comparison = json.loads((out_dir / "comparison.json").read_text(encoding="utf-8"))
    meta = json.loads((out_dir / "model_meta.json").read_text(encoding="utf-8"))
    gru_meta = json.loads((out_dir / "gru_meta.json").read_text(encoding="utf-8"))

    uri = _tracking_uri(out_dir)
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    mlflow.set_experiment("sepsis-ews")
    client = mlflow.MlflowClient()

    runs: dict[str, dict] = {}

    with mlflow.start_run(run_name="lightgbm-baseline") as run:
        mlflow.log_params(
            {
                "model": "lightgbm",
                "best_iteration": metrics.get("best_iteration"),
                "n_features": metrics.get("n_features"),
                "threshold": meta.get("threshold"),
            }
        )
        mlflow.log_metrics(_numeric_metrics(metrics))
        mlflow.log_artifact(str(out_dir / "model.txt"))
        mlflow.log_artifact(str(out_dir / "calibrator.joblib"))
        mlflow.log_artifact(str(out_dir / "metrics.json"))
        booster = lgb.Booster(model_file=str(out_dir / "model.txt"))
        mlflow.lightgbm.log_model(
            booster,
            name="model",
            registered_model_name="sepsis-lightgbm",
        )
        runs["lightgbm"] = {"run_id": run.info.run_id}

    model = GRUNet(hidden=int(gru_meta["hidden"]))
    model.load_state_dict(torch.load(out_dir / "gru.pt", map_location="cpu"))
    model.eval()
    with mlflow.start_run(run_name="gru-challenger") as run:
        mlflow.log_params(
            {
                "model": "gru",
                "window": gru_meta["window"],
                "hidden": gru_meta["hidden"],
                "epochs": len(gru.get("history", [])),
                "threshold": gru_meta.get("threshold"),
            }
        )
        mlflow.log_metrics(_numeric_metrics(gru))
        mlflow.log_artifact(str(out_dir / "gru.pt"))
        mlflow.log_artifact(str(out_dir / "gru_calibrator.joblib"))
        mlflow.log_artifact(str(out_dir / "gru_metrics.json"))
        mlflow.pytorch.log_model(
            model,
            name="model",
            registered_model_name="sepsis-gru",
            serialization_format="pickle",
        )
        runs["gru"] = {"run_id": run.info.run_id}

    champion = comparison["champion"]
    runs["champion"] = champion
    runs["champion_version"] = _alias_champion(
        client, f"sepsis-{champion}"
    )
    runs["tracking_uri"] = uri
    (out_dir / "mlflow_runs.json").write_text(
        json.dumps(runs, indent=2, sort_keys=True), encoding="utf-8"
    )
    return runs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MLflow tracking + registry")
    parser.add_argument("--out", type=Path, default=ARTIFACTS)
    args = parser.parse_args(argv)
    try:
        runs = register(args.out)
    except Exception as exc:  # noqa: BLE001 - surface the real MLflow error
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"champion: {runs['champion']} v{runs['champion_version']}")
    print(f"tracking: {runs['tracking_uri']}")
    print(f"runs -> {args.out / 'mlflow_runs.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
