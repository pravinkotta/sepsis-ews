"""Evidently drift report: train vs test, all model features.

    python -m src.drift_report

The machine-readable PSI summary stays in `drift_summary.json`; this adds the
full Evidently HTML report and records where it came from. The report is not
committed (large, regenerable).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import ARTIFACTS
from .schema import feature_columns


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the Evidently drift report")
    parser.add_argument("--gold", type=Path, default=ARTIFACTS / "gold_hourly.parquet")
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    parser.add_argument("--rows", type=int, default=20_000)
    args = parser.parse_args(argv)

    from evidently import Report
    from evidently.presets import DataDriftPreset

    gold = pd.read_parquet(args.gold)
    features = feature_columns(gold)
    reference = gold[gold["split"] == "train"][features].sample(
        n=min(args.rows, int((gold["split"] == "train").sum())), random_state=42
    )
    current = gold[gold["split"] == "test"][features].sample(
        n=min(args.rows, int((gold["split"] == "test").sum())), random_state=42
    )

    started = time.time()
    report = Report(metrics=[DataDriftPreset()])
    result = report.run(reference_data=reference, current_data=current)
    output = args.artifacts / "drift_report.html"
    result.save_html(str(output))

    summary_path = args.artifacts / "drift_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["evidently"] = {
        "report": str(output),
        "n_reference": int(len(reference)),
        "n_current": int(len(current)),
        "n_features": len(features),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(time.time() - started, 1),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"report -> {output} ({len(reference):,} vs {len(current):,} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
