"""Train the GRU challenger and compare it with the LightGBM baseline.

    python -m src.train_gru --epochs 12

The champion decision uses validation only. Test metrics for both models are
reported; the GRU is not wired into the runtime unless it wins.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .config import ALERT_BUDGET_PER_PATIENT_DAY, ARTIFACTS
from .gru import GRUError, compare, run_challenger
from .threshold import ThresholdError


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Train the GRU challenger")
    parser.add_argument("--gold", type=Path, default=ARTIFACTS / "gold_hourly.parquet")
    parser.add_argument("--out", type=Path, default=ARTIFACTS)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--max-per-patient", type=int, default=2)
    parser.add_argument("--budget", type=float, default=ALERT_BUDGET_PER_PATIENT_DAY)
    parser.add_argument("--no-ig", action="store_true")
    args = parser.parse_args(argv)

    if not args.gold.exists():
        print(f"ERROR: Gold table not found: {args.gold}", file=sys.stderr)
        return 2

    try:
        gold = pd.read_parquet(args.gold)
        metrics = run_challenger(
            gold,
            args.out,
            epochs=args.epochs,
            window=args.window,
            hidden=args.hidden,
            max_per_patient=args.max_per_patient,
            budget=args.budget,
            write_ig=not args.no_ig,
        )
        comparison = compare(args.out)
    except (GRUError, ThresholdError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        f"GRU  val AUROC={metrics['val_auroc']:.4f} AUPRC={metrics['val_auprc']:.4f} | "
        f"test AUROC={metrics['auroc']:.4f} AUPRC={metrics['auprc']:.4f}"
    )
    print(
        f"LGBM val AUROC={comparison['lightgbm']['val_auroc']:.4f} "
        f"AUPRC={comparison['lightgbm']['val_auprc']:.4f} | "
        f"test AUROC={comparison['lightgbm']['test_auroc']:.4f} "
        f"AUPRC={comparison['lightgbm']['test_auprc']:.4f}"
    )
    print(f"champion: {comparison['champion']}")
    print(f"comparison -> {args.out / 'comparison.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
