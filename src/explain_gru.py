"""Recompute Integrated Gradients for the saved GRU without retraining.

    python -m src.explain_gru --steps 64
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

from .config import ARTIFACTS
from .gru import GRUNet, integrated_gradients
from .sequence import SequenceStats, build_sequences


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="GRU Integrated Gradients")
    parser.add_argument("--gold", type=Path, default=ARTIFACTS / "gold_hourly.parquet")
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--sample", type=int, default=128)
    args = parser.parse_args(argv)

    meta = json.loads((args.artifacts / "gru_meta.json").read_text(encoding="utf-8"))
    stats = SequenceStats.from_dict(meta["stats"])
    gold = pd.read_parquet(args.gold)
    sample = build_sequences(
        gold,
        "test",
        stats=stats,
        window=int(meta["window"]),
        max_per_patient=1,
        seed=7,
    )
    model = GRUNet(hidden=int(meta["hidden"]))
    model.load_state_dict(torch.load(args.artifacts / "gru.pt", map_location="cpu"))
    result = integrated_gradients(
        model, sample, sample_n=args.sample, steps=args.steps
    )
    metrics_path = args.artifacts / "gru_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["integrated_gradients"] = result
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    print(f"updated -> {metrics_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
