"""Export the test cohort as the small serving table.

The API only needs the cohort it replays; loading 1.55M training rows into the
container is wasteful and OOMs a 2 GB Docker VM. Labels and split metadata are
dropped — the service needs features, not the answer.

    python -m src.export_serving
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .config import ARTIFACTS

DROP = [
    "split",
    "y_label",
    "is_sepsis_patient",
    "t_sepsis_est",
    "hours_to_onset",
    "onset_censored",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export serving cohort")
    parser.add_argument("--gold", type=Path, default=ARTIFACTS / "gold_hourly.parquet")
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "serving.parquet")
    args = parser.parse_args(argv)

    gold = pd.read_parquet(args.gold)
    serving = gold[gold["split"] == "test"].drop(columns=DROP)
    serving.to_parquet(args.out, index=False)
    print(
        f"serving -> {args.out} "
        f"({len(serving):,} rows, {serving['patient_id'].nunique():,} patients, "
        f"{args.out.stat().st_size / 1e6:.1f} MB)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
