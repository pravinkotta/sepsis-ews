"""Build the Gold table from the verified raw PhysioNet copy.

    python -m src.build_gold --raw data/raw --out artifacts

The builder is batched: 1,000 patients are concatenated and written as a part,
then each part is streamed into the final Parquet file after the patient-level
split is known. Peak memory stays bounded regardless of cohort size.

No raw data is committed. The output is derived and reproducible from
`scripts/download_data.py`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import ARTIFACTS, DATA_RAW, EXPECTED_SETS, RAW_COLUMNS, SPLIT_SEED
from .features import build_feature_matrix, build_patient_features
from .labels import LabelError, build_patient_index
from .schema import (
    GoldSchemaError,
    REQUIRED_GOLD_COLUMNS,
    assign_patient_splits,
    validate_gold_frame,
)


class GoldBuildError(RuntimeError):
    """Raised when the raw cohort cannot produce a valid Gold table."""


def _raw_files(raw_root: Path, limit: int | None):
    seen = 0
    for set_name, dir_name in EXPECTED_SETS.items():
        directory = raw_root / dir_name
        if not directory.is_dir():
            raise GoldBuildError(
                f"missing {directory}. Download with:\n"
                f"  python scripts/download_data.py\n"
                f"or\n"
                f"  aws s3 sync --no-sign-request "
                f"s3://physionet-open/challenge-2019/1.0.0/training {raw_root}"
            )
        for path in sorted(directory.glob("*.psv")):
            yield set_name, path
            seen += 1
            if limit is not None and seen >= limit:
                return


def build_gold(
    raw_root: Path = DATA_RAW,
    out_dir: Path = ARTIFACTS,
    *,
    limit: int | None = None,
    batch_size: int = 1000,
    seed: int = SPLIT_SEED,
    keep_parts: bool = False,
    progress: bool = True,
) -> dict:
    raw_root = Path(raw_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = out_dir / "_gold_parts"
    shutil.rmtree(parts_dir, ignore_errors=True)
    parts_dir.mkdir(parents=True)

    sepsis_by_patient: dict[str, bool] = {}
    bad: list[str] = []
    buffer: list[pd.DataFrame] = []
    n_patients = 0
    n_rows = 0
    part_index = 0
    started = time.time()

    def flush() -> None:
        nonlocal buffer, part_index
        if not buffer:
            return
        batch = build_feature_matrix(buffer)
        batch.to_parquet(parts_dir / f"part-{part_index:05d}.parquet", index=False)
        part_index += 1
        buffer = []

    for _, path in _raw_files(raw_root, limit):
        try:
            raw = pd.read_csv(path, sep="|")
            raw.columns = [str(c).strip() for c in raw.columns]
            if len(raw.columns) != len(RAW_COLUMNS) or set(raw.columns) != set(RAW_COLUMNS):
                raise GoldSchemaError(
                    f"{path.name}: expected {len(RAW_COLUMNS)} columns, "
                    f"got {len(raw.columns)}"
                )
            features = build_patient_features(raw, path.stem)
        except (GoldSchemaError, LabelError, pd.errors.ParserError) as exc:
            if len(bad) < 20:
                bad.append(str(exc))
            continue

        sepsis_by_patient[path.stem] = bool(features["is_sepsis_patient"].iloc[0])
        buffer.append(features)
        n_patients += 1
        n_rows += len(features)
        if len(buffer) >= batch_size:
            flush()
        if progress and n_patients % 5000 == 0:
            elapsed = time.time() - started
            print(f"  {n_patients:,} patients / {n_rows:,} hours ({elapsed:.0f}s)")

    flush()

    if bad:
        raise GoldBuildError(
            f"{len(bad)} raw files violated the contract; first failures: {bad[:5]}"
        )
    if n_patients < 3:
        raise GoldBuildError(f"only {n_patients} patients parsed; need at least 3")

    splits = assign_patient_splits(
        list(sepsis_by_patient),
        sepsis_by_patient=sepsis_by_patient,
        seed=seed,
    )

    gold_path = out_dir / "gold_hourly.parquet"
    writer: pq.ParquetWriter | None = None
    split_hours = {"train": 0, "val": 0, "test": 0}
    try:
        for part in sorted(parts_dir.glob("*.parquet")):
            frame = pd.read_parquet(part)
            frame["split"] = frame["patient_id"].map(splits).astype("string")
            table = pa.Table.from_pandas(
                frame[REQUIRED_GOLD_COLUMNS], preserve_index=False
            )
            if writer is None:
                writer = pq.ParquetWriter(
                    gold_path, table.schema, compression="snappy"
                )
            writer.write_table(table)
            counts = frame["split"].value_counts()
            for name in split_hours:
                split_hours[name] += int(counts.get(name, 0))
    finally:
        if writer is not None:
            writer.close()

    gold = pd.read_parquet(gold_path)
    validate_gold_frame(gold)

    index = build_patient_index(gold)
    index["split"] = index["patient_id"].map(splits).astype("string")
    index.to_parquet(out_dir / "patient_index.parquet", index=False)

    split_patients = Counter(splits.values())
    sepsis_total = sum(1 for value in sepsis_by_patient.values() if value)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "raw_root": str(raw_root),
        "gold_path": str(gold_path),
        "patients": n_patients,
        "hours": n_rows,
        "sepsis_patients": sepsis_total,
        "prevalence": round(sepsis_total / n_patients, 6),
        "split_patients": {name: int(split_patients.get(name, 0)) for name in split_hours},
        "split_hours": split_hours,
        "columns": len(REQUIRED_GOLD_COLUMNS),
        "limit": limit,
        "duration_seconds": round(time.time() - started, 1),
        "bad_files": bad,
    }
    (out_dir / "gold_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    if not keep_parts:
        shutil.rmtree(parts_dir, ignore_errors=True)

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the sepsis Gold table")
    parser.add_argument("--raw", type=Path, default=DATA_RAW)
    parser.add_argument("--out", type=Path, default=ARTIFACTS)
    parser.add_argument("--limit", type=int, help="build only the first N patients")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--keep-parts", action="store_true")
    args = parser.parse_args(argv)

    if args.limit is not None and args.limit < 3:
        parser.error("--limit must be >= 3 (three splits need three patients)")

    try:
        summary = build_gold(
            args.raw,
            args.out,
            limit=args.limit,
            batch_size=args.batch_size,
            seed=args.seed,
            keep_parts=args.keep_parts,
        )
    except GoldBuildError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.limit:
        print(
            "WARNING: partial build (--limit); split counts are not the full cohort."
        )
    print(
        f"patients={summary['patients']:,} hours={summary['hours']:,} "
        f"prevalence={summary['prevalence']:.4f} "
        f"in {summary['duration_seconds']}s"
    )
    print(f"gold -> {summary['gold_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
