import json

import pandas as pd
import pytest

from src.build_gold import GoldBuildError, build_gold
from src.schema import validate_gold_frame
from tests.raw_fixtures import write_raw_cohort

SPECS = [
    ("A", "p1", 6, 4),
    ("A", "p2", 6, 4),
    ("A", "p3", 6, None),
    ("B", "p4", 6, 4),
    ("B", "p5", 6, None),
    ("B", "p6", 6, None),
]


def test_build_gold_end_to_end(tmp_path):
    raw_root = tmp_path / "raw"
    out = tmp_path / "artifacts"
    write_raw_cohort(raw_root, SPECS)

    summary = build_gold(raw_root, out, batch_size=2, progress=False)

    assert summary["patients"] == 6
    assert summary["hours"] == 36
    assert summary["prevalence"] == 0.5
    assert summary["split_patients"] == {"train": 2, "val": 2, "test": 2}

    gold = pd.read_parquet(out / "gold_hourly.parquet")
    validate_gold_frame(gold)
    assert set(gold["split"]) == {"train", "val", "test"}
    assert gold.groupby("patient_id")["split"].nunique().max() == 1

    index = pd.read_parquet(out / "patient_index.parquet")
    assert len(index) == 6
    assert set(index.columns) >= {"patient_id", "n_hours", "split", "t_sepsis_est"}

    summary_file = json.loads((out / "gold_summary.json").read_text())
    assert summary_file["patients"] == 6


def test_bad_raw_file_fails_loudly(tmp_path):
    raw_root = tmp_path / "raw"
    out = tmp_path / "artifacts"
    write_raw_cohort(raw_root, SPECS)
    bad = raw_root / "training_setA" / "p999999.psv"
    bad.write_text("HR|O2Sat|Temp\n90|97|37\n", encoding="utf-8")

    with pytest.raises(GoldBuildError, match="violated the contract"):
        build_gold(raw_root, out, batch_size=2, progress=False)


def test_missing_set_directory_fails_with_the_fix(tmp_path):
    raw_root = tmp_path / "raw"
    (raw_root / "training_setA").mkdir(parents=True)
    with pytest.raises(GoldBuildError, match="python scripts/download_data.py"):
        build_gold(raw_root, tmp_path / "artifacts", progress=False)
