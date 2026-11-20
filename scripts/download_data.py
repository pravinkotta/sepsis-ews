"""Download and verify the PhysioNet/CinC 2019 training database.

Source (documented on the dataset page):

    s3://physionet-open/challenge-2019/1.0.0/training
    https://physionet-open.s3.amazonaws.com/challenge-2019/1.0.0/training

License: CC BY 4.0 (open access, no credentialing). Cite:
Reyna et al., "Early Prediction of Sepsis From Clinical Data: The
PhysioNet/Computing in Cardiology Challenge 2019", Crit Care Med 48(2), 2020.
DOI 10.13026/v64v-d857

Usage:
    python scripts/download_data.py                 # download + verify + profile
    python scripts/download_data.py --verify-only   # verify an existing copy
    python scripts/download_data.py --source D:\\physionet-2019
    python scripts/download_data.py --sample 5      # fast smoke test (sampled=True)

A truncated/renamed/badly-encoded file exits non-zero with the file name and
the fix. The profile written to ``artifacts/data_profile.json`` is the only
input to the data gate; it is never hand-edited.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # allow `python scripts/download_data.py`
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.config import (  # noqa: E402
    DATA_RAW,
    EXPECTED_RAW_COLUMNS,
    EXPECTED_SETS,
    LABEL,
    RAW_COLUMNS,
    SPLIT_RATIOS,
    SPLIT_SEED,
)
from src.labels import LabelError, derive_onset  # noqa: E402
from src.schema import GoldSchemaError, assign_patient_splits, validate_raw_frame  # noqa: E402

S3_BUCKET = "physionet-open"
S3_PREFIX = "challenge-2019/1.0.0/training/"
HTTP_BASE = f"https://{S3_BUCKET}.s3.amazonaws.com/"


def _fail(message: str) -> "NoReturn":  # type: ignore[name-defined]
    print(f"\nERROR: {message}", file=sys.stderr)
    raise SystemExit(2)


def list_s3_objects(prefix: str = S3_PREFIX):
    """Yield (key, size) for every object under the prefix, paginated."""
    token: str | None = None
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        url = f"{HTTP_BASE}?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=60) as response:
            root = ET.fromstring(response.read())
        for item in root.iter():
            if item.tag.split("}")[-1] != "Contents":
                continue
            fields = {child.tag.split("}")[-1]: child.text for child in item}
            yield fields["Key"], int(fields["Size"])
        token = None
        for element in root.iter():
            if element.tag.split("}")[-1] == "NextContinuationToken":
                token = element.text
        if not token:
            return


def download_python(dest: Path, workers: int, force: bool) -> None:
    print(f"listing s3://{S3_BUCKET}/{S3_PREFIX} ...")
    objects = [(key, size) for key, size in list_s3_objects() if key.endswith(".psv")]
    if not objects:
        _fail("S3 listing returned no .psv files; the dataset path may have moved")
    print(f"found {len(objects):,} patient files")

    dest.mkdir(parents=True, exist_ok=True)
    counter = {"done": 0, "skipped": 0, "failed": 0}
    lock = threading.Lock()

    def fetch(key: str, size: int) -> None:
        relative = key[len(S3_PREFIX):]
        target = dest / relative
        if target.exists() and target.stat().st_size == size and not force:
            with lock:
                counter["skipped"] += 1
                counter["done"] += 1
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        url = HTTP_BASE + key
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=120) as response:
                    payload = response.read()
                if len(payload) != size:
                    raise IOError(f"size mismatch: got {len(payload)}, expected {size}")
                temporary = target.with_suffix(target.suffix + ".part")
                temporary.write_bytes(payload)
                temporary.replace(target)
                break
            except Exception as exc:  # noqa: BLE001 - retry then fail loudly
                if attempt == 2:
                    with lock:
                        counter["failed"] += 1
                    raise RuntimeError(f"{key}: {exc}") from exc
                time.sleep(1.5 * (attempt + 1))
        with lock:
            counter["done"] += 1
            if counter["done"] % 1000 == 0:
                print(f"  {counter['done']:,}/{len(objects):,} files")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, key, size) for key, size in objects]
        for future in as_completed(futures):
            future.result()

    print(f"downloaded {counter['done'] - counter['skipped']:,}, "
          f"skipped {counter['skipped']:,} existing files")
    if counter["failed"]:
        _fail(f"{counter['failed']} files failed to download")


def download_aws(dest: Path) -> bool:
    if shutil.which("aws") is None:
        return False
    dest.mkdir(parents=True, exist_ok=True)
    command = [
        "aws", "s3", "sync", "--no-sign-request",
        f"s3://{S3_BUCKET}/{S3_PREFIX}", str(dest), "--only-show-errors",
    ]
    print("using AWS CLI:", " ".join(command))
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        print("aws s3 sync failed; falling back to the Python downloader")
        return False
    return True


def verify(source: Path, sample: int | None) -> dict:
    set_dirs = [source / name for name in EXPECTED_SETS.values()]
    missing = [str(path) for path in set_dirs if not path.is_dir()]
    if missing:
        _fail(
            "expected training_setA/ and training_setB/ under "
            f"{source}; missing {missing}. Download with:\n"
            f"  aws s3 sync --no-sign-request s3://{S3_BUCKET}/{S3_PREFIX} {source}"
        )

    checks = {
        "columns_ok": True,
        "label_monotonic": True,
        "icu_hours_contiguous": True,
        "duplicate_ids": 0,
        "bad_files": [],
    }
    sets: dict[str, dict] = {}
    sepsis_by_patient: dict[str, bool] = {}
    seen: set[str] = set()
    total_hours = 0
    total_patients = 0

    for set_name, dir_name in EXPECTED_SETS.items():
        directory = source / dir_name
        files = sorted(directory.glob("*.psv"))
        if not files:
            _fail(f"no .psv files in {directory}")

        to_check = files if sample is None else files[:sample]
        patients = hours = sepsis_patients = 0

        for index, path in enumerate(to_check, start=1):
            pid = path.stem
            try:
                frame = pd.read_csv(path, sep="|")
            except Exception as exc:  # noqa: BLE001
                checks["bad_files"].append(f"{path.name}: unreadable ({exc})")
                continue

            frame.columns = [str(c).strip() for c in frame.columns]
            if len(frame.columns) != EXPECTED_RAW_COLUMNS or set(frame.columns) != set(RAW_COLUMNS):
                checks["columns_ok"] = False
                checks["bad_files"].append(
                    f"{path.name}: expected {EXPECTED_RAW_COLUMNS} columns, "
                    f"got {len(frame.columns)}"
                )
                continue

            try:
                validate_raw_frame(frame, patient_id=pid)
                onset = derive_onset(
                    frame[LABEL].to_numpy(), frame["ICULOS"].to_numpy()
                )
            except (GoldSchemaError, LabelError) as exc:
                message = str(exc)
                if "monotonic" in message:
                    checks["label_monotonic"] = False
                if "contiguous" in message:
                    checks["icu_hours_contiguous"] = False
                if len(checks["bad_files"]) < 20:
                    checks["bad_files"].append(message)
                continue

            if pid in seen:
                checks["duplicate_ids"] += 1
            seen.add(pid)
            sepsis_by_patient[pid] = onset.sepsis_patient

            patients += 1
            hours += len(frame)
            sepsis_patients += int(onset.sepsis_patient)

            if index % 2000 == 0:
                print(f"  {dir_name}: {index:,}/{len(to_check):,} verified")

        sets[set_name] = {
            "patients": patients,
            "hours": hours,
            "sepsis_patients": sepsis_patients,
        }
        total_patients += patients
        total_hours += hours

    sampled = sample is not None
    if sampled:
        print(
            f"WARNING: sampled verification ({sample} files per set). "
            "The data gate rejects sampled profiles."
        )

    prevalence = (
        sum(sets[s]["sepsis_patients"] for s in sets) / total_patients
        if total_patients
        else 0.0
    )

    split_counts = {"train": 0, "val": 0, "test": 0}
    overlap = 0  # assign_patient_splits asserts exactly one split per patient
    if total_patients >= 3:
        assignment = assign_patient_splits(
            list(sepsis_by_patient),
            sepsis_by_patient=sepsis_by_patient,
            seed=SPLIT_SEED,
        )
        for value in assignment.values():
            split_counts[value] += 1
        overlap = sum(1 for pid in set(sepsis_by_patient) if pid not in assignment)

    profile = {
        "dataset": "physionet-challenge-2019",
        "version": "1.0.0",
        "license": "CC BY 4.0",
        "doi": "10.13026/v64v-d857",
        "source": f"s3://{S3_BUCKET}/{S3_PREFIX.rstrip('/')}",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sampled": sampled,
        "sets": sets,
        "totals": {
            "patients": total_patients,
            "hours": total_hours,
            "sepsis_patients": sum(s["sepsis_patients"] for s in sets.values()),
            "prevalence": round(prevalence, 6),
        },
        "checks": checks,
        "split": {
            "ratios": list(SPLIT_RATIOS),
            "seed": SPLIT_SEED,
            **split_counts,
            "overlap": overlap,
        },
    }
    return profile


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=DATA_RAW)
    parser.add_argument("--source", type=Path, help="verify this existing copy")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--method", choices=["auto", "aws", "python"], default="auto")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--sample", type=int, help="verify only N files per set")
    parser.add_argument(
        "--profile", type=Path, default=ROOT / "artifacts" / "data_profile.json"
    )
    parser.add_argument("--no-profile", action="store_true")
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args(argv)
    if args.sample is not None and args.sample < 1:
        parser.error("--sample must be >= 1")

    source = args.source
    if source is None:
        source = args.dest
        if not args.verify_only:
            downloaded = False
            if args.method in ("auto", "aws"):
                downloaded = download_aws(args.dest)
            if not downloaded and args.method in ("auto", "python"):
                download_python(args.dest, args.workers, args.force)
            elif not downloaded and args.method == "aws":
                _fail(
                    "aws CLI not found or `aws s3 sync` failed. Install the AWS "
                    "CLI, or rerun with --method python."
                )

    profile = verify(source, args.sample)

    if not args.no_profile:
        args.profile.parent.mkdir(parents=True, exist_ok=True)
        args.profile.write_text(
            json.dumps(profile, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"\nprofile -> {args.profile}")

    print(
        f"patients={profile['totals']['patients']:,} "
        f"hours={profile['totals']['hours']:,} "
        f"prevalence={profile['totals']['prevalence']:.4f}"
    )
    if profile["checks"]["bad_files"]:
        print(f"bad files: {profile['checks']['bad_files'][:5]}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
