"""In-process latency check for `/score` (no network hop).

    python -m src.benchmark --n 200

This measures the API + model path as the process sees it, not a remote client
round-trip. The artefact records exactly that distinction.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

from .config import ARTIFACTS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure /score latency")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--patient", default="p000206")
    parser.add_argument("--hour", type=int, default=15)
    parser.add_argument("--out", type=Path, default=ARTIFACTS)
    args = parser.parse_args(argv)

    os.environ.setdefault("SEPSIS_APPROVAL_SECRET", "benchmark-secret-" + "x" * 24)
    os.environ.setdefault("SEPSIS_APPROVAL_API_KEY", "benchmark-key")

    from fastapi.testclient import TestClient

    from .api import app

    client = TestClient(app)
    health = client.get("/health")
    if health.status_code != 200:
        print(f"ERROR: /health -> {health.status_code}: {health.text}")
        return 2

    from . import api

    backend = api._state()["backend"]
    payload = {"patient_id": args.patient, "as_of_hour": args.hour}
    for _ in range(10):  # warm-up
        backend.score(args.patient, args.hour)
        client.post("/score", json=payload)

    def measure(call) -> dict:
        samples = []
        for _ in range(args.n):
            started = time.perf_counter()
            call()
            samples.append((time.perf_counter() - started) * 1000.0)
        samples.sort()
        return {
            "n": args.n,
            "p50_ms": round(statistics.median(samples), 3),
            "p95_ms": round(samples[int(0.95 * len(samples)) - 1], 3),
            "p99_ms": round(samples[int(0.99 * len(samples)) - 1], 3),
            "max_ms": round(max(samples), 3),
        }

    model_samples = measure(lambda: backend.score(args.patient, args.hour))

    def api_call():
        response = client.post("/score", json=payload)
        if response.status_code != 200:
            raise RuntimeError(f"/score -> {response.status_code}: {response.text}")

    api_samples = measure(api_call)
    result = {
        "model_score": model_samples,
        "api_score": api_samples,
        "note": (
            "model_score is the store+model path; api_score adds FastAPI/Pydantic "
            "and TestClient in-process overhead (no network hop)."
        ),
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "latency.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
