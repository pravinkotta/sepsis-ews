"""Measure a 48-hour SSE replay against a real uvicorn server.

    python -m src.stream_check

Starts uvicorn on a local port, waits for /health, streams the SSE endpoint with
urllib, counts events, then stops the server. TestClient buffers SSE and gives
misleading latency; this is the transport the browser actually uses.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from .config import ARTIFACTS


def _wait_for_health(base: str, timeout: float = 40.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(base + "/health", timeout=3) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError):
            time.sleep(1.0)
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="48 h SSE replay check")
    parser.add_argument("--patient", default="p000206")
    parser.add_argument("--start", type=int, default=2)
    parser.add_argument("--end", type=int, default=49)
    parser.add_argument("--port", type=int, default=8031)
    parser.add_argument("--out", type=Path, default=ARTIFACTS)
    args = parser.parse_args(argv)

    env = {
        **os.environ,
        "SEPSIS_APPROVAL_SECRET": "stream-check-secret-" + "x" * 20,
        "SEPSIS_APPROVAL_API_KEY": "stream-check-key",
    }
    process = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn", "src.api:app",
            "--host", "127.0.0.1", "--port", str(args.port),
            "--log-level", "warning",
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{args.port}"
    try:
        if not _wait_for_health(base):
            print("ERROR: uvicorn did not become healthy", file=sys.stderr)
            return 2
        url = (
            f"{base}/stream?patient_id={args.patient}"
            f"&start_hour={args.start}&end_hour={args.end}"
        )
        started = time.perf_counter()
        events = 0
        with urllib.request.urlopen(url, timeout=120) as response:
            if response.status != 200:
                print(f"ERROR: /stream -> {response.status}", file=sys.stderr)
                return 2
            for line in response:
                if line.startswith(b"event: risk"):
                    events += 1
        elapsed = time.perf_counter() - started
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()

    result = {
        "patient_id": args.patient,
        "start_hour": args.start,
        "end_hour": args.end,
        "events": events,
        "expected_events": args.end - args.start + 1,
        "duration_seconds": round(elapsed, 3),
        "ms_per_event": round(1000 * elapsed / max(events, 1), 2),
        "transport": "uvicorn + urllib SSE (real HTTP)",
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (args.out / "stream_check.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0 if events == result["expected_events"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
