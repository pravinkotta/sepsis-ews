"""Enable P1 (webhook) and E1 (HAPI FHIR) in the local `.env`.

    python scripts/setup_transports.py

Creates a disposable webhook.site endpoint and points the app at the public HAPI
FHIR R4 sandbox. No real pager key or PHI is used.
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / ".env"


def _upsert(lines: list[str], key: str, value: str) -> None:
    for index, line in enumerate(lines):
        if line.startswith(key + "="):
            lines[index] = f"{key}={value}"
            return
    lines.append(f"{key}={value}")


def main() -> int:
    try:
        response = httpx.post("https://webhook.site/token", json={}, timeout=30)
        response.raise_for_status()
        token = response.json()["uuid"]
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: could not create webhook.site token: {exc}", file=sys.stderr)
        return 2

    url = f"https://webhook.site/{token}"
    lines = ENV.read_text(encoding="utf-8").splitlines() if ENV.exists() else []
    _upsert(lines, "SEPSIS_PAGER", "webhook")
    _upsert(lines, "SEPSIS_PAGER_WEBHOOK_URL", url)
    _upsert(lines, "SEPSIS_EHR", "fhir")
    _upsert(lines, "SEPSIS_FHIR_BASE", "https://hapi.fhir.org/baseR4")
    _upsert(lines, "SEPSIS_FHIR_PATIENT_PREFIX", "sepsis-demo-")
    lines = [line for line in lines if not line.startswith("SEPSIS_FHIR_PATIENT=")]
    ENV.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    print(f"wrote {ENV}")
    print(f"P1 webhook: {url}")
    print("E1 FHIR: https://hapi.fhir.org/baseR4 (prefix sepsis-demo-)")
    print("Restart the local app to load the transports.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
