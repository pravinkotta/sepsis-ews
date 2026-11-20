"""Verify P1 and E1 through the local API, not just the transport classes.

    python scripts/local_transport_check.py

Uses the shared demo key from `.env`; prints no token or key.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8010"
ENV = ROOT / ".env"


def _env(key: str) -> str:
    for line in ENV.read_text(encoding="utf-8").splitlines():
        if line.startswith(key + "="):
            return line.split("=", 1)[1]
    return ""


def main() -> int:
    key = _env("SEPSIS_APPROVAL_API_KEY")
    if not key:
        print("ERROR: SEPSIS_APPROVAL_API_KEY missing from .env")
        return 2

    session = requests.Session()
    agent = session.post(
        f"{BASE}/agent/run",
        json={"patient_id": "p000206", "as_of_hour": 15},
        timeout=120,
    ).json()
    alert_id = agent["alert_id"]

    def approve(tool: str) -> str:
        return session.post(
            f"{BASE}/alerts/{alert_id}/approve",
            json={"patient_id": "p000206", "tool": tool, "approver": "local.check"},
            headers={"X-API-Key": key},
            timeout=30,
        ).json()["approval_token"]

    page_token = approve("page_rapid_response")
    page = session.post(
        f"{BASE}/alerts/{alert_id}/page",
        json={"patient_id": "p000206", "approval_token": page_token},
        timeout=30,
    )

    ehr_token = approve("write_ehr_note")
    ehr = session.post(
        f"{BASE}/alerts/{alert_id}/ehr-note",
        json={"patient_id": "p000206", "approval_token": ehr_token},
        timeout=60,
    )

    payload = {
        "agent": agent.get("status"),
        "page": {"status": page.status_code, "body": page.json()},
        "ehr": {"status": ehr.status_code, "body": ehr.json()},
    }
    (ROOT / "artifacts" / "p1_e1_enable.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({
        "page_status": page.status_code,
        "page_transport": page.json().get("transport"),
        "page_delivered": page.json().get("delivered"),
        "ehr_status": ehr.status_code,
        "ehr_transport": ehr.json().get("transport"),
        "ehr_delivered": ehr.json().get("delivered"),
    }, indent=2))
    return 0 if (
        agent.get("status") == "pending_approval"
        and page.status_code == 200 and page.json().get("delivered")
        and ehr.status_code == 200 and ehr.json().get("delivered")
    ) else 1


if __name__ == "__main__":
    sys.exit(main())
