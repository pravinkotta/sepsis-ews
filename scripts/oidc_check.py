"""End-to-end T1 check through the local Dex + oauth2-proxy stack.

    python scripts/oidc_check.py

Logs in as the demo clinician, then verifies `/me`, agent, approval, page, and
replay denial. Writes `artifacts/oidc_check.json`; never prints cookies/tokens.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://localhost:8011"
EMAIL = "clinician@example.com"
PASSWORD = "sepsis-demo"


def main() -> int:
    session = requests.Session()
    login_page = session.get(f"{BASE}/health", allow_redirects=True, timeout=30)
    soup = BeautifulSoup(login_page.text, "html.parser")
    form = soup.find("form")
    if form is None:
        print("ERROR: no Dex login form; final URL:", login_page.url)
        print(login_page.text[:800])
        return 2

    action = urljoin(login_page.url, form.get("action") or login_page.url)
    data = {
        field.get("name"): field.get("value", "")
        for field in form.find_all("input")
        if field.get("name")
    }
    data["login"] = EMAIL
    data["password"] = PASSWORD

    after_login = session.post(action, data=data, allow_redirects=True, timeout=30)

    # Dex may show a consent screen even with skipApprovalScreen; grant it.
    if "/dex/approval" in after_login.url:
        consent = BeautifulSoup(after_login.text, "html.parser").find("form")
        if consent is not None:
            consent_data = {
                field.get("name"): field.get("value", "")
                for field in consent.find_all("input")
                if field.get("name")
            }
            consent_data.setdefault("approval", "approve")
            consent_action = urljoin(
                after_login.url, consent.get("action") or after_login.url
            )
            after_login = session.post(
                consent_action, data=consent_data, allow_redirects=True, timeout=30
            )

    me = session.get(f"{BASE}/me", timeout=15)
    if me.status_code != 200:
        print("ERROR: /me after login:", me.status_code, me.text[:300])
        print("final URL:", after_login.url)
        return 2

    agent = session.post(
        f"{BASE}/agent/run",
        json={"patient_id": "p000206", "as_of_hour": 15},
        timeout=120,
    )
    agent_body = agent.json()
    alert_id = agent_body.get("alert_id")

    approve = session.post(
        f"{BASE}/alerts/{alert_id}/approve",
        json={"patient_id": "p000206"},
        timeout=30,
    )
    approve_body = approve.json()
    token = approve_body.get("approval_token")

    page = session.post(
        f"{BASE}/alerts/{alert_id}/page",
        json={"patient_id": "p000206", "approval_token": token},
        timeout=30,
    )
    replay = session.post(
        f"{BASE}/alerts/{alert_id}/page",
        json={"patient_id": "p000206", "approval_token": token},
        timeout=30,
    )

    approve_ehr = session.post(
        f"{BASE}/alerts/{alert_id}/approve",
        json={"patient_id": "p000206", "tool": "write_ehr_note"},
        timeout=30,
    )
    ehr = session.post(
        f"{BASE}/alerts/{alert_id}/ehr-note",
        json={"patient_id": "p000206", "approval_token": approve_ehr.json()["approval_token"]},
        timeout=60,
    )

    payload = {
        "proxy": BASE,
        "login_final_url": after_login.url,
        "me": me.json(),
        "agent": {"status": agent_body.get("status"), "validation": agent_body.get("validation")},
        "approve": {
            "status_code": approve.status_code,
            "approver": approve_body.get("approver"),
        },
        "page": {"status_code": page.status_code, "body": page.json() if page.ok else page.text[:200]},
        "replay": {"status_code": replay.status_code},
        "ehr": {"status_code": ehr.status_code, "body": ehr.json() if ehr.ok else ehr.text[:200]},
    }
    (ROOT / "artifacts" / "oidc_check.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({key: payload[key] for key in ("me", "agent", "approve", "page", "replay", "ehr")}, indent=2))
    ok = (
        me.json().get("auth_mode") == "proxy"
        and me.json().get("user") == EMAIL
        and agent_body.get("status") == "pending_approval"
        and approve.status_code == 200
        and approve_body.get("approver") == EMAIL
        and page.status_code == 200
        and page.json().get("delivered") is True
        and replay.status_code == 403
        and ehr.status_code == 200
        and ehr.json().get("delivered") is True
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
