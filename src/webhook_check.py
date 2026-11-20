"""P1 check: deliver a pager payload to a disposable webhook and verify receipt.

    python -m src.webhook_check

Creates a webhook.site token, sends the Paged payload, then reads the received
requests back. No PHI; ids and a note hash only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

from .config import ARTIFACTS
from .transports import WebhookPager


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Webhook pager check")
    parser.add_argument("--out", type=Path, default=ARTIFACTS)
    args = parser.parse_args(argv)

    headers = {"Accept": "application/json"}
    with httpx.Client(timeout=20, headers=headers) as client:
        token = client.post("https://webhook.site/token", json={})
        token.raise_for_status()
        token_id = token.json()["uuid"]
        url = f"https://webhook.site/{token_id}"

        pager = WebhookPager(url, client=client)
        delivery = pager.send(
            {
                "alert_id": "pager-check-001",
                "patient_id": "p000206",
                "note": "De-identified demo SBAR; no PHI.",
            }
        )
        time.sleep(1.0)
        received = client.get(f"https://webhook.site/token/{token_id}/requests")
        received.raise_for_status()
        requests = received.json().get("data", [])

    payload = {
        "url": url,
        "delivery": delivery.as_dict(),
        "received_count": len(requests),
        "received_body": requests[0].get("content") if requests else None,
    }
    (args.out / "webhook_check.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    return 0 if delivery.delivered and requests else 2


if __name__ == "__main__":
    sys.exit(main())
