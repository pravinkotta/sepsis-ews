"""End-to-end replay: score → retrieve → cited SBAR → human approval → page.

    python -m src.demo                 # extractive (no key), local demo transport
    python -m src.demo --llm           # requires OPENAI_API_KEY

The agent stops at `pending_approval`. The demo then attempts the write without
a token (and shows the denial), mints a token as a human would, and executes the
page through the tool registry. No real pager or EHR is touched.
"""

from __future__ import annotations

import argparse
import json
import secrets
import sys
from pathlib import Path

from .agent import execute_approved, run_agent
from .approval import ApprovalError
from .config import ARTIFACTS
from .generate import OpenAIClient
from .runtime import load_backend, make_registry
from .tools import ApprovalRequired


def pick_case(backend, gold) -> tuple[str, int, float]:
    """Choose a pre-onset hour for a septic test patient with the highest risk."""
    index = gold[
        ["patient_id", "split", "is_sepsis_patient", "t_sepsis_est", "onset_censored"]
    ].drop_duplicates("patient_id")
    candidates = index[
        (index["split"] == "test")
        & (index["is_sepsis_patient"] == 1)
        & index["t_sepsis_est"].notna()
        & (index["onset_censored"] == 0)
    ].head(40)

    best: tuple[str, int, float] | None = None
    for _, row in candidates.iterrows():
        onset = int(row["t_sepsis_est"])
        for offset in (6, 8, 10):
            hour = onset - offset
            if hour < 1:
                continue
            try:
                score = backend.score(row["patient_id"], hour)
            except Exception:  # noqa: BLE001 - missing early hours are expected
                continue
            if best is None or score["risk"] > best[2]:
                best = (str(row["patient_id"]), hour, float(score["risk"]))
            if score["risk"] >= score["threshold"]:
                return str(row["patient_id"]), hour, float(score["risk"])
    if best is None:
        raise RuntimeError("no usable septic test patient with an early hour")
    return best


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Sepsis EWS end-to-end demo")
    parser.add_argument("--gold", type=Path, default=ARTIFACTS / "gold_hourly.parquet")
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    parser.add_argument("--patient")
    parser.add_argument("--hour", type=int)
    parser.add_argument("--llm", action="store_true", help="use OpenAI instead of extractive")
    args = parser.parse_args(argv)

    client = OpenAIClient() if args.llm else None
    try:
        backend = load_backend(
            gold_path=args.gold, artifacts_dir=args.artifacts, llm_client=client
        )
    except Exception as exc:  # noqa: BLE001 - startup failure must be actionable
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    gold = backend.gold
    if args.patient and args.hour:
        patient_id, hour = args.patient, args.hour
    else:
        patient_id, hour, risk = pick_case(backend, gold)
        print(f"case: {patient_id} at ICU hour {hour} (risk {risk:.4f})")

    state = run_agent(patient_id, hour, backend)
    print(f"agent status: {state['status']}")
    if state["status"] != "pending_approval":
        print(f"reason: {state.get('reason')}")
        (args.artifacts / "demo_run.json").write_text(
            json.dumps({"state": state}, indent=2, default=str), encoding="utf-8"
        )
        return 0

    sbar = state["sbar"]
    print("\nSBAR")
    for section in ("situation", "background", "assessment", "recommendation"):
        print(f"  {section.upper()}: {sbar[section]}")
    print(f"  validation: {state['validation']}")
    print(f"  generator: {state['generator']}")

    # A fresh secret per demo run; in deployment it lives in the approval service.
    secret = secrets.token_bytes(32)
    registry, authority = make_registry(
        backend, secret=secret, artifacts_dir=args.artifacts
    )

    print("\nAttempting the write without approval ...")
    denied = None
    try:
        execute_approved(state, registry=registry, approval_token=None)
    except (ApprovalRequired, ApprovalError) as exc:
        denied = f"{type(exc).__name__}: {exc}"
        print(f"  denied: {denied}")

    token = authority.approve(
        alert_id=state["alert_id"],
        patient_id=state["patient_id"],
        tool="page_rapid_response",
        approver="demo.clinician",
    )
    print("Human approval granted (token not displayed).")
    result = execute_approved(state, registry=registry, approval_token=token)
    print(f"page result: {result['status']} via {result['transport']}")

    payload = {
        "case": {"patient_id": patient_id, "as_of_hour": hour},
        "state": state,
        "denied_without_token": denied,
        "page": result,
        "audit": registry.audit.records,
    }
    (args.artifacts / "demo_run.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    print(f"\ndemo trace -> {args.artifacts / 'demo_run.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
