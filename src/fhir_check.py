"""E1 check: write a de-identified SBAR `Communication` to a public FHIR sandbox.

    python -m src.fhir_check

Uses HAPI's public R4 sandbox with a random demo patient id. No PHI.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from .config import ARTIFACTS
from .transports import FhirEhr


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FHIR sandbox write check")
    parser.add_argument("--base", default="https://hapi.fhir.org/baseR4")
    parser.add_argument("--out", type=Path, default=ARTIFACTS)
    args = parser.parse_args(argv)

    patient_id = f"sepsis-demo-{uuid.uuid4().hex[:8]}"
    ehr = FhirEhr(args.base, patient_id=patient_id)
    note = (
        "SBAR — de-identified demo\n"
        "S: ICU hour 15, calibrated sepsis risk 3.9%.\n"
        "B: HR 109, lactate 3.1, creatinine 5.1.\n"
        "A: Concerning for evolving sepsis [S1].\n"
        "R: Consider clinician review and local escalation [S2].\n"
        "This is an AI draft, not a diagnosis or order."
    )
    delivery = ehr.write_note(
        {"alert_id": f"fhir-{uuid.uuid4().hex[:8]}", "patient_id": patient_id, "note": note}
    )
    payload = {"base": args.base, **delivery.as_dict()}
    (args.out / "fhir_check.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    return 0 if delivery.delivered else 2


if __name__ == "__main__":
    sys.exit(main())
