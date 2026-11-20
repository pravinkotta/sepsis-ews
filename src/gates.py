"""Four CI gates: data, model, grounding, authority.

Honesty rules:

* A missing artefact is ``SKIP`` on a fresh clone and ``FAIL`` when the gate is
  explicitly required (release / deploy).
* A gate never compares against a number that has not been measured. The
  floors are pre-registered in ``config.py`` and in SPEC.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import (
    ARTIFACTS,
    GATE_DATA,
    GATE_GROUNDING,
    GATE_MODEL,
)
from .redteam import run_all, surface_checks

DEFAULT_PATHS = {
    "data": ARTIFACTS / "data_profile.json",
    "model": ARTIFACTS / "metrics.json",
    "grounding": ARTIFACTS / "eval_scores.json",
}


@dataclass
class GateResult:
    gate: str
    status: str  # PASS | FAIL | SKIP
    checks: list[tuple[str, bool, str]] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "gate": self.gate,
            "status": self.status,
            "note": self.note,
            "checks": [
                {"name": name, "ok": ok, "detail": detail}
                for name, ok, detail in self.checks
            ],
        }


def _load(path: Path) -> tuple[dict | None, str | None]:
    if not path.exists():
        return None, f"artifact missing: {path}"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except json.JSONDecodeError as exc:
        return None, f"artifact is not valid JSON: {path}: {exc}"


def _finish(gate: str, checks: list[tuple[str, bool, str]]) -> GateResult:
    status = "PASS" if checks and all(ok for _, ok, _ in checks) else "FAIL"
    return GateResult(gate, status, checks)


def gate_data(path: Path = DEFAULT_PATHS["data"]) -> GateResult:
    profile, error = _load(path)
    if error:
        return GateResult("data", "SKIP", note=error)

    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append((name, bool(ok), detail))

    check("dataset", profile.get("dataset") == "physionet-challenge-2019",
          f"dataset={profile.get('dataset')!r}")
    check("not_sampled", profile.get("sampled") is False,
          f"sampled={profile.get('sampled')!r}")

    sets = profile.get("sets", {})
    for name in ("A", "B"):
        present = name in sets and int(sets[name].get("patients", 0)) > 0
        check(f"set_{name}_present", present, f"set {name}={sets.get(name)}")

    totals = profile.get("totals", {})
    patients = int(totals.get("patients", 0))
    check("patient_count", patients >= GATE_DATA["min_patients"],
          f"{patients} >= {GATE_DATA['min_patients']}")

    prevalence = float(totals.get("prevalence", -1))
    inside = GATE_DATA["min_prevalence"] <= prevalence <= GATE_DATA["max_prevalence"]
    check("prevalence_recorded", inside, f"prevalence={prevalence:.4f}")

    checks_block = profile.get("checks", {})
    check("columns_ok", checks_block.get("columns_ok") is True,
          f"columns_ok={checks_block.get('columns_ok')!r}")
    check("label_monotonic", checks_block.get("label_monotonic") is True,
          f"label_monotonic={checks_block.get('label_monotonic')!r}")
    check("icu_hours_contiguous", checks_block.get("icu_hours_contiguous") is True,
          f"icu_hours_contiguous={checks_block.get('icu_hours_contiguous')!r}")
    duplicates = int(checks_block.get("duplicate_ids", -1))
    check("no_duplicate_patients", duplicates == 0, f"duplicates={duplicates}")
    bad = checks_block.get("bad_files", [])
    check("no_bad_files", not bad, f"bad_files={bad[:3]}")

    split = profile.get("split", {})
    overlap = int(split.get("overlap", -1))
    check("split_disjoint", overlap == 0, f"overlap={overlap}")
    covered = sum(int(split.get(k, 0)) for k in ("train", "val", "test"))
    check("split_covers_cohort", covered == patients, f"{covered} == {patients}")

    return _finish("data", checks)


def gate_model(path: Path = DEFAULT_PATHS["model"]) -> GateResult:
    metrics, error = _load(path)
    if error:
        return GateResult("model", "SKIP", note=error)

    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append((name, bool(ok), detail))

    auroc = float(metrics.get("auroc", -1))
    check("auroc_floor", auroc >= GATE_MODEL["min_auroc"],
          f"{auroc:.4f} >= {GATE_MODEL['min_auroc']}")

    prevalence = float(metrics.get("prevalence", 0))
    auprc = float(metrics.get("auprc", -1))
    lift = (auprc / prevalence) if prevalence > 0 else -1
    check("auprc_lift", lift >= GATE_MODEL["min_auprc_lift"],
          f"lift={lift:.2f}x >= {GATE_MODEL['min_auprc_lift']}x "
          f"(auprc={auprc:.4f}, prevalence={prevalence:.4f})")

    policy = metrics.get("policy", {})
    sensitivity = float(policy.get("sensitivity", -1))
    check("sensitivity_at_budget",
          sensitivity >= GATE_MODEL["min_sensitivity_at_budget"],
          f"{sensitivity:.3f} >= {GATE_MODEL['min_sensitivity_at_budget']}")

    burden = float(policy.get("alerts_per_patient_day", 1e9))
    check("alert_budget_respected",
          burden <= GATE_MODEL["max_alerts_per_patient_day"],
          f"{burden:.2f} <= {GATE_MODEL['max_alerts_per_patient_day']} alerts/patient-day")

    lead = policy.get("median_lead_hours")
    lead_ok = lead is not None and float(lead) >= GATE_MODEL["min_median_lead_hours"]
    check("median_lead", lead_ok,
          f"lead={lead} h >= {GATE_MODEL['min_median_lead_hours']} h")

    calibration = metrics.get("calibration", {})
    brier = float(calibration.get("brier", 1e9))
    check("brier", brier <= GATE_MODEL["max_brier"],
          f"{brier:.4f} <= {GATE_MODEL['max_brier']}")

    slope = float(calibration.get("slope", -1))
    low, high = GATE_MODEL["calibration_slope"]
    check("calibration_slope", low <= slope <= high, f"slope={slope:.2f} in [{low}, {high}]")

    return _finish("model", checks)


def gate_grounding(path: Path = DEFAULT_PATHS["grounding"]) -> GateResult:
    scores, error = _load(path)
    if error:
        return GateResult("grounding", "SKIP", note=error)

    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append((name, bool(ok), detail))

    mapping = {
        "faithfulness": "min_faithfulness",
        "context_precision": "min_context_precision",
        "citation_validity": "min_citation_validity",
        "refusal_correctness": "min_refusal_correctness",
        "retrieval_recall_at_5": "min_retrieval_recall_at_5",
    }
    for metric, gate_name in mapping.items():
        value = float(scores.get(metric, -1))
        floor = GATE_GROUNDING[gate_name]
        check(metric, value >= floor, f"{value:.3f} >= {floor}")

    n_queries = int(scores.get("n_queries", 0))
    check("golden_set_size", n_queries >= 20, f"n_queries={n_queries} >= 20")

    return _finish("grounding", checks)


def gate_authority() -> GateResult:
    checks = [(r.name, r.passed, r.detail) for r in run_all()]
    checks += [(r.name, r.passed, r.detail) for r in surface_checks()]
    return _finish("authority", checks)


def run_gates(
    *,
    require: set[str] | None = None,
    paths: dict[str, Path] | None = None,
) -> list[GateResult]:
    require = require or set()
    paths = {**DEFAULT_PATHS, **(paths or {})}
    results = [
        gate_data(paths["data"]),
        gate_model(paths["model"]),
        gate_grounding(paths["grounding"]),
        gate_authority(),
    ]
    for result in results:
        if result.gate in require and result.status == "SKIP":
            result.status = "FAIL"
            result.note = f"required gate has no artifact: {result.note}"
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the four Sepsis EWS gates")
    parser.add_argument(
        "--require",
        default="",
        help="comma list of gates that must not SKIP (data,model,grounding,authority) or 'all'",
    )
    parser.add_argument("--json", type=Path, help="write results as JSON")
    args = parser.parse_args(argv)

    known = {"data", "model", "grounding", "authority"}
    if args.require.strip() == "all":
        require = known
    else:
        require = {x.strip() for x in args.require.split(",") if x.strip()}
    unknown = require - known
    if unknown:
        parser.error(f"unknown gates: {sorted(unknown)}")

    results = run_gates(require=require)
    for result in results:
        passed = sum(1 for _, ok, _ in result.checks if ok)
        total = len(result.checks)
        detail = result.note or f"{passed}/{total} checks"
        print(f"{result.status:<4}  {result.gate:<10} {detail}")
        if result.status != "PASS":
            for name, ok, text in result.checks:
                if not ok:
                    print(f"        - {name}: {text}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8"
        )

    return 1 if any(r.status == "FAIL" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
