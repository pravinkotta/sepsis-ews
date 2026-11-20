"""Deterministic SBAR validator.

The LLM drafts; this module decides whether the draft may reach a human as-is.
No model call happens here. The two load-bearing rules:

1. Every ``[S#]`` citation must resolve to a chunk that was actually retrieved
   for this patient. A fabricated citation is rejected, not softened.
2. The draft may recommend *considering* a bundle action, but may not issue a
   treatment order. Orders are a clinician act; language that looks like one is
   flagged for review.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

REQUIRED_SECTIONS = ("situation", "background", "assessment", "recommendation")
CITATION_RE = re.compile(r"\[(S\d+)\]", re.IGNORECASE)

# Targeted imperatives, not a general verb ban ("in order to" is harmless).
TREATMENT_ORDER_RE = re.compile(
    r"\b(?:start|begin|administer|give|push|bolus|intubate|transfuse)\b[^.]{0,40}"
    r"\b(?:antibiotic|antibiotics|vasopressor|vasopressors|fluids|noradrenaline|"
    r"norepinephrine|oxygen|blood|cultures?|sedation)\b",
    re.IGNORECASE,
)
DISCHARGE_ORDER_RE = re.compile(
    r"\bdischarge\s+(?:the\s+)?patient\b", re.IGNORECASE
)
FINALIZATION_RE = re.compile(
    r"\b(?:sepsis\s+(?:is\s+)?confirmed|confirmed\s+sepsis|"
    r"diagnosis\s+(?:is\s+)?confirmed|final\s+diagnosis)\b",
    re.IGNORECASE,
)

SBAR_SCHEMA = {
    "type": "object",
    "required": list(REQUIRED_SECTIONS),
    "properties": {
        "situation": {"type": "string"},
        "background": {"type": "string"},
        "assessment": {"type": "string"},
        "recommendation": {"type": "string"},
    },
}


@dataclass
class ValidationResult:
    status: str  # valid | needs_review | rejected
    flags: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "valid"

    def as_dict(self) -> dict:
        return {"status": self.status, "flags": list(self.flags)}


def _normalise_ids(retrieved_ids: Iterable[str]) -> set[str]:
    return {str(x).strip().upper() for x in retrieved_ids}


def validate_sbar(sbar: dict, *, retrieved_ids: Iterable[str]) -> ValidationResult:
    """Validate a drafted SBAR against the chunks retrieved for this patient."""
    if not isinstance(sbar, dict):
        return ValidationResult("rejected", ["not_an_object"])

    allowed = _normalise_ids(retrieved_ids)
    flags: list[str] = []
    status = "valid"

    def review(flag: str) -> None:
        nonlocal status
        flags.append(flag)
        if status == "valid":
            status = "needs_review"

    def reject(flag: str) -> None:
        nonlocal status
        flags.append(flag)
        status = "rejected"

    for section in REQUIRED_SECTIONS:
        value = sbar.get(section)
        if not isinstance(value, str) or not value.strip():
            reject(f"missing_section:{section}")
            continue

    cited: set[str] = set()
    for section in REQUIRED_SECTIONS:
        value = sbar.get(section)
        if not isinstance(value, str):
            continue
        found = {m.upper() for m in CITATION_RE.findall(value)}
        for citation in sorted(found - allowed):
            reject(f"citation_forgery:{citation}")
        cited |= found

    if status != "rejected":
        for section in ("assessment", "recommendation"):
            value = sbar.get(section, "")
            if isinstance(value, str) and value.strip() and not CITATION_RE.search(value):
                review(f"uncited_claim:{section}")

    joined = " ".join(
        sbar.get(s, "") for s in REQUIRED_SECTIONS if isinstance(sbar.get(s), str)
    )
    if TREATMENT_ORDER_RE.search(joined) or DISCHARGE_ORDER_RE.search(joined):
        review("treatment_order")
    if FINALIZATION_RE.search(joined):
        review("finalization_language")

    return ValidationResult(status, sorted(set(flags)))
