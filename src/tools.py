"""The tool surface the agent may call — and the boundary it may not cross.

Three classes of tool:

* ``none``    — read-only. Safe for the agent to call freely.
* ``propose`` — produces a draft. No side effect until a human approves.
* ``write``   — irreversible. Requires a single-use approval token bound to
                this alert, this patient and this tool.

Decision tools (start antibiotics, order cultures, admit, discharge, suppress
an alert) **do not exist**. Their absence is test-asserted: it is a structural
boundary, not a prompt instruction.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from .approval import ApprovalAuthority, ApprovalError

# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class ToolError(Exception):
    """Base class for tool-call failures."""


class UnknownTool(ToolError):
    """Tool is not registered."""


class ForbiddenTool(ToolError):
    """Tool is on the explicit no-such-capability list."""


class ToolInputError(ToolError):
    """Required argument missing or malformed."""


class ApprovalRequired(ToolError):
    """A write tool was called without an approval token."""


class BackendNotConfigured(ToolError):
    """The service is running without its data/model plane (fail loud)."""


# --------------------------------------------------------------------------- #
# Tool registry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    side_effect: str  # none | propose | write
    requires_approval: bool
    required: tuple[str, ...]
    properties: dict[str, str]


def _schema(required: tuple[str, ...], properties: dict[str, str]) -> dict:
    return {
        "type": "object",
        "required": list(required),
        "properties": {k: {"type": v} for k, v in properties.items()},
    }


READ_TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec(
        "get_patient_state",
        "Latest vitals, labs, demographics and model features up to as_of_hour.",
        "none",
        False,
        ("patient_id", "as_of_hour"),
        {"patient_id": "string", "as_of_hour": "integer"},
    ),
    ToolSpec(
        "get_recent_labs",
        "Lab values in the last N hours, with hours since each measurement.",
        "none",
        False,
        ("patient_id", "as_of_hour"),
        {"patient_id": "string", "as_of_hour": "integer", "hours": "integer"},
    ),
    ToolSpec(
        "score_sepsis",
        "Calibrated sepsis risk for this hour, with top drivers if available.",
        "none",
        False,
        ("patient_id", "as_of_hour"),
        {"patient_id": "string", "as_of_hour": "integer"},
    ),
    ToolSpec(
        "retrieve_guideline",
        "Retrieve top-k Surviving Sepsis / local escalation chunks with ids.",
        "none",
        False,
        ("query",),
        {"query": "string", "k": "integer"},
    ),
    ToolSpec(
        "get_alert_history",
        "Prior alerts and their approval decisions for this patient.",
        "none",
        False,
        ("patient_id",),
        {"patient_id": "string"},
    ),
)

PROPOSE_TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec(
        "draft_sbar",
        "Draft a cited SBAR from the current patient state. No side effect.",
        "propose",
        False,
        ("patient_id", "as_of_hour"),
        {"patient_id": "string", "as_of_hour": "integer"},
    ),
    ToolSpec(
        "propose_escalation",
        "Record a proposed escalation for human review. Does not page anyone.",
        "propose",
        False,
        ("patient_id", "as_of_hour", "reason", "citations"),
        {
            "patient_id": "string",
            "as_of_hour": "integer",
            "reason": "string",
            "citations": "array",
        },
    ),
)

WRITE_TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec(
        "page_rapid_response",
        "Page the rapid response team. Requires a human approval token.",
        "write",
        True,
        ("alert_id", "patient_id", "note"),
        {"alert_id": "string", "patient_id": "string", "note": "string"},
    ),
    ToolSpec(
        "write_ehr_note",
        "Write an escalation note to the EHR. Requires a human approval token.",
        "write",
        True,
        ("alert_id", "patient_id", "note"),
        {"alert_id": "string", "patient_id": "string", "note": "string"},
    ),
)

# Capabilities that must never exist. Kept explicit so a future contributor
# cannot add one without breaking the authority gate.
FORBIDDEN_TOOLS = frozenset(
    {
        "start_antibiotics",
        "order_cultures",
        "order_labs",
        "admit_to_icu",
        "discharge_patient",
        "disable_alert",
        "override_alert",
        "approve_alert",
        "deny_alert",
        "enroll_trial",
        "place_order",
    }
)

TOOLS: dict[str, ToolSpec] = {t.name: t for t in READ_TOOLS + PROPOSE_TOOLS + WRITE_TOOLS}
if set(TOOLS) & FORBIDDEN_TOOLS:  # pragma: no cover - import-time contract
    raise RuntimeError("a forbidden capability was registered as a tool")


# --------------------------------------------------------------------------- #
# Audit
# --------------------------------------------------------------------------- #


@dataclass
class AuditLog:
    """Append-only record of every tool call and its outcome.

    Note text is never stored verbatim; its SHA-256 and length are recorded so
    an auditor can match a note without copying PHI into a log.
    """

    path: Path | None = None
    records: list[dict] = field(default_factory=list)

    def record(self, **fields: Any) -> None:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "monotonic": round(time.monotonic(), 3),
            **fields,
        }
        self.records.append(entry)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, sort_keys=True) + "\n")


def _redact(args: dict) -> dict:
    safe = {}
    for key, value in args.items():
        if key == "note" and isinstance(value, str):
            safe[key] = {
                "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest()[:16],
                "chars": len(value),
            }
        else:
            safe[key] = value
    return safe


# --------------------------------------------------------------------------- #
# Backend protocol
# --------------------------------------------------------------------------- #


class Backend(Protocol):
    def patient_state(self, patient_id: str, as_of_hour: int) -> dict: ...
    def recent_labs(self, patient_id: str, as_of_hour: int, hours: int) -> dict: ...
    def score(self, patient_id: str, as_of_hour: int) -> dict: ...
    def retrieve(self, query: str, k: int) -> dict: ...
    def alert_history(self, patient_id: str) -> dict: ...
    def draft_sbar(self, patient_id: str, as_of_hour: int) -> dict: ...
    def propose_escalation(
        self, patient_id: str, as_of_hour: int, reason: str, citations: list
    ) -> dict: ...
    def page_rapid_response(self, alert_id: str, patient_id: str, note: str) -> dict: ...
    def write_ehr_note(self, alert_id: str, patient_id: str, note: str) -> dict: ...


class UnavailableBackend:
    """Used until the data/model plane is wired. Every method fails loudly."""

    def _fail(self) -> dict:
        raise BackendNotConfigured(
            "backend is not configured: build the Gold table and model first "
            "(see README 'What exists' and SPEC.md phase 3). Refusing to "
            "return an ungrounded answer."
        )

    def patient_state(self, *args, **kwargs) -> dict:
        return self._fail()

    def recent_labs(self, *args, **kwargs) -> dict:
        return self._fail()

    def score(self, *args, **kwargs) -> dict:
        return self._fail()

    def retrieve(self, *args, **kwargs) -> dict:
        return self._fail()

    def alert_history(self, *args, **kwargs) -> dict:
        return self._fail()

    def draft_sbar(self, *args, **kwargs) -> dict:
        return self._fail()

    def propose_escalation(self, *args, **kwargs) -> dict:
        return self._fail()

    def page_rapid_response(self, *args, **kwargs) -> dict:
        return self._fail()

    def write_ehr_note(self, *args, **kwargs) -> dict:
        return self._fail()


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


class ToolRegistry:
    def __init__(
        self,
        backend: Backend,
        authority: ApprovalAuthority,
        audit: AuditLog | None = None,
    ) -> None:
        self.backend = backend
        self.authority = authority
        self.audit = audit or AuditLog()

    def call(
        self,
        name: str,
        args: dict | None = None,
        *,
        approval_token: str | None = None,
        actor: str = "agent",
    ) -> dict:
        args = dict(args or {})
        try:
            spec = self._resolve(name)
            self._check_inputs(spec, args)
            if spec.requires_approval:
                if not approval_token:
                    raise ApprovalRequired(
                        f"{name} is an irreversible action; a human approval "
                        f"token bound to this alert and patient is required"
                    )
                self.authority.consume(
                    approval_token,
                    alert_id=str(args["alert_id"]),
                    patient_id=str(args["patient_id"]),
                    tool=name,
                )
            result = self._dispatch(name, args)
        except ToolError as exc:
            self.audit.record(
                tool=name, actor=actor, args=_redact(args), outcome="denied",
                error=type(exc).__name__, detail=str(exc),
            )
            raise
        except ApprovalError as exc:
            self.audit.record(
                tool=name, actor=actor, args=_redact(args), outcome="denied",
                error=type(exc).__name__, detail=str(exc),
            )
            raise
        self.audit.record(
            tool=name, actor=actor, args=_redact(args), outcome="ok",
            side_effect=spec.side_effect,
        )
        return result

    @staticmethod
    def _resolve(name: str) -> ToolSpec:
        if name in FORBIDDEN_TOOLS:
            raise ForbiddenTool(
                f"{name} is not a capability of this system by design; the "
                f"agent may propose, never decide"
            )
        spec = TOOLS.get(name)
        if spec is None:
            raise UnknownTool(f"unknown tool: {name}")
        return spec

    @staticmethod
    def _check_inputs(spec: ToolSpec, args: dict) -> None:
        missing = [key for key in spec.required if key not in args]
        if missing:
            raise ToolInputError(f"{spec.name} missing required args: {missing}")

    def _dispatch(self, name: str, args: dict) -> dict:
        backend = self.backend
        if name == "get_patient_state":
            return backend.patient_state(args["patient_id"], args["as_of_hour"])
        if name == "get_recent_labs":
            return backend.recent_labs(
                args["patient_id"], args["as_of_hour"], int(args.get("hours", 24))
            )
        if name == "score_sepsis":
            return backend.score(args["patient_id"], args["as_of_hour"])
        if name == "retrieve_guideline":
            return backend.retrieve(args["query"], int(args.get("k", 5)))
        if name == "get_alert_history":
            return backend.alert_history(args["patient_id"])
        if name == "draft_sbar":
            return backend.draft_sbar(args["patient_id"], args["as_of_hour"])
        if name == "propose_escalation":
            return backend.propose_escalation(
                args["patient_id"],
                args["as_of_hour"],
                args["reason"],
                args["citations"],
            )
        if name == "page_rapid_response":
            return backend.page_rapid_response(
                args["alert_id"], args["patient_id"], args["note"]
            )
        if name == "write_ehr_note":
            return backend.write_ehr_note(
                args["alert_id"], args["patient_id"], args["note"]
            )
        raise UnknownTool(f"no dispatch for {name}")


def tool_schemas() -> list[dict]:
    """MCP/JSON-schema view of the surface, for docs and registration checks."""
    return [
        {
            "name": spec.name,
            "description": spec.description,
            "side_effect": spec.side_effect,
            "requires_approval": spec.requires_approval,
            "inputSchema": json.loads(json.dumps(_schema(spec.required, spec.properties))),
        }
        for spec in TOOLS.values()
    ]


def assert_tool_surface() -> list[tuple[str, bool, str]]:
    """Static checks used by the authority gate. Returns (name, ok, detail)."""
    checks: list[tuple[str, bool, str]] = []

    overlap = sorted(set(TOOLS) & FORBIDDEN_TOOLS)
    checks.append(("no_forbidden_tool_registered", not overlap, f"overlap={overlap}"))

    writes = [s for s in TOOLS.values() if s.side_effect == "write"]
    unguarded = [s.name for s in writes if not s.requires_approval]
    checks.append(("all_writes_require_approval", not unguarded, f"unguarded={unguarded}"))

    unbound = [
        s.name for s in writes if not {"alert_id", "patient_id"} <= set(s.required)
    ]
    checks.append(("all_writes_are_bound", not unbound, f"unbound={unbound}"))

    decision_prefixes = ("start_", "order_", "admit_", "discharge_", "enroll_", "place_")
    shadow = sorted(
        name for name in TOOLS if name.startswith(decision_prefixes)
    )
    checks.append(("no_decision_tool_names", not shadow, f"names={shadow}"))

    empty = [s.name for s in TOOLS.values() if len(s.description.strip()) < 10]
    checks.append(("every_tool_documented", not empty, f"empty={empty}"))

    return checks
