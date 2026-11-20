import pytest

from src.agent import AgentError, execute_approved, run_agent
from src.approval import ApprovalAuthority, TokenReplayed
from src.generate import draft_sbar
from src.tools import ApprovalRequired, AuditLog, ToolRegistry

SECRET = b"agent-test-secret-with-at-least-32-bytes"


class FakeBackend:
    def __init__(self, risk=0.5, forged=False):
        self.risk = risk
        self.forged = forged
        self.page_calls = []

    def score(self, patient_id, as_of_hour):
        return {
            "risk": self.risk,
            "threshold": 0.0062,
            "top_drivers": [{"feature": "Lactate_last", "value": 3.1}],
            "facts": {"HR": 112, "Lactate": 3.1, "MAP": 61},
        }

    def retrieve(self, query, k):
        return {
            "chunks": [
                {
                    "citation_id": "S1",
                    "doc": "ssc-2021",
                    "title": "SSC",
                    "text": "Sepsis is life-threatening; measure lactate.",
                    "source": "test",
                },
                {
                    "citation_id": "S2",
                    "doc": "local-escalation-policy",
                    "title": "Policy",
                    "text": "SBAR rapid response requires an approval token.",
                    "source": "test",
                },
            ][:k]
        }

    def draft_sbar(self, patient_id, as_of_hour, chunks=None):
        chunks = chunks or self.retrieve("", 2)["chunks"]
        patient = {
            "patient_id": patient_id,
            "as_of_hour": as_of_hour,
            "risk": self.risk,
            "threshold": 0.0062,
            "drivers": [{"feature": "Lactate_last"}],
            "facts": {"HR": 112, "Lactate": 3.1},
        }
        draft = draft_sbar(patient, chunks)
        if self.forged:
            from src.validate import validate_sbar

            draft.sbar["assessment"] = "Definite sepsis [S9]."
            draft.validation = validate_sbar(
                draft.sbar, retrieved_ids={c["citation_id"] for c in chunks}
            )
        return draft

    def propose_escalation(self, patient_id, as_of_hour, reason, citations):
        return {"alert_id": "a1", "patient_id": patient_id, "status": "proposed"}

    def page_rapid_response(self, alert_id, patient_id, note):
        self.page_calls.append((alert_id, patient_id))
        return {"alert_id": alert_id, "status": "paged"}


def make_registry(backend):
    authority = ApprovalAuthority(SECRET)
    return ToolRegistry(backend, authority, AuditLog()), authority


def test_high_risk_stops_at_pending_approval():
    backend = FakeBackend(risk=0.5)
    state = run_agent("p1", 12, backend)
    assert state["status"] == "pending_approval"
    assert state["alert_id"] == "a1"
    assert state["sbar"]["assessment"]
    assert backend.page_calls == []  # never pages during the run


def test_low_risk_does_not_draft_or_propose():
    backend = FakeBackend(risk=0.0)
    state = run_agent("p1", 12, backend)
    assert state["status"] == "no_alert"
    assert "sbar" not in state
    assert backend.page_calls == []


def test_rejected_draft_does_not_reach_proposal():
    backend = FakeBackend(risk=0.5, forged=True)
    state = run_agent("p1", 12, backend)
    assert state["status"] == "rejected"
    assert "citation_forgery:S9" in state["reason"]


def test_execute_without_token_is_denied():
    backend = FakeBackend()
    state = run_agent("p1", 12, backend)
    registry, _ = make_registry(backend)
    with pytest.raises(ApprovalRequired):
        execute_approved(state, registry=registry, approval_token=None)
    assert backend.page_calls == []


def test_execute_with_token_pages_once():
    backend = FakeBackend()
    state = run_agent("p1", 12, backend)
    registry, authority = make_registry(backend)
    token = authority.approve(
        alert_id="a1", patient_id="p1", tool="page_rapid_response", approver="rn.kim"
    )
    result = execute_approved(state, registry=registry, approval_token=token)
    assert result["status"] == "paged"
    assert backend.page_calls == [("a1", "p1")]

    with pytest.raises(TokenReplayed):
        execute_approved(state, registry=registry, approval_token=token)


def test_execute_on_non_pending_state_is_rejected():
    backend = FakeBackend(risk=0.0)
    state = run_agent("p1", 12, backend)
    registry, _ = make_registry(backend)
    with pytest.raises(AgentError, match="not pending_approval"):
        execute_approved(state, registry=registry, approval_token="anything")
