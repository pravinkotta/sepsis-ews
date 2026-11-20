import pytest

from src.approval import (
    ApprovalAuthority,
    BindingMismatch,
    ExpiredToken,
    InvalidToken,
    TokenReplayed,
)
from src.redteam import FakeBackend, FakeClock
from src.tools import (
    FORBIDDEN_TOOLS,
    TOOLS,
    ApprovalRequired,
    AuditLog,
    ForbiddenTool,
    ToolRegistry,
    UnknownTool,
    assert_tool_surface,
)

SECRET = b"unit-test-secret-with-at-least-32-bytes"


def make_registry(clock=None, ttl=900):
    clock = clock or FakeClock()
    authority = ApprovalAuthority(SECRET, ttl_seconds=ttl, clock=clock)
    return ToolRegistry(FakeBackend(), authority, AuditLog()), authority, clock


def test_read_tool_is_open():
    registry, _, _ = make_registry()
    result = registry.call("score_sepsis", {"patient_id": "p1", "as_of_hour": 12})
    assert result["risk"] == 0.71


def test_write_without_token_is_denied():
    registry, _, _ = make_registry()
    with pytest.raises(ApprovalRequired):
        registry.call(
            "page_rapid_response",
            {"alert_id": "a1", "patient_id": "p1", "note": "deteriorating"},
        )


def test_write_with_token_succeeds_once():
    registry, authority, _ = make_registry()
    token = authority.approve(
        alert_id="a1", patient_id="p1", tool="page_rapid_response", approver="rn.kim"
    )
    args = {"alert_id": "a1", "patient_id": "p1", "note": "deteriorating"}
    assert registry.call("page_rapid_response", args, approval_token=token)["status"] == "paged"
    with pytest.raises(TokenReplayed):
        registry.call("page_rapid_response", args, approval_token=token)
    assert any(r["outcome"] == "ok" for r in registry.audit.records)
    assert any(r["outcome"] == "denied" for r in registry.audit.records)


def test_token_is_bound_to_patient_and_tool():
    _, authority, _ = make_registry()
    token = authority.approve(
        alert_id="a1", patient_id="p1", tool="page_rapid_response", approver="rn.kim"
    )
    with pytest.raises(BindingMismatch):
        authority.consume(
            token, alert_id="a1", patient_id="p2", tool="page_rapid_response"
        )
    with pytest.raises(BindingMismatch):
        authority.consume(token, alert_id="a1", patient_id="p1", tool="write_ehr_note")


def test_expired_token_is_rejected():
    clock = FakeClock()
    _, authority, _ = make_registry(clock=clock, ttl=60)
    token = authority.approve(
        alert_id="a1", patient_id="p1", tool="page_rapid_response", approver="rn.kim"
    )
    clock.advance(61)
    with pytest.raises(ExpiredToken):
        authority.consume(
            token, alert_id="a1", patient_id="p1", tool="page_rapid_response"
        )


def test_forged_token_is_rejected():
    _, authority, _ = make_registry()
    token = authority.approve(
        alert_id="a1", patient_id="p1", tool="page_rapid_response", approver="rn.kim"
    )
    with pytest.raises(InvalidToken):
        authority.consume(
            token[:-2] + "AA",
            alert_id="a1",
            patient_id="p1",
            tool="page_rapid_response",
        )


def test_short_secret_is_rejected():
    with pytest.raises(ValueError, match="at least 32"):
        ApprovalAuthority(b"too-short")


def test_unknown_and_forbidden_tools():
    registry, _, _ = make_registry()
    with pytest.raises(UnknownTool):
        registry.call("read_the_notes", {"patient_id": "p1"})
    with pytest.raises(ForbiddenTool):
        registry.call("start_antibiotics", {"patient_id": "p1"})


def test_forbidden_capabilities_do_not_exist():
    assert not (set(TOOLS) & FORBIDDEN_TOOLS)
    assert "disable_alert" in FORBIDDEN_TOOLS
    assert "start_antibiotics" in FORBIDDEN_TOOLS


def test_tool_surface_checks_pass():
    checks = assert_tool_surface()
    assert checks
    assert all(ok for _, ok, _ in checks), checks
