import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from src import api  # noqa: E402
from src.approval import ApprovalAuthority  # noqa: E402
from src.tools import AuditLog, ToolRegistry  # noqa: E402

SECRET = b"auth-test-secret-with-at-least-32-bytes"


class FakeBackend:
    def page_rapid_response(self, alert_id, patient_id, note):
        return {
            "alert_id": alert_id,
            "patient_id": patient_id,
            "status": "paged",
            "transport": "fake",
        }


def _state(alert_id: str, patient_id: str = "p1") -> dict:
    return {
        "patient_id": patient_id,
        "alert_id": alert_id,
        "as_of_hour": 12,
        "status": "pending_approval",
        "sbar": {
            "situation": "s",
            "background": "b",
            "assessment": "a [S1]",
            "recommendation": "r [S1]",
        },
        "chunks": [{"citation_id": "S1", "doc": "d", "source": "s"}],
        "validation": {"status": "valid", "flags": []},
    }


@pytest.fixture()
def proxy_client(monkeypatch):
    monkeypatch.setenv("SEPSIS_AUTH_MODE", "proxy")
    authority = ApprovalAuthority(SECRET)
    registry = ToolRegistry(FakeBackend(), authority, AuditLog())
    api.configure(
        backend=FakeBackend(),
        authority=authority,
        registry=registry,
        api_key="unused-in-proxy-mode",
        model_info={"model": "fake", "threshold": 0.5},
    )
    return TestClient(api.app)


def test_me_requires_proxy_identity(proxy_client):
    anonymous = proxy_client.get("/me")
    assert anonymous.status_code == 401

    me = proxy_client.get("/me", headers={"X-Auth-Request-Email": "clinician@example.com"})
    assert me.status_code == 200
    assert me.json() == {"auth_mode": "proxy", "user": "clinician@example.com"}


def test_token_is_bound_to_the_verified_clinician(proxy_client):
    proxy_client.app  # keep the fixture state
    api._STATE["alerts"]["a1"] = _state("a1")
    api._STATE["alerts"]["a2"] = _state("a2")

    approved = proxy_client.post(
        "/alerts/a1/approve",
        json={"patient_id": "p1"},
        headers={"X-Auth-Request-Email": "clinician@example.com"},
    )
    assert approved.status_code == 200, approved.text
    token = approved.json()["approval_token"]
    assert approved.json()["approver"] == "clinician@example.com"

    same_user = proxy_client.post(
        "/alerts/a1/page",
        json={"patient_id": "p1", "approval_token": token},
        headers={"X-Auth-Request-Email": "clinician@example.com"},
    )
    assert same_user.status_code == 200, same_user.text
    assert same_user.json()["status"] == "paged"

    # A different verified clinician cannot use the first clinician's token.
    approved2 = proxy_client.post(
        "/alerts/a2/approve",
        json={"patient_id": "p1"},
        headers={"X-Auth-Request-Email": "a@example.com"},
    )
    token2 = approved2.json()["approval_token"]
    other = proxy_client.post(
        "/alerts/a2/page",
        json={"patient_id": "p1", "approval_token": token2},
        headers={"X-Auth-Request-Email": "b@example.com"},
    )
    assert other.status_code == 403
