import os
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from src.config import ARTIFACTS  # noqa: E402

API_KEY = "clinician-test-key"
SECRET = "api-test-secret-with-at-least-32-bytes"

pytestmark = pytest.mark.skipif(
    not (ARTIFACTS / "model.txt").exists()
    or not (ARTIFACTS / "chunks.json").exists(),
    reason="API integration tests need the measured model and RAG index",
)


@pytest.fixture(scope="module")
def client():
    os.environ["SEPSIS_APPROVAL_SECRET"] = SECRET
    os.environ["SEPSIS_APPROVAL_API_KEY"] = API_KEY
    os.environ["SEPSIS_AUTH_MODE"] = "apikey"
    os.environ["SEPSIS_PAGER"] = "local"
    os.environ["SEPSIS_EHR"] = "local"
    from src import api

    api._STATE.clear()
    return TestClient(api.app)


def test_health_score_and_ui(client):
    health = client.get("/health")
    assert health.status_code == 200, health.text
    assert health.json()["status"] == "ok"
    assert health.json()["rag_index"] is True

    score = client.post(
        "/score", json={"patient_id": "p000206", "as_of_hour": 15}
    )
    assert score.status_code == 200, score.text
    body = score.json()
    assert 0.0 <= body["risk"] <= 1.0
    assert body["top_drivers"]

    assert client.get("/review/").status_code == 200
    assert client.get("/", follow_redirects=False).status_code in (307, 308)


def test_agent_approval_boundary(client):
    run = client.post(
        "/agent/run", json={"patient_id": "p000206", "as_of_hour": 15}
    )
    assert run.status_code == 200, run.text
    state = run.json()
    assert state["status"] == "pending_approval"
    alert_id = state["alert_id"]
    assert state["sbar"]["assessment"]
    assert state["validation"]["status"] in ("valid", "needs_review")

    wrong = client.post(
        f"/alerts/{alert_id}/approve",
        json={"patient_id": "p000206", "approver": "rn.test"},
        headers={"X-API-Key": "wrong"},
    )
    assert wrong.status_code == 401

    approved = client.post(
        f"/alerts/{alert_id}/approve",
        json={"patient_id": "p000206", "approver": "rn.test"},
        headers={"X-API-Key": API_KEY},
    )
    assert approved.status_code == 200, approved.text
    token = approved.json()["approval_token"]

    no_token = client.post(
        f"/alerts/{alert_id}/page",
        json={"patient_id": "p000206", "approval_token": "x"},
    )
    assert no_token.status_code in (403, 422)

    paged = client.post(
        f"/alerts/{alert_id}/page",
        json={"patient_id": "p000206", "approval_token": token},
    )
    assert paged.status_code == 200, paged.text
    assert paged.json()["status"] == "paged"

    replay = client.post(
        f"/alerts/{alert_id}/page",
        json={"patient_id": "p000206", "approval_token": token},
    )
    assert replay.status_code == 403

    ehr_approved = client.post(
        f"/alerts/{alert_id}/approve",
        json={"patient_id": "p000206", "approver": "rn.test", "tool": "write_ehr_note"},
        headers={"X-API-Key": API_KEY},
    )
    assert ehr_approved.status_code == 200, ehr_approved.text
    ehr_token = ehr_approved.json()["approval_token"]
    ehr = client.post(
        f"/alerts/{alert_id}/ehr-note",
        json={"patient_id": "p000206", "approval_token": ehr_token},
    )
    assert ehr.status_code == 200, ehr.text
    assert ehr.json()["status"] == "written"

    ehr_replay = client.post(
        f"/alerts/{alert_id}/ehr-note",
        json={"patient_id": "p000206", "approval_token": ehr_token},
    )
    assert ehr_replay.status_code == 403


def test_stream_replays_hourly_risk(client):
    response = client.get(
        "/stream",
        params={"patient_id": "p000206", "start_hour": 10, "end_hour": 12},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.text.count("event: risk") == 3


def test_no_decision_routes_exist(client):
    paths = {route.path for route in client.app.routes}
    forbidden = ("order", "antibiotic", "admit", "discharge", "disable")
    assert not any(term in path for path in paths for term in forbidden)
