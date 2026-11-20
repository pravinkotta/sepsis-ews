import json

import httpx

from src.transports import FhirEhr, PagerDutyPager, WebhookPager


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_webhook_retries_and_sends_idempotency_key():
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(500, text="temporary")
        return httpx.Response(200, json={"ok": True})

    pager = WebhookPager(
        "https://hooks.example.test/sepsis",
        retries=1,
        client=_client(handler),
        sleep=lambda _: None,
    )
    delivery = pager.send({"alert_id": "a1", "patient_id": "p1", "note": "secret note"})
    assert delivery.delivered
    assert delivery.detail["attempts"] == 2
    assert calls[0].headers["Idempotency-Key"] == "a1"
    body = json.loads(calls[-1].content)
    assert body["alert_id"] == "a1"
    assert "note" not in body  # note is not sent to a third-party pager
    assert body["note_sha256"]


def test_webhook_4xx_is_not_retried():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(400, text="bad payload")

    pager = WebhookPager(
        "https://hooks.example.test/sepsis",
        retries=2,
        client=_client(handler),
        sleep=lambda _: None,
    )
    delivery = pager.send({"alert_id": "a1", "patient_id": "p1", "note": "n"})
    assert not delivery.delivered
    assert len(calls) == 1
    assert delivery.detail["status_code"] == 400


def test_pagerduty_events_payload_and_dedup_key():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(202, json={"status": "success", "dedup_key": "a1"})

    pager = PagerDutyPager(
        "routing-key", client=_client(handler), sleep=lambda _: None
    )
    delivery = pager.send({"alert_id": "a1", "patient_id": "p1", "note": "n"})
    assert delivery.delivered
    assert seen["event_action"] == "trigger"
    assert seen["dedup_key"] == "a1"
    assert seen["payload"]["severity"] == "critical"
    assert seen["payload"]["custom_details"]["patient_id"] == "p1"


def test_fhir_communication_write_and_read_back():
    calls = []

    def handler(request):
        calls.append((request.method, request.url.path))
        if request.method == "PUT":
            return httpx.Response(
                201, json={"resourceType": "Patient", "id": "p1"}
            )
        if request.method == "POST":
            body = json.loads(request.content)
            assert body["resourceType"] == "Communication"
            assert body["subject"]["reference"] == "Patient/p1"
            assert body["payload"][0]["contentString"] == "SBAR note"
            return httpx.Response(
                201, json={"resourceType": "Communication", "id": "c1"}
            )
        return httpx.Response(200, json={"resourceType": "Communication", "id": "c1"})

    ehr = FhirEhr(
        "https://fhir.example.test/baseR4",
        patient_id="p1",
        client=_client(handler),
    )
    delivery = ehr.write_note(
        {"alert_id": "a1", "patient_id": "p1", "note": "SBAR note"}
    )
    assert delivery.delivered
    assert delivery.detail["communication_id"] == "c1"
    assert calls == [
        ("PUT", "/baseR4/Patient/p1"),
        ("POST", "/baseR4/Communication"),
        ("GET", "/baseR4/Communication/c1"),
    ]
