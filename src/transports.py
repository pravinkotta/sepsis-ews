"""Outbound transports: pager (P1) and EHR (E1).

Both are deliberately narrow:

* Pager payloads carry ids, a hash, and a short summary — not the full note.
* EHR write-back creates a FHIR `Communication` (an AI-labelled draft), then
  reads it back before reporting success.
* Delivery failures are returned, never reported as success.

`local` remains the default so tests and keyless demos do not need accounts.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

import httpx


class DeliveryError(RuntimeError):
    """Raised when an outbound transport cannot complete a request."""


@dataclass(frozen=True)
class Delivery:
    transport: str
    delivered: bool
    detail: dict

    def as_dict(self) -> dict:
        return {
            "transport": self.transport,
            "delivered": self.delivered,
            "detail": self.detail,
        }


def _note_hash(note: str) -> str:
    return hashlib.sha256(note.encode("utf-8")).hexdigest()[:16]


class Pager(Protocol):
    name: str

    def send(self, alert: dict) -> Delivery: ...


class Ehr(Protocol):
    name: str

    def write_note(self, alert: dict) -> Delivery: ...


class LocalPager:
    name = "local"

    def send(self, alert: dict) -> Delivery:
        return Delivery(
            "local",
            True,
            {"mode": "local-jsonl", "alert_id": alert.get("alert_id")},
        )


class LocalEhr:
    name = "local"

    def write_note(self, alert: dict) -> Delivery:
        return Delivery(
            "local",
            True,
            {"mode": "local-jsonl", "alert_id": alert.get("alert_id")},
        )


class _HttpTransport:
    def __init__(
        self,
        *,
        timeout: float = 8.0,
        retries: int = 2,
        client: httpx.Client | None = None,
        sleep=time.sleep,
    ) -> None:
        self.timeout = timeout
        self.retries = max(0, retries)
        self._client = client or httpx.Client(timeout=timeout)
        self._sleep = sleep

    def _post(self, url: str, payload: dict, headers: dict) -> tuple[httpx.Response, int]:
        last_error = "no attempt"
        for attempt in range(self.retries + 1):
            try:
                response = self._client.post(url, json=payload, headers=headers)
                if 200 <= response.status_code < 300:
                    return response, attempt + 1
                if 400 <= response.status_code < 500:
                    return response, attempt + 1
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt < self.retries:
                self._sleep(0.4 * (2 ** attempt))
        raise DeliveryError(last_error)


class WebhookPager(_HttpTransport):
    """Generic JSON webhook (Teams relay, automation, ntfy/webhook.site demo)."""

    name = "webhook"

    def __init__(
        self,
        url: str,
        *,
        headers: dict | None = None,
        timeout: float = 8.0,
        retries: int = 2,
        client: httpx.Client | None = None,
        sleep=time.sleep,
    ) -> None:
        super().__init__(timeout=timeout, retries=retries, client=client, sleep=sleep)
        if not url:
            raise ValueError("webhook URL is required")
        self.url = url
        self.headers = headers or {}

    def send(self, alert: dict) -> Delivery:
        note = str(alert.get("note", ""))
        payload = {
            "source": "sepsis-ews",
            "event": "sepsis_escalation",
            "alert_id": alert.get("alert_id"),
            "patient_id": alert.get("patient_id"),
            "summary": f"Sepsis escalation alert {alert.get('alert_id')}",
            "note_sha256": _note_hash(note),
        }
        headers = {
            "Content-Type": "application/json",
            "Idempotency-Key": str(alert.get("alert_id")),
            **self.headers,
        }
        try:
            response, attempts = self._post(self.url, payload, headers)
        except DeliveryError as exc:
            return Delivery(self.name, False, {"error": str(exc)})
        delivered = 200 <= response.status_code < 300
        return Delivery(
            self.name,
            delivered,
            {
                "status_code": response.status_code,
                "attempts": attempts,
                "idempotency_key": str(alert.get("alert_id")),
                "body": response.text[:300],
            },
        )


class PagerDutyPager(_HttpTransport):
    """PagerDuty Events API v2 with alert_id as dedup key."""

    name = "pagerduty"
    endpoint = "https://events.pagerduty.com/v2/enqueue"

    def __init__(
        self,
        routing_key: str,
        *,
        endpoint: str | None = None,
        timeout: float = 8.0,
        retries: int = 2,
        client: httpx.Client | None = None,
        sleep=time.sleep,
    ) -> None:
        super().__init__(timeout=timeout, retries=retries, client=client, sleep=sleep)
        if not routing_key:
            raise ValueError("PagerDuty routing key is required")
        self.routing_key = routing_key
        if endpoint:
            self.endpoint = endpoint

    def send(self, alert: dict) -> Delivery:
        note = str(alert.get("note", ""))
        alert_id = str(alert.get("alert_id"))
        payload = {
            "routing_key": self.routing_key,
            "event_action": "trigger",
            "dedup_key": alert_id,
            "payload": {
                "summary": f"Sepsis escalation alert {alert_id} (patient {alert.get('patient_id')})",
                "source": "sepsis-ews",
                "severity": "critical",
                "custom_details": {
                    "alert_id": alert_id,
                    "patient_id": alert.get("patient_id"),
                    "note_sha256": _note_hash(note),
                },
            },
        }
        headers = {"Content-Type": "application/json"}
        try:
            response, attempts = self._post(self.endpoint, payload, headers)
        except DeliveryError as exc:
            return Delivery(self.name, False, {"error": str(exc)})
        delivered = 200 <= response.status_code < 300
        return Delivery(
            self.name,
            delivered,
            {
                "status_code": response.status_code,
                "attempts": attempts,
                "dedup_key": alert_id,
                "body": response.text[:300],
            },
        )


class FhirEhr:
    """FHIR R4 write-back as a `Communication`, with read-back verification."""

    name = "fhir"

    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        patient_id: str | None = None,
        patient_prefix: str = "",
        timeout: float = 10.0,
        client: httpx.Client | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("FHIR base URL is required")
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.patient_id = patient_id
        self.patient_prefix = patient_prefix
        self._client = client or httpx.Client(timeout=timeout)

    def _headers(self) -> dict:
        headers = {
            "Content-Type": "application/fhir+json",
            "Accept": "application/fhir+json",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    @staticmethod
    def _fhir_id(value: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9\-.]", "-", value)
        return (cleaned or "sepsis-demo")[:64]

    def _ensure_patient(self, patient_id: str) -> tuple[bool, dict]:
        patient = {
            "resourceType": "Patient",
            "id": patient_id,
            "active": True,
            "identifier": [
                {
                    "system": "https://sepsis-ews.local/demo",
                    "value": patient_id,
                }
            ],
        }
        response = self._client.put(
            f"{self.base_url}/Patient/{patient_id}",
            json=patient,
            headers=self._headers(),
        )
        return response.status_code in (200, 201, 204, 409), {
            "status_code": response.status_code,
            "patient_id": patient_id,
        }

    def write_note(self, alert: dict) -> Delivery:
        raw_patient = self.patient_id or f"{self.patient_prefix}{alert.get('patient_id')}"
        patient_id = self._fhir_id(str(raw_patient))
        note = str(alert.get("note", ""))
        try:
            ok, patient_detail = self._ensure_patient(patient_id)
            if not ok:
                return Delivery(self.name, False, {"stage": "patient", **patient_detail})

            communication = {
                "resourceType": "Communication",
                "status": "completed",
                "subject": {"reference": f"Patient/{patient_id}"},
                "sent": datetime.now(timezone.utc).isoformat(),
                "category": [{"text": "sepsis escalation"}],
                "payload": [{"contentString": note}],
                "meta": {
                    "tag": [
                        {
                            "system": "https://sepsis-ews.local/provenance",
                            "code": "ai-draft",
                            "display": "AI-generated draft; clinician review required",
                        }
                    ]
                },
            }
            response = self._client.post(
                f"{self.base_url}/Communication",
                json=communication,
                headers=self._headers(),
            )
            if response.status_code not in (200, 201):
                return Delivery(
                    self.name,
                    False,
                    {"stage": "communication", "status_code": response.status_code,
                     "body": response.text[:300]},
                )
            resource = response.json()
            communication_id = resource.get("id")
            read_back = self._client.get(
                f"{self.base_url}/Communication/{communication_id}",
                headers=self._headers(),
            )
            delivered = read_back.status_code == 200
            return Delivery(
                self.name,
                delivered,
                {
                    "patient_id": patient_id,
                    "communication_id": communication_id,
                    "create_status": response.status_code,
                    "readback_status": read_back.status_code,
                    "resource_url": f"{self.base_url}/Communication/{communication_id}",
                    "note_sha256": _note_hash(note),
                },
            )
        except (httpx.HTTPError, ValueError) as exc:
            return Delivery(self.name, False, {"error": f"{type(exc).__name__}: {exc}"})
