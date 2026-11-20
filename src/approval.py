"""Single-use, patient-scoped approval tokens for write tools.

The agent can *propose* an escalation. Only a human can authorise the two
irreversible actions (paging rapid response, writing an EHR note). That
authorisation is represented by an opaque token:

    v1.<base64url(payload)>.<base64url(HMAC-SHA256(secret, payload))>

Binding: the token names one alert, one patient and one tool, and expires.
Consumption: each nonce is accepted once. Replay, expiry, forgery, or a token
presented against a different patient/tool all raise a named exception.

The secret belongs to the approval service and the tool executor — never to the
LLM, the prompt, or the agent's context. A token is not confidential in transit
(it carries ids, not PHI), but it is unforgeable and non-reusable.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Callable


class ApprovalError(Exception):
    """Base class for approval failures."""


class InvalidToken(ApprovalError):
    """Malformed, tampered, or wrongly signed token."""


class ExpiredToken(ApprovalError):
    """Token past its expiry."""


class BindingMismatch(ApprovalError):
    """Token is valid but was issued for another alert/patient/tool."""


class TokenReplayed(ApprovalError):
    """Token nonce was already consumed."""


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64d(text: str) -> bytes:
    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + padding)


@dataclass(frozen=True)
class Approval:
    alert_id: str
    patient_id: str
    tool: str
    approver: str
    issued_at: int
    expires_at: int
    nonce: str


class ApprovalAuthority:
    """Mint and consume approval tokens.

    ``ttl_seconds`` is deliberately short: a page justified at 14:05 should not
    still be executable at 15:00.
    """

    def __init__(
        self,
        secret: bytes,
        *,
        ttl_seconds: int = 900,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if len(secret) < 32:
            raise ValueError("approval secret must be at least 32 bytes")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self._secret = secret
        self._ttl = ttl_seconds
        self._clock = clock
        self._used: set[str] = set()
        self._lock = threading.Lock()

    def approve(
        self,
        *,
        alert_id: str,
        patient_id: str,
        tool: str,
        approver: str,
    ) -> str:
        if not all(isinstance(v, str) and v.strip() for v in (alert_id, patient_id, tool, approver)):
            raise ValueError("alert_id, patient_id, tool and approver are required")
        now = int(self._clock())
        payload = {
            "a": alert_id,
            "p": patient_id,
            "t": tool,
            "by": approver,
            "iat": now,
            "exp": now + self._ttl,
            "n": secrets.token_urlsafe(16),
        }
        body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        sig = hmac.new(self._secret, body, hashlib.sha256).digest()
        return f"v1.{_b64e(body)}.{_b64e(sig)}"

    def _verify(
        self,
        token: str,
        *,
        alert_id: str,
        patient_id: str,
        tool: str,
        now: int | None = None,
    ) -> Approval:
        if not isinstance(token, str):
            raise InvalidToken("token must be a string")
        parts = token.split(".")
        if len(parts) != 3 or parts[0] != "v1":
            raise InvalidToken("token is not a v1 approval token")
        try:
            body = _b64d(parts[1])
            signature = _b64d(parts[2])
        except Exception as exc:  # noqa: BLE001 - decode errors are opaque
            raise InvalidToken("token encoding is invalid") from exc

        expected = hmac.new(self._secret, body, hashlib.sha256).digest()
        if not hmac.compare_digest(signature, expected):
            raise InvalidToken("token signature does not verify")

        try:
            payload = json.loads(body)
            approval = Approval(
                alert_id=str(payload["a"]),
                patient_id=str(payload["p"]),
                tool=str(payload["t"]),
                approver=str(payload["by"]),
                issued_at=int(payload["iat"]),
                expires_at=int(payload["exp"]),
                nonce=str(payload["n"]),
            )
        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise InvalidToken("token payload is malformed") from exc

        current = int(self._clock()) if now is None else now
        if current > approval.expires_at:
            raise ExpiredToken(
                f"approval expired at {approval.expires_at} (now {current})"
            )
        if (
            approval.alert_id != alert_id
            or approval.patient_id != patient_id
            or approval.tool != tool
        ):
            raise BindingMismatch(
                "approval is bound to "
                f"alert={approval.alert_id} patient={approval.patient_id} "
                f"tool={approval.tool}, not "
                f"alert={alert_id} patient={patient_id} tool={tool}"
            )
        return approval

    def verify(
        self, token: str, *, alert_id: str, patient_id: str, tool: str
    ) -> Approval:
        """Check a token without consuming it (read-only inspection)."""
        return self._verify(
            token, alert_id=alert_id, patient_id=patient_id, tool=tool
        )

    def consume(
        self, token: str, *, alert_id: str, patient_id: str, tool: str
    ) -> Approval:
        """Verify and burn a token. The second use always fails."""
        approval = self._verify(
            token, alert_id=alert_id, patient_id=patient_id, tool=tool
        )
        with self._lock:
            if approval.nonce in self._used:
                raise TokenReplayed("approval token has already been used")
            self._used.add(approval.nonce)
        return approval
