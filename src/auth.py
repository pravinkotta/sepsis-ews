"""Authentication modes.

`apikey` (default, local demo): a shared clinician key gates `/approve`.
`proxy` (T1): an OIDC reverse proxy authenticates the clinician and injects
`X-Auth-Request-Email`; the app trusts that header **only** when the service is
not directly exposed. The verified email becomes the approval identity and is
bound into the single-use token.
"""

from __future__ import annotations

import os

from fastapi import HTTPException, Request


def auth_mode() -> str:
    return os.environ.get("SEPSIS_AUTH_MODE", "apikey").strip().lower()


def proxy_user(request: Request) -> str:
    email = (
        request.headers.get("X-Auth-Request-Email")
        or request.headers.get("X-Auth-Request-User")
        or request.headers.get("X-Forwarded-Email")
    )
    if not email or "@" not in email:
        raise HTTPException(
            status_code=401,
            detail="no verified clinician identity from the auth proxy",
        )
    return email.strip().lower()
