"""Generate the local Dex + oauth2-proxy demo configuration.

    python scripts/setup_identity_demo.py

Creates `deploy/identity/dex.yaml` and `deploy/identity/oauth2-proxy.cfg` with
random client/cookie secrets. Login: clinician@example.com / sepsis-demo.
These files are gitignored; only the templates and compose file are committed.
"""

from __future__ import annotations

import base64
import secrets
import uuid
from pathlib import Path

import bcrypt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deploy" / "identity"
DEX = OUT / "dex.yaml"
PROXY = OUT / "oauth2-proxy.cfg"

EMAIL = "clinician@example.com"
PASSWORD = "sepsis-demo"
CLIENT_ID = "sepsis-ews"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    client_secret = secrets.token_urlsafe(24)
    cookie_secret = base64.b64encode(secrets.token_bytes(32)).decode("ascii")
    password_hash = bcrypt.hashpw(PASSWORD.encode(), bcrypt.gensalt()).decode()

    dex = f"""issuer: http://localhost:5556/dex
storage:
  type: memory
web:
  http: 0.0.0.0:5556
oauth2:
  skipApprovalScreen: true
enablePasswordDB: true
staticClients:
  - id: {CLIENT_ID}
    redirectURIs:
      - http://localhost:8011/oauth2/callback
    name: Sepsis EWS
    secret: {client_secret}
staticPasswords:
  - email: {EMAIL}
    hash: {password_hash}
    username: clinician
    userID: {uuid.uuid4()}
"""
    DEX.write_text(dex, encoding="utf-8")

    proxy = f"""provider = "oidc"
oidc_issuer_url = "http://localhost:5556/dex"
insecure_oidc_skip_issuer_verification = true
skip_oidc_discovery = true
login_url = "http://localhost:5556/dex/auth"
redeem_url = "http://dex:5556/dex/token"
oidc_jwks_url = "http://dex:5556/dex/keys"
client_id = "{CLIENT_ID}"
client_secret = "{client_secret}"
redirect_url = "http://localhost:8011/oauth2/callback"
cookie_secret = "{cookie_secret}"
cookie_secure = false
cookie_samesite = "lax"
email_domains = ["*"]
scope = "openid email profile"
upstreams = ["http://sepsis-ews:8000/"]
pass_user_headers = true
set_xauthrequest = true
reverse_proxy = true
skip_provider_button = true
http_address = "0.0.0.0:4180"
"""
    PROXY.write_text(proxy, encoding="utf-8")
    print(f"wrote {DEX}")
    print(f"wrote {PROXY}")
    print(f"demo login: {EMAIL} / {PASSWORD}")
    print("start: docker compose -f deploy/identity/docker-compose.yml up -d --build")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
