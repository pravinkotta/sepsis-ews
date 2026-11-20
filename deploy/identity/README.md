# T1 — local OIDC identity demo

A self-contained Dex + oauth2-proxy stack in front of the app. No external
identity provider or account is needed.

```
browser :8011 ──► oauth2-proxy :4180 ──► sepsis-ews :8000
                         │
                         └──► Dex :5556 (static demo user)
```

## Start

```powershell
cd "C:\Users\pravi\Documents\Default Project\sepsis-ews"
python scripts/setup_identity_demo.py
docker compose -f deploy/identity/docker-compose.yml up -d --build
```

Open <http://localhost:8011/review/>.

**Demo login:** `clinician@example.com` / `sepsis-demo`

The app is **not published**; only oauth2-proxy on 8011 and Dex on 5556 are
reachable. The app trusts `X-Auth-Request-Email` only because it is behind the
proxy.

## What T1 proves

- The clinician is authenticated by an OIDC provider before `/approve`.
- The approval token is bound to the verified email (`sub` equivalent), not a
  free-text name.
- `/page` rejects the token if presented by a different verified clinician.
- The audit record uses the verified actor, not the agent.

`GET /me` through the proxy returns the verified user:

```powershell
# after signing in, from a browser session or with the cookie jar
curl.exe -s http://localhost:8011/me
```

## What this is not

- Dex uses an in-memory store and a static demo password; no MFA.
- HTTP on localhost, not TLS.
- No RBAC (viewer/approver/admin), step-up MFA, sessions, or SCIM.
Those are T2/T3 in `docs/PRODUCTION-SCOPE.md`.

## Stop

```powershell
docker compose -f deploy/identity/docker-compose.yml down
```

Generated `dex.yaml` and `oauth2-proxy.cfg` are gitignored; regenerate them with
the setup script.
