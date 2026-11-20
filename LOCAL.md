# Run locally (no public exposure)

The service binds to `127.0.0.1` only. Nothing is deployed; there is no public
URL. Use this for the demo and for development.

## 1. Secrets

`.env` (gitignored) holds three values:

```text
OPENAI_API_KEY=...              # optional LLM SBAR / RAGAS
SEPSIS_APPROVAL_SECRET=...      # 32+ random bytes, HMAC signing key
SEPSIS_APPROVAL_API_KEY=...     # clinician key for Approve in the UI
SEPSIS_USE_LLM=1                # 1 = LLM SBAR; 0 = extractive
```

Generate the two approval values without printing them:

```powershell
python -c "import secrets; print('SEPSIS_APPROVAL_SECRET=' + secrets.token_hex(32))" >> .env
python -c "import secrets; print('SEPSIS_APPROVAL_API_KEY=' + secrets.token_urlsafe(24))" >> .env
```

Read `SEPSIS_APPROVAL_API_KEY` from `.env` when the workbench asks for the
clinician key. Never commit `.env`.

## 2. Start

```powershell
cd "C:\Users\pravi\Documents\Default Project\sepsis-ews"
python -m src.export_serving
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/review/>.

The API loads the model, the test-cohort serving table (10.8 MB), and the RAG
index on the first request. If it fails, `artifacts/local-server.err.log` has the
cause.

## 3. Stop

```powershell
Stop-Process -Id (Get-Content artifacts\local-server.pid) -Force
```

(Only if started with the background helper described in the private demo runbook.)

## 4. Verify

```powershell
curl.exe -s http://127.0.0.1:8000/health
curl.exe -s -X POST http://127.0.0.1:8000/score `
  -H "Content-Type: application/json" `
  -d '{\"patient_id\":\"p000206\",\"as_of_hour\":15}'
```

Expected: `status: ok`, RAG index true, risk around **0.0391**.

## 5. T1 identity stack (optional)

A local Dex + oauth2-proxy stack, no external IdP:

```powershell
python scripts/setup_identity_demo.py
docker compose -f deploy/identity/docker-compose.yml up -d --build
```

Open <http://localhost:8011/review/> and sign in as
`clinician@example.com` / `sepsis-demo`. The app is not published; only the
proxy (8011) and Dex (5556) are reachable. See
[deploy/identity/README.md](deploy/identity/README.md).

Stop it with:

```powershell
docker compose -f deploy/identity/docker-compose.yml down
```

## 6. P1 pager / E1 EHR

**Currently enabled** in `.env`: a webhook pager and the HAPI FHIR sandbox
(prefix `sepsis-demo-`). `python scripts/local_transport_check.py` verifies both
through the API. To change them, set these in `.env` and restart:

| Mode | Required values |
|---|---|
| Pager webhook | `SEPSIS_PAGER=webhook` · `SEPSIS_PAGER_WEBHOOK_URL=https://...` |
| PagerDuty | `SEPSIS_PAGER=pagerduty` · `SEPSIS_PAGERDUTY_ROUTING_KEY=...` |
| FHIR | `SEPSIS_EHR=fhir` · `SEPSIS_FHIR_BASE=https://hapi.fhir.org/baseR4` (test) |
| FHIR auth | optional `SEPSIS_FHIR_TOKEN`, `SEPSIS_FHIR_PATIENT` |

Checks:

```powershell
python -m src.webhook_check
python -m src.fhir_check
```

Both write evidence to `artifacts/`. They use de-identified demo data only.

## 7. Later, when you decide to deploy

The container is already built and tested at 4 GB, and T1/P1/E1 are implemented
as a demo slice. Production still needs T2/T3 (RBAC, MFA, enterprise SSO), a
PagerDuty/ServiceNow contract, an E2/E3 EHR onboarding with BAA, and a public
host with TLS. Do not expose this demo publicly without those.
