# Deploy — local service, Docker, and what is still missing

The service is built and measured locally. There is **no live public host** in
this repository; the commands below run it on your machine or a host you own.

## 1. Secrets

Two independent secrets:

| Variable | Purpose |
|---|---|
| `SEPSIS_APPROVAL_SECRET` | HMAC key for approval tokens (≥32 bytes) |
| `SEPSIS_APPROVAL_API_KEY` | Clinician API key for `POST /alerts/{id}/approve` |

```powershell
python -c "import secrets; print(secrets.token_hex(32))"   # approval secret
python -c "import secrets; print(secrets.token_urlsafe(24))" # clinician key
$env:SEPSIS_APPROVAL_SECRET="<value>"
$env:SEPSIS_APPROVAL_API_KEY="<value>"
```

Never commit either. The approval secret belongs to the approval service and the
tool executor; the model never sees it.

## 2. Run locally

```powershell
pip install -r requirements-service.txt
uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/review/>.

| Endpoint | What it does |
|---|---|
| `GET /health` | model, threshold, RAG index, budget, uptime |
| `POST /score` | calibrated risk + exact SHAP drivers |
| `POST /agent/run` | assess → retrieve → cited SBAR → proposal |
| `GET /alerts?patient_id=…` | proposals/pages for a patient |
| `POST /alerts/{id}/approve` | **clinician API key** → single-use token |
| `POST /alerts/{id}/page` | **token required**; local JSONL transport |
| `GET /stream?patient_id=…&start_hour=…&end_hour=…` | SSE hourly replay |
| `GET /review/` | review workbench |

Example:

```powershell
curl -s http://127.0.0.1:8000/score -H "Content-Type: application/json" `
  -d '{\"patient_id\":\"p000206\",\"as_of_hour\":15}'
```

The approval token is returned only by `/approve`; `/page` consumes it once.

## 3. Docker

```powershell
$env:SEPSIS_APPROVAL_SECRET="<value>"
$env:SEPSIS_APPROVAL_API_KEY="<value>"
docker compose up --build
```

The image bakes the MiniLM embedding model (no runtime download) and the
measured artefacts. `artifacts/` is mounted so proposals/pages/audit survive a
container restart. The image is large (the PyTorch wheel plus Chroma); a CPU-only torch pin would
shrink it substantially.

## 4. Measured latency

`python -m src.benchmark --n 200` writes `artifacts/latency.json`:

| Path | p50 | p95 |
|---|---:|---:|
| store + model score | ~9.4 ms | ~10.0 ms |
| in-process API (`/score`) | ~19.0 ms | ~20.5 ms |
| 48 h SSE replay | — | **12.5 ms/event** (real uvicorn) |

The second row includes FastAPI/Pydantic and `TestClient` overhead but **no
network hop**. A remote client adds latency; measure it on the target host.

## 5. Demo integration slice (T1/P1/E1)

- **T1 identity:** [deploy/identity/README.md](deploy/identity/README.md) —
  Dex + oauth2-proxy; login `clinician@example.com / sepsis-demo`; app not
  published; token bound to the verified email. No MFA/RBAC (T2).
- **P1 pager:** `src/transports.py` (webhook / PagerDuty Events API v2);
  `python -m src.webhook_check` for a real receipt; set
  `SEPSIS_PAGER=pagerduty` + `SEPSIS_PAGERDUTY_ROUTING_KEY` for a real page.
- **E1 EHR:** `python -m src.fhir_check` writes a de-identified `Communication`
  to the HAPI R4 sandbox and reads it back; set `SEPSIS_EHR=fhir` +
  `SEPSIS_FHIR_BASE` to use it from the app.

Evidence: `artifacts/oidc_check.json`, `artifacts/webhook_check.json`,
`artifacts/fhir_check.json`. These are demo-tier, not clinical production.

## 6. What is deliberately not here

- **No public deployment.** No host, TLS certificate, DNS, or cloud account is
  claimed. Add a reverse proxy with TLS and an identity provider before any use
  outside a demo.
- **No real pager or EHR.** `page_rapid_response` writes a local JSONL record.
  The approval-token boundary is real; the transport is a demo.
- **No live streaming feed.** `/stream` replays recorded hours. A real feed needs
  HL7/FHIR ingestion, late-arrival handling, and identity resolution.
- **No SSO.** The clinician API key is a demo gate, not an identity system.
- **No backups or retention policy** for proposals/pages/audit beyond the local
  files.

## 6. Release checklist

- [ ] `python -m pytest -q` (92 tests)
- [ ] T1: `python scripts/setup_identity_demo.py` + `docker compose -f deploy/identity/docker-compose.yml up -d --build`
- [ ] T1: `python scripts/oidc_check.py` (login → approve → page → replay 403)
- [ ] P1: `python -m src.webhook_check` or a PagerDuty routing key
- [ ] E1: `python -m src.fhir_check` (FHIR sandbox write + read-back)
- [ ] `python -m src.gates` (data/model/grounding/authority all PASS)
- [ ] `python -m src.eval_llm` when `OPENAI_API_KEY` is available
- [ ] `python -m src.benchmark --n 200`
- [ ] `/health` reports `status: ok` and `rag_index: true`
- [ ] `/agent/run` returns `pending_approval`
- [ ] `/page` fails without a token and succeeds once with one
- [ ] secrets injected at runtime, not baked into the image
