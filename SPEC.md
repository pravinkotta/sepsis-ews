# Case study spec — Sepsis Early-Warning & Escalation Copilot

This document frames the project as **Role → Context → Task → Constraints**.
Every file exists to satisfy a line below, and every design decision traces to a
constraint. Where a number appears it is either a published dataset fact, a
pre-registered gate, or a measured artefact — never an estimate presented as a
result.

---

## R — Role

**Applied-AI engineer on a hospital's clinical informatics team**, producing an
end-to-end, mentor-reviewed case study.

The role owns, and must be able to defend:

- the real-time contract: what is scored, how often, how late data is handled,
  and what latency the care team can rely on;
- the **alert-budget** decision — why the operating point is chosen by tolerated
  alerts per patient-day, not by F1;
- the label semantics, including the six-hour shift and left-censoring;
- calibration, subgroup performance, and drift, with counts beside every ratio;
- the GenAI half: guideline retrieval with citations, a deterministic SBAR
  validator, and an escalation agent with a hard human-authority boundary;
- the measurement design: what "early" means, what counts as a miss, and why
  the test split is touched once;
- packaging: one-command data verification, service, review UI, audit trail, CI,
  and a live deployment.

The deliverable is judged on whether the method survives questioning, not on a
headline AUROC.

---

## C — Context

### Problem

Sepsis is a life-threatening organ dysfunction caused by a dysregulated response
to infection. The PhysioNet 2019 challenge introduction cites ~1.7 million US
cases and ~270,000 deaths per year, ~$24B in annual US hospital costs (13% of US
healthcare expenses), and a **4–8% relative increase in mortality per hour of
delayed treatment** (Kumar et al. 2006; Seymour et al. 2017).

The question modelled here:

> Given the last 24 hours of ICU vitals and labs, what is this patient's
> calibrated sepsis risk **now**, which factors drove it, and what should the
> care team consider doing — grounded in a guideline sentence?

### Why this is a real-time project

Every existing project in the portfolio is batch or request-response. Sepsis is
the one where *time itself* is the product:

- risk is emitted per ICU hour, not per discharge;
- **time-to-alert** is the headline metric, not a latency footnote;
- late-arriving labs, duplicate monitor events, and missing hours are normal;
- alarm fatigue is the binding constraint: a model that alerts on 20% of
  patient-days will be muted, and a muted alarm has zero sensitivity.

### Dataset

**PhysioNet/Computing in Cardiology Challenge 2019 — Early Prediction of Sepsis.**

| | |
|---|---|
| Access | **Open Access, CC BY 4.0** — no credentialing, no DUA |
| Size | 40,336 ICU stays (set A 20,336 + set B 20,000) |
| Shape | one pipe-delimited `.psv` per patient; one row per ICU hour |
| Columns | 41 = 8 vitals + 26 labs + 6 demographics + `SepsisLabel` |
| Onset | Sepsis-3: suspected infection + ≥2-point SOFA rise (see below) |
| Prevalence | **7.27%** measured (2,932 / 40,336); written to `data_profile.json` |
| Source | `s3://physionet-open/challenge-2019/1.0.0/training` |
| Citation | Reyna et al., Crit Care Med 48(2):210–217, 2020 · DOI 10.13026/v64v-d857 |

### Label semantics (load-bearing)

The released label is shifted **ahead by six hours**:

```
SepsisLabel(t) = 1  if t >= t_sepsis - 6
SepsisLabel(t) = 0  otherwise (and always 0 for non-sepsis patients)
```

Therefore:

```
t_sepsis_est = first_positive_hour + 6
```

If the record *starts* positive, the true onset is left-censored — the patient
was already inside the six-hour window at ICU admission. Such patients are
flagged `onset_censored = 1` and **excluded from lead-time claims** (they still
contribute to training and to alert burden).

The Sepsis-3 operational definition used by the challenge:

- `t_suspicion`: earlier of IV antibiotics and blood cultures. If antibiotics
  came first, cultures must follow within 24 h; if cultures came first,
  antibiotics within 72 h; antibiotics must run ≥72 consecutive hours.
- `t_SOFA`: ≥2-point deterioration in SOFA within 24 h.
- `t_sepsis = min(t_suspicion, t_SOFA)` provided
  `t_suspicion − 24 ≤ t_SOFA ≤ t_suspicion + 12`; otherwise the patient is not a
  sepsis patient.

### Data pathologies that shape the design

| Pathology | Consequence if ignored |
|---|---|
| Label is shifted −6 h | onset is wrong by six hours; every lead-time number is wrong |
| First label may already be 1 | left-censored onset; inflated "early" performance |
| `ICULOS` may start at 2, 3, … and must be contiguous | rebasing to 1 would shift onset; a gap breaks rolling windows |
| Labels must be monotonic (`0*1*`) | a 1→0 transition means a corrupt or misparsed file |
| Vitals are dense, labs are sparse | naive forward-fill invents measurements; missingness is itself signal |
| 40k small files, not one CSV | naive loading is slow and memory-hostile; verification must stream |
| Same patient can have multiple ICU stays? | split by patient, never by row |
| Prevalence ~7% | accuracy is meaningless; AUPRC and calibration are required |
| Alert burden is not in the dataset | it must be computed from a chosen policy on validation |

### Available environment

Python 3.10, Windows, CPU only. Installed for this build: `pandas 2.3`,
`numpy 2.2`, `pytest 9.1`, `mcp 2.3`. Phase 2+ adds `lightgbm`, `torch`,
`shap`, `fairlearn`, `evidently`, `mlflow`, `chromadb`, `openai`, `ragas`,
`fastapi`. No GPU assumption.

### Stakeholders and timing

Mentor-reviewed phases; the repository is a portfolio artefact with a live
deployment at the end. Proposal at week 3, milestones M1/M2/M3, demo at week 9,
matching the program calendar.

---

## T — Task

### Phase 1 — spec, contracts, authority (delivered)

1. **Spec** — this document.
2. **Data plane contract** — raw columns, label derivation, Gold schema, leakage
   guard, patient-level stratified splits.
3. **Alert policy** — budget-constrained threshold selection with hysteresis.
4. **Authority plane** — approval tokens, tool registry, forbidden capabilities,
   audit log.
5. **Grounding validator** — deterministic SBAR checks.
6. **Red team** — 11 scripted attacks + 5 surface checks, offline.
7. **Gates and CI** — data, model, grounding, authority.

**Definition of done**

- [x] `python scripts/download_data.py --verify-only` validates a full copy;
      a truncated/renamed file exits non-zero naming the file and the fix.
- [x] `data_profile.json` is generated from files, never hand-written; it
      records prevalence, split counts, and every check.
- [x] `derive_onset` implements `first_positive + 6` and flags censoring;
      non-monotonic labels raise.
- [x] `validate_gold_frame` rejects label leakage, split leakage, non-contiguous
      hours, and label/onset disagreement.
- [x] `select_threshold` maximises sensitivity under the alert budget and raises
      `InfeasibleAlertBudget` rather than returning a violating threshold.
- [x] A write tool without a valid token is denied; replay, expiry, forgery, and
      cross-patient/cross-tool use each raise a named exception.
- [x] Decision capabilities (`start_antibiotics`, `order_cultures`,
      `admit_to_icu`, `disable_alert`, …) do not exist and are test-asserted.
- [x] 11/11 red-team scenarios and 5/5 surface checks pass with no network.
- [x] `python -m src.gates` reports SKIP for absent artefacts and PASS for
      authority; CI runs the authority gate on every push.
- [x] 92 tests pass locally; with `requirements-dev.txt` only, 91 pass and the
      MCP smoke test skips (no network, no key).

### Phase 2 — data plane (M1)

1. **Bronze** — the verified raw copy plus per-file parse status in
   `data_profile.json`; no rewrite, no imputation.
2. **Silver** — causal hourly features, computed in the same pass as Gold:
   - last value, 6 h and 24 h mean, 24 h missing fraction for the 8 vitals;
   - last value and hours-since-measurement for the 26 labs;
   - demographics; `hours_to_onset` as metadata only.
3. **Gold** — `gold_hourly.parquet` + `patient_index.parquet`, validated by
   `src.schema.validate_gold_frame`.
4. **Splits** — stratified 70/15/15 by patient, seeded, disjoint.

**Definition of done**

- [x] `python -m src.build_gold` writes both artefacts from the verified raw copy
      (40,336 patients, 1,552,210 hours, 1,133.7 s, 0 bad files).
- [x] The Gold validator passes and split disjointness is asserted, not assumed
      (28,235 train · 6,051 val · 6,050 test; overlap 0).
- [x] A **future-perturbation test** proves causality: changing any row after
      hour `t` leaves every feature at hour `t` byte-identical.
- [x] Missingness is represented explicitly; no lab is silently forward-filled.
- [x] The data gate passes on a full (non-sampled) profile (13/13 checks).

### Phase 3 — model, calibration, threshold (M1/M2)

1. **Baseline** — LightGBM with `is_unbalance`/`scale_pos_weight`, early stopping
   on validation AUPRC.
2. **Challenger** — GRU (24 h window, 34–40 inputs, 64 units) only kept if it
   beats the baseline on validation; the comparison is reported either way.
3. **Calibration** — isotonic or Platt on validation; Brier score and calibration
   slope reported.
4. **Operating point** — `select_threshold` on validation at the agreed budget;
   the test split is evaluated **once**.
5. **Explainability** — SHAP for the tree model; Integrated Gradients with a
   completeness check for the GRU. Top-3 drivers returned with every score.
6. **Fairness** — Fairlearn `MetricFrame` by age band, sex, and ICU unit
   (`Unit1`/`Unit2`; the released cohort carries no race/ethnicity field);
   counts and confidence intervals beside every rate; groups < 50 patients are
   suppressed.
7. **Drift** — Evidently on the top-10 features; `recommend_retrain` is recorded,
   never executed.
8. **Registry** — MLflow params, metrics, signature, model artefact, git commit.

**Definition of done**

- [x] `python -m src.train` regenerates every model artefact in 61.6 s on CPU.
- [x] `metrics.json` contains AUROC, AUPRC, prevalence, policy (threshold,
      sensitivity, alerts/patient-day, median lead), calibration (Brier, slope),
      and test cohort counts.
- [x] The model gate passes (7/7); a failing gate exits 1.
- [x] SHAP additivity is asserted at runtime (max diff 0.000000); Integrated
      Gradients completeness max diff 0.0166, mean 0.0030.
- [x] Per-subgroup counts and 95% Wilson confidence intervals are committed; no
      claim is made for n < 50.
- [x] `drift_summary.json` records the verdict and `recommend_retrain`; Evidently
      HTML report generated; no automatic retraining path exists.
- [x] MLflow runs for both models, registered versions, and a `champion` alias
      (`artifacts/mlflow_runs.json`).

### Measured M1 baseline (test split, evaluated once)

90 features · 94 boosting rounds · 28,235 train / 6,051 val / 6,050 test
patients. Policy selected on validation with the 2.0 alerts/patient-day budget;
threshold grid is empirical quantiles, not a fixed 0.01 step.

| Metric | Value | Note |
|---|---:|---|
| ROC-AUC, pre-onset hours | **0.8312** | 232,887 pre-onset test hours |
| ROC-AUC, all hours | 0.8424 | secondary, includes post-onset hours |
| AUPRC | **0.0609** | pre-onset prevalence 1.131% → **5.4× lift** |
| Patient-level ROC-AUC | 0.8623 | max risk per stay |
| Brier / calibration slope | 0.0165 / 0.9877 | isotonic on validation |
| Operating threshold | 0.0062 | selected on validation |
| Sensitivity | **0.9512** (351/369) | 12 missed, 6 on-time |
| Alerts / patient-day | **1.463** (14,291 alerts) | budget 2.0 |
| Median lead | **7.0 h** | among early detections |
| SHAP additivity | 0.000000 | exact match to boosted margin |
| Fairness, recall Δ | age 0.038 · sex 0.036 · unit 0.066 | all groups ≥50 |
| Fairness, FPR Δ | age 0.034 · sex 0.011 · **unit 0.158** | largest finding |
| Drift, max PSI (top-10) | 0.0024 | stable; `recommend_retrain=false` |

**Honest read.** Retrospective benchmark, no external validation and no
prospective use. AUPRC is low in absolute terms because positives are rare; the
defensible headline is the operating point (95% sensitivity at 1.46
alerts/patient-day) and the 7-hour median lead. The ICU-unit FPR gap is recorded
as a finding, not explained away.

### Phase 4 — grounding, agent, MCP (M2)

1. **Corpus** — Surviving Sepsis Campaign 2021, Hour-1 bundle, and one local
   escalation policy (public or de-identified). Chunk 512 tokens / 50 overlap.
2. **Index** — Chroma; hybrid BM25 + dense retrieval; top-5 with stable ids
   `[S1]…[S5]`; a 20-query golden set with expected chunks.
3. **Draft** — structured SBAR; the model may only cite retrieved ids, must
   refuse when no chunk supports a claim, and may not issue treatment orders.
4. **Validate** — `src.validate.validate_sbar`; forged citation = rejected,
   uncited claim = needs review, order language = needs review.
5. **Agent** — LangGraph state machine:
   `assess_risk → retrieve_guideline → draft_sbar → propose_escalation →
   [interrupt: human approval] → page/write`.
6. **MCP** — expose the registry; read tools open, write tools token-gated.

**Definition of done**

- [x] Corpus built and verified: 3 documents / 28,179 words, with licences and
      checksums in `guidelines/MANIFEST.json`.
- [x] Index built: 87 chunks at 512/50, Chroma + BM25 + MiniLM, RRF fusion;
      retrieval recall@5 **1.000**, MRR **0.917** on 20 authored queries.
- [x] Cited SBAR + deterministic validator: forged `[S9]` rejected in code;
      treatment orders flagged.
- [x] The agent cannot page without a token; the token cannot be minted by the
      model, reused, or moved to another patient/tool. Demo proves denial →
      approval → page.
- [x] MCP protocol check over stdio lists exactly **9 tools**, no forbidden
      name appears, and a write without a token is blocked
      (`artifacts/mcp_tools.json`).
- [x] RAGAS gate and `eval_scores.json`: faithfulness **0.893**, context
      precision **0.921**, context recall **0.950**, answer relevancy **0.607**,
      citation validity **1.000**, refusal correctness **1.000**, recall@5
      **1.000**. Grounding gate **PASS 6/6**.

### Phase 3 model comparison (validation decides)

| Model | Val AUROC | Val AUPRC | Test AUROC | Test AUPRC | Champion |
|---|---:|---:|---:|---:|---|
| LightGBM | **0.8344** | **0.0685** | **0.8312** | **0.0609** | **yes** |
| GRU 24 h | 0.7584 | 0.0443 | 0.7341 | 0.0343 | no |

The GRU trained for 10 epochs (CPU), was evaluated on the full validation split,
and lost on both ranking metrics. The decision rule is recorded in
`comparison.json`; the challenger remains runnable and explained rather than
deleted. IG top features were Hct, SBP, potassium, temperature, and respiratory
rate.

### Measured Phase 4 (grounding + agent)

| Item | Measured |
|---|---|
| Corpus | 3 docs · 28,179 words · SSC 2021 (PMC OA) + Hour-1 + local policy |
| Chunks | 87 · 512 tokens / 50 overlap · MiniLM 384-d |
| Retrieval | recall@5 **1.000** · MRR **0.917** · 20 queries |
| Demo | `p000206` hour 15 · risk 3.91% vs 0.62% · SBAR **valid** · `generator: llm` |
| Authority | write denied without token; page only after approval |
| RAGAS | faithfulness **0.893** · context precision **0.921** · context recall **0.950** · answer relevancy **0.607** |
| Citation / refusal | **1.000** / **1.000** (5/5) · false refusals **0.100** (2/20) |
| Generation cost | 63,897 in / 1,847 out tokens · **~$0.0107** (judge usage excluded) |
| MCP | **9 tools** · no forbidden names · write without token blocked |
| Grounding gate | **PASS 6/6** |

The 20-question set is authored as a regression suite, not an unbiased
benchmark. Answer relevancy is the weakest metric, partly because two answerable
questions were conservatively refused. The extractive generator remains
available for keyless runs.

### Phase 5 — service and deployment (M3)

1. **API** — `GET /health`, `POST /score`, `GET /stream` (SSE replay),
   `GET /alerts`, `POST /alerts/{id}/approve` (authenticated human),
   `POST /alerts/{id}/page` (token required).
2. **State** — `patients → hours → alerts → proposals → decisions → audits`.
3. **Fail loud** — missing model, index, or secret → 503 with the build command,
   never an ungrounded answer.
4. **Docker** — serving-only requirements; artefacts mounted or baked.
5. **SLOs** — scoring p95 < 50 ms; end-to-end stream event < 250 ms; uptime
   measured over a stated window.

**Definition of done**

- [x] A replay of one 48 h stay produces a risk curve: **48 events / 0.598 s /
      12.5 ms per event** over real HTTP (`src/stream_check.py`).
- [x] `/health` reports model, threshold, RAG index, budget, and uptime.
- [x] Latency is measured: model p95 **10.0 ms**, in-process API p95 **20.5 ms**.
- [x] Container builds and passes health → score → agent → approve → page at
      4 GB (`sepsis-ews:0.5.0`).
- [ ] Live public deployment — **not claimed**. No host/TLS/identity provider.

### Measured Phase 5 (service)

| Item | Measured |
|---|---|
| Model score | p50 9.4 ms · p95 10.0 ms · p99 10.7 ms |
| In-process API | p50 19.0 ms · p95 20.5 ms (no network hop) |
| 48 h SSE | 48 events · 0.598 s · 12.5 ms/event (uvicorn + urllib) |
| Container | health/score/agent/approve/page verified; 4 GB limit |
| Serving table | 234,431 rows / 6,050 patients / 10.8 MB |
| UI | `/review/` served from the same image |

`stream_check` uses a real uvicorn server because `TestClient` buffers SSE and
reports ~550 ms/event. The difference is measurement, not model latency.

### Phase 6 — final deliverables (W9)

- [x] 5-7 minute demo runbook (kept privately) and a committed demo trace
      (`artifacts/demo_run.json`); a recorded video is still pending.
- [x] 1-page writeup, 10-slide deck, 8-page report (built; kept privately).
- [x] README with architecture, measured tables, fairness/drift, and limits.
- [x] Every resume bullet contains a measured number; the project is committed
      locally (`24b3253`). Remote/GitHub links are pending a push.

---

### Post-capstone integration slice (T1 / P1 / E1)

| Layer | Implemented and measured | Remaining |
|---|---|---|
| **T1 identity** | Dex + oauth2-proxy; `/me` verified `clinician@example.com`; token bound to that email; page 200, replay 403 | T2 RBAC/step-up MFA; T3 enterprise SSO |
| **P1 pager** | Through the app: webhook **200**, delivered=true, idempotency key; PagerDuty Events transport implemented and mock-tested | Provide a routing key and run a real PagerDuty page; T2 escalation/ack |
| **E1 EHR** | Through the app: HAPI FHIR R4 `Communication` **854944**, create 201, read-back 200 | Hospital sandbox, note types; T2/E2 production onboarding |

Evidence: `artifacts/oidc_check.json`, `artifacts/p1_e1_enable.json`,
`artifacts/webhook_check.json`, `artifacts/fhir_check.json`. T1 uses a local Dex
password database without MFA; P1's real delivery is a generic webhook, not
PagerDuty; E1 writes de-identified demo data to a public sandbox.

## Evaluation - four gates

All gates read committed artefacts. A missing artefact is `SKIP` on a fresh
clone and `FAIL` under `--require`. Floors may be **tightened** after the first
honest baseline; they are never lowered to make a red gate green.

### G1 — data contract (`data_profile.json`)

| Check | Rule |
|---|---|
| dataset / version | `physionet-challenge-2019`, `1.0.0` |
| not sampled | `sampled is False` |
| sets present | A and B both non-empty |
| patient count | ≥ 40,000 |
| prevalence | recorded, 0.02–0.20 |
| columns | exactly 41, all named |
| labels | monotonic in every file |
| hours | contiguous from the first recorded hour (≥1) in every file |
| duplicates | 0 patient ids |
| bad files | none |
| split | overlap 0; train + val + test = cohort |

### G2 — model (`metrics.json`)

| Check | Floor |
|---|---|
| AUROC | ≥ 0.75 (target 0.80+ from literature, not a claim) |
| AUPRC lift over prevalence | ≥ 2.0× |
| sensitivity at policy | ≥ 0.60 |
| alerts per patient-day | ≤ 2.0 |
| median lead, early detections | ≥ 4.0 h |
| Brier | ≤ 0.08 |
| calibration slope | 0.80–1.20 |

Every rate is reported with numerator and denominator.

### G3 — grounding (`eval_scores.json`)

| Check | Floor |
|---|---|
| faithfulness | ≥ 0.85 |
| context precision | ≥ 0.80 |
| citation validity | 1.00 |
| refusal correctness | 1.00 |
| retrieval recall@5 | ≥ 0.90 |
| golden set size | ≥ 20 queries |

### G4 — authority (computed offline)

| Check | Rule |
|---|---|
| red team | 11/11 attacks blocked |
| tool surface | 5/5 checks pass |
| forbidden tools | none registered |
| write tools | all require approval |
| write binding | all require `alert_id` and `patient_id` |
| token semantics | replay, expiry, forgery, cross-patient, cross-tool all rejected |

---

## Gold table schema

`artifacts/gold_hourly.parquet` — one row per `(patient_id, icu_hour)`.

| Group | Columns | Type | Rule |
|---|---|---|---|
| keys | `patient_id`, `icu_hour`, `split` | str, int, category | unique pair; hours contiguous from the first recorded ICU hour (≥1); one split per patient |
| labels | `y_label`, `is_sepsis_patient`, `t_sepsis_est`, `hours_to_onset`, `onset_censored` | int8, int8, Int32, Int32, int8 | `y_label` matches `icu_hour >= t_sepsis_est − 6`; non-sepsis has null onset; censored flagged |
| demographics | `Age`, `Gender`, `Unit1`, `Unit2`, `HospAdmTime`, `ICULOS` | float/int | as released |
| vitals (8) | `{v}_last`, `{v}_mean_6h`, `{v}_mean_24h`, `{v}_missing_frac_24h` | float32 | causal only; missingness explicit |
| labs (26) | `{lab}_last`, `{lab}_hours_since` | float32 | no forward-fill beyond the last real measurement |

`src.schema.feature_columns()` strips `patient_id`, `icu_hour`, `split`, and all
five label columns. A training matrix is built only through that function, so a
label cannot enter `X` by accident.

`artifacts/patient_index.parquet` — one row per patient: `n_hours`, `first_hour`,
`last_hour`, `is_sepsis_patient`, `onset_censored`, `t_sepsis_est`, `split`.

---

## Alert policy

```
for tau in grid:
    alerts = episodes(p >= tau, persistence, refractory)
    burden = alerts / patient_days
    keep tau if burden <= budget
choose max(sensitivity, median_lead, tau)
if no tau fits: raise InfeasibleAlertBudget
```

Definitions:

- **episode**: `p ≥ τ` for `persistence` consecutive hours, and at least
  `refractory` hours since the previous alert (defaults 1 h and 12 h);
- **patient-day**: 24 patient-hours in the evaluated split;
- **early detection**: first alert in `[t_sepsis − 12, t_sepsis)`;
- **on time**: first alert in `[t_sepsis, t_sepsis + 3]` (reported, not credited
  as early);
- **late**: first alert after `t_sepsis + 3`;
- **missed**: none of the above;
- **median lead**: median `t_sepsis − first_early_alert` among early detections.

`select_threshold` is called on **validation only**. The test split is scored
once and the result is not used for tuning.

---

## MCP tool surface

| Tool | Class | Side effect | Approval | Purpose |
|---|---|---|---|---|
| `get_patient_state` | read | none | — | latest features up to an hour |
| `get_recent_labs` | read | none | — | labs in a window, with hours-since |
| `score_sepsis` | read | none | — | calibrated risk + drivers |
| `retrieve_guideline` | read | none | — | top-k chunks with ids |
| `get_alert_history` | read | none | — | prior alerts and decisions |
| `draft_sbar` | propose | none | — | cited SBAR draft |
| `propose_escalation` | propose | writes a proposal only | — | queue for human review |
| `page_rapid_response` | write | pages a human | **token** | irreversible |
| `write_ehr_note` | write | writes to EHR | **token** | irreversible |

**Absent by design:** `start_antibiotics`, `order_cultures`, `order_labs`,
`admit_to_icu`, `discharge_patient`, `disable_alert`, `override_alert`,
`approve_alert`, `deny_alert`, `enroll_trial`, `place_order`.

### Approval token

```
v1.<base64url(payload)>.<base64url(HMAC-SHA256(secret, payload))>
payload = {alert, patient, tool, approver, iat, exp, nonce}
```

| Property | Implementation |
|---|---|
| integrity | HMAC-SHA256, constant-time compare |
| expiry | 900 s default; expired token rejected |
| binding | alert + patient + tool must match the call |
| single use | nonce consumed atomically; replay rejected |
| minting | only the approval service, after an authenticated human action |
| agent access | the agent never holds the secret; it only receives a token as an opaque argument |
| audit | every call, allowed or denied, is appended to `audit.jsonl`; note text is hashed, not stored |

| Threat | Mitigation |
|---|---|
| replay | nonce consumed once |
| forgery | HMAC over payload |
| confused deputy (token for another patient) | binding check |
| token theft and reuse | short TTL + single use + binding |
| prompt injection in a note | injection grants nothing; authority is the token |
| alert suppression | suppression tools do not exist |

---

## C — Constraints

| # | Constraint | Enforcement |
|---|---|---|
| C1 | **Real, public data only** — no synthetic training rows | PhysioNet CC BY 4.0; synthetic PSVs exist only in a temp fixture for parser smoke tests and are never committed or trained on |
| C2 | **Measured numbers only** | README/writeup read from `artifacts/*.json`; no metric appears before the artefact exists |
| C3 | **No leakage of any kind** | patient-level splits; `feature_columns()` strips labels; future-perturbation test; preprocessor fit on train only |
| C4 | **Label semantics exactly as released** | `first_positive + 6`; monotonic check; censored onsets excluded from lead claims |
| C5 | **Alert budget, not F1** | `select_threshold`; raises if infeasible; validation-only tuning |
| C6 | **Calibration is mandatory** | Brier and slope in `metrics.json`; G2 gate |
| C7 | **Explanations are verifiable** | SHAP additivity / IG completeness asserted at runtime |
| C8 | **Fairness with honest uncertainty** | counts and CIs; groups < 50 suppressed |
| C9 | **Drift is recorded, never auto-acted** | Evidently summary + `recommend_retrain`; no retrain job |
| C10 | **Closed-world citations** | `validate_sbar` rejects any `[S#]` not retrieved for that patient |
| C11 | **The agent proposes, never decides** | write tools require a bound single-use token; in proxy mode the token is bound to the verified clinician email; decision tools do not exist; 11/11 red-team |
| C12 | **LLM spend is gated and logged** | key-gated LLM; token/cost logged; no silent template fallback. The extractive generator is a separate, explicitly labelled mode (`generator: extractive`), never presented as an LLM narrative |
| C13 | **Fail loudly** | missing data/model/index/secret → non-zero exit or 503 with the fix |
| C14 | **One-command reproducibility** | download/verify, build gold, train, rag, serve; raw data gitignored |
| C15 | **Governance documented** | GOVERNANCE.md maps EU AI Act Art. 9–15 to files |
| C16 | **Real-time contract measured** | model p95 **10.0 ms**; real-HTTP SSE **12.5 ms/event** over 48 events; hysteresis + refractory cap alerts; replay is not a live feed |

---

## Out of scope (agreed)

- Autonomous treatment: antibiotics, fluids, vasopressors, intubation, admission,
  discharge. These are clinician acts and no tool exists for them.
- A live EHR integration (HL7/FHIR) — the demo replays recorded hours.
- Closed-loop paging without a human approval UI.
- Paediatric sepsis (the dataset is adult ICU).
- Prospective clinical validation or regulatory submission.
- Fine-tuning a foundation model — the retrieval + validator path is the
  updatable, auditable choice for guideline text that changes.

## Risks and limitations

- **Label noise.** Sepsis-3 is operational; two clinicians can disagree at the
  boundary. The model predicts the label, not "true sepsis".
- **Left-censoring.** Records that start positive cannot support lead-time
  claims; they are flagged and excluded from that metric.
- **Cohort shift.** Three hospitals, de-identified, retrospective. External
  validation is required before any clinical claim.
- **Alert burden is a negotiation.** 2/patient-day is a design choice, not a
  universal safe value; it must be agreed with the clinical owner.
- **Replay latency is not production latency.** The SLO is measured on the
  service, but a real feed adds network and identity-resolution time.
- **Not a medical device.** Any clinical use triggers regulatory review
  (EU MDR / AI Act Annex I).

## References

1. Reyna MA et al. Early Prediction of Sepsis From Clinical Data: The
   PhysioNet/Computing in Cardiology Challenge 2019. *Crit Care Med*
   48(2):210–217, 2020. DOI 10.1097/CCM.0000000000004145.
2. Singer M et al. The Third International Consensus Definitions for Sepsis and
   Septic Shock (Sepsis-3). *JAMA* 315(8):801–810, 2016.
3. Kumar A et al. Duration of hypotension before initiation of effective
   antimicrobial therapy is the critical determinant of survival in human
   septic shock. *Crit Care Med* 34(6):1589–1596, 2006.
4. Seymour CW et al. Time to Treatment and Mortality during Mandated Emergency
   Care for Sepsis. *N Engl J Med* 376:2235–2244, 2017.
5. Evans L et al. Surviving Sepsis Campaign: International Guidelines for
   Management of Sepsis and Septic Shock 2021. *Crit Care Med* 49(11):e1063.
6. Wong A et al. External Validation of a Widely Implemented Proprietary Sepsis
   Prediction Model in Hospitalized Patients. *JAMA Intern Med* 181(8), 2021.
