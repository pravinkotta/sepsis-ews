# Governance — EU AI Act mapping

This is a portfolio case study, not a certified medical device. If deployed as
clinical decision support it would likely fall within Annex I (medical device
safety component) and be **high-risk**; this repository documents Articles 9–15
as if that were true, deliberately, because the design decisions are the point.

Every row names the file that implements the control — not an intention.

| Article | Obligation | Where it lives | How it is enforced |
|---|---|---|---|
| **Art. 9** Risk management | Continuous, documented risk process | `SPEC.md` §Risks, `src/redteam.py` | 11 attack scenarios + 5 surface checks run on every push; a failure exits non-zero |
| **Art. 10** Data governance | Relevant, representative, error-checked data; bias examination | `scripts/download_data.py`, `src/schema.py`, `src/labels.py` | 40,336 patients / 1,552,210 hours measured; 41/41 columns, monotonic labels, contiguous hours, 0 duplicate ids, 7.27% prevalence; patient-level splits; censoring flagged; subgroup audit in `fairness.json` |
| **Art. 11** Technical documentation | Architecture, design rationale, limitations | `SPEC.md`, `README.md`, private detailed report and working notes | Spec precedes code; README and the private report state what is *not* built and why |
| **Art. 12** Record-keeping | Automatic logs over the system's lifetime | `src/tools.py::AuditLog`, `src/api.py`, `artifacts/audit.jsonl` | Every tool call and HTTP approve/page action is recorded; proposals/pages are local JSONL; note text is hashed, not copied |
| **Art. 13** Transparency | Users know they interact with an AI system and understand its output | `README.md`, `src/generate.py`, `artifacts/demo_run.json` | SBAR carries `[S#]` sources, a validator verdict, and a `generator` label (`extractive` vs `llm`); measured limits are public |
| **Art. 14** Human oversight | A human can intervene, stop, or override | `src/approval.py`, `src/api.py`, `src/static/review.html`, `src/mcp_server.py` | `/alerts/{id}/approve` requires the clinician API key and returns the token; `/alerts/{id}/page` consumes it once; the agent cannot mint it; no decision tools exist |
| **Art. 15** Accuracy, robustness, cybersecurity | Declared accuracy; resilience to errors and adversarial input | `src/gates.py`, `src/validate.py`, `src/threshold.py`, `src/gru.py` | Measured AUROC 0.831 / Brier 0.0165 / slope 0.988; GRU challenger compared and rejected on validation; SHAP + IG completeness; subgroup Wilson CIs; forged-citation rejection; fail-loud paths; HMAC token; ≥32-byte secret; no secret in the image |

## Data governance detail

- **Source and licence.** PhysioNet 2019, CC BY 4.0, DOI 10.13026/v64v-d857.
  No credentialing; citation is carried in the README and the profile.
- **No synthetic training data.** C1. Synthetic PSVs exist only in a temporary
  parser fixture and are never committed or trained on; the data gate rejects
  `sampled: true`.
- **Minimisation.** Only the 41 released columns are used. No identifiers beyond
  the de-identified patient id; the agent sees features, never labels.
- **Representativeness.** Subgroup rates are reported with counts and confidence
  intervals; groups under 50 patients are suppressed rather than described.
- **Retention.** Raw data is gitignored; the profile, metrics, and eval artefacts
  are committed because they are the evidence.

## Human-oversight detail

The boundary is **structural**, not prompt-level:

1. The agent can read and draft freely.
2. It can write a *proposal* — no side effect.
3. Paging and EHR notes require `approval_token`.
4. The token is minted only by the approval service after an authenticated human
   confirms the specific alert.
5. The token is bound to one alert, one patient, and one tool, expires in 900 s,
   and is consumed atomically.
6. Replay, forgery, expiry, and cross-patient/cross-tool use raise named
   exceptions and are audited.

`start_antibiotics`, `order_cultures`, `admit_to_icu`, `discharge_patient`,
`disable_alert`, and the rest of the forbidden list **do not exist**. A future
contributor cannot add one without breaking `assert_tool_surface()` and the
authority gate.

## Transparency note (5 lines, Art. 13)

- This system estimates sepsis risk from ICU vitals and labs; it does not
  diagnose sepsis.
- Every score is accompanied by the model version, the operating threshold, and
  the top drivers that moved *this model's* score.
- Every suggested action cites a retrieved guideline chunk; unsupported claims
  are marked "insufficient guidance".
- A clinician must review and approve any escalation; the system never pages,
  orders, or writes to the EHR on its own.
- Performance is measured on a retrospective public cohort and does not transfer
  automatically to another hospital.

## Known gaps (honest)

- No conformity assessment, clinical evaluation, or post-market surveillance.
- No prospective validation; calibration may drift on a new cohort.
- No hospital EHR integration; replay is not a live feed, and E1 writes only
  de-identified demo data to a public HAPI FHIR sandbox.
- Fairness is audited on three attributes, not intersectionally.
- The RAGAS evaluation is a 20-question authored regression suite; answer
  relevancy is 0.607 and two answerable questions were conservatively refused.
  Judge token usage is not included in the cost estimate.
- The MCP inspector has not yet been run; the tool surface is test-asserted.
- The review UI exists, but there is **no public deployment or TLS**. Local Dex
  SSO is a demo without MFA/RBAC; API-key mode remains a demo gate.
- P1 delivery is a generic webhook; the PagerDuty transport is implemented and
  mock-tested but has not run against a real routing key.
- The service replays the test cohort; a real EHR/feature-store feed is not
  integrated.
