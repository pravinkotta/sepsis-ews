<!--
title: Project-local escalation policy
source: authored in this repository
license: Project artefact; not clinical guidance.
retrieved_at: 2026-11-19T12:29:18.825160+00:00
-->
# Project-local escalation policy (authored artefact)

> **Status:** authored for this case study. It is **not** clinical guidance and
> has not been approved by any hospital. It exists so the retrieval corpus has a
> concrete escalation procedure to cite alongside the SSC guideline.

## 1. Trigger

Escalate when the calibrated sepsis risk is at or above the deployed threshold,
or when a bedside clinician is concerned regardless of the score. The model is a
screening aid; clinician concern overrides it.

## 2. SBAR structure

Every escalation draft contains four sections:

- **Situation** — who the patient is, the current hour, the calibrated risk and
  the top model drivers.
- **Background** — objective facts available at that hour: vitals, lactate,
  cultures, antibiotics, fluids, and relevant history.
- **Assessment** — whether the risk and facts are concerning for evolving
  sepsis, with the guideline citation that supports the statement.
- **Recommendation** — what a clinician should consider, citing the local policy
  or the guideline. The draft may recommend *considering* an action; it may not
  issue a treatment order.

## 3. Human approval

The agent may draft and propose. It may not page, order, or write to the EHR.
A rapid-response page requires a human approval token minted after a clinician
confirms the specific alert. The token is bound to one alert, one patient and
one tool, expires in 15 minutes, and is consumed once. The model never sees the
signing secret.

## 4. Escalation tiers

1. **Bedside review** — the assigned nurse or clinician reviews the draft.
2. **Rapid response** — if the clinician confirms, the approved token releases a
   page to the rapid-response team.
3. **ICU transfer** — a separate clinical decision; this system has no tool for
   admission or transfer.

## 5. Documentation and audit

Every tool call, approval, denial, and page is appended to the audit log. Note
text is hashed rather than copied into the log. A rejected draft is kept with
the reason so the prompt and retrieval can be reviewed.

## 6. Hard prohibitions

- No autonomous antibiotic, fluid, vasopressor, or culture order.
- No autonomous ICU admission, discharge, or alert suppression.
- No page without a bound, single-use approval token.
- No claim of confirmed sepsis; the draft says "concerning for" and defers to
  the clinician.
