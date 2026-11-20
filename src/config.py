"""Central configuration for the Sepsis Early-Warning & Escalation Copilot.

Every module reads from here. Where a number appears it is either a published
dataset fact (with a citation in SPEC.md) or a **pre-registered gate** — not a
measured result. Measured values live only in ``artifacts/*.json``.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
ARTIFACTS = ROOT / "artifacts"

# --- Dataset -----------------------------------------------------------------
# PhysioNet/CinC Challenge 2019 · CC BY 4.0 · DOI 10.13026/v64v-d857
PHYSIONET_S3 = "s3://physionet-open/challenge-2019/1.0.0/training"
PHYSIONET_HTTP = "https://physionet-open.s3.amazonaws.com/challenge-2019/1.0.0/training"
EXPECTED_SETS = {"A": "training_setA", "B": "training_setB"}
EXPECTED_PATIENTS = 40_336

VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
LABS = [
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets",
]
DEMOGRAPHICS = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"]
LABEL = "SepsisLabel"
RAW_COLUMNS = VITALS + LABS + DEMOGRAPHICS + [LABEL]
EXPECTED_RAW_COLUMNS = 41

# --- Label semantics ---------------------------------------------------------
# The released SepsisLabel is 1 when t >= t_sepsis - 6. The first positive hour
# therefore estimates onset as first_positive_hour + 6 (SPEC.md, "Label").
LABEL_SHIFT_HOURS = 6
EARLY_WINDOW_HOURS = 12   # challenge reward window starts 12 h before onset
ON_TIME_WINDOW_HOURS = 3  # challenge reward window ends 3 h after onset

# --- Alert policy ------------------------------------------------------------
ALERT_BUDGET_PER_PATIENT_DAY = 2.0
REFRACTORY_HOURS = 12
PERSISTENCE_HOURS = 1
THRESHOLD_GRID = (0.01, 1.00, 0.01)  # start, stop (exclusive), step

# --- Splits ------------------------------------------------------------------
SPLIT_RATIOS = (0.70, 0.15, 0.15)
SPLIT_SEED = 42
SPLITS = ("train", "val", "test")

# --- Pre-registered gates (floors; see SPEC.md §Evaluation) ------------------
GATE_DATA = {
    "min_patients": 40_000,
    "min_prevalence": 0.02,
    "max_prevalence": 0.20,
}
GATE_MODEL = {
    "min_auroc": 0.75,              # floor; target 0.80+ from literature
    "min_auprc_lift": 2.0,          # AUPRC >= 2 x prevalence
    "min_sensitivity_at_budget": 0.60,
    "max_alerts_per_patient_day": 2.0,
    "min_median_lead_hours": 4.0,
    "max_brier": 0.08,
    "calibration_slope": (0.80, 1.20),
}
GATE_GROUNDING = {
    "min_faithfulness": 0.85,
    "min_context_precision": 0.80,
    "min_citation_validity": 1.00,
    "min_refusal_correctness": 1.00,
    "min_retrieval_recall_at_5": 0.90,
}

# --- Retrieval / generation (M2) --------------------------------------------
RAG_CHUNK_TOKENS = 512
RAG_CHUNK_OVERLAP = 50
RAG_TOP_K = 5
RAG_EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

# --- Governance --------------------------------------------------------------
FORBIDDEN_FEATURE_NAMES = frozenset(
    {
        "y_label",
        "is_sepsis_patient",
        "t_sepsis_est",
        "hours_to_onset",
        "onset_censored",
        "split",
    }
)
