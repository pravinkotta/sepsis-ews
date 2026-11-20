"""Backend over the Gold table, model, retriever and local proposal store.

Reads are open; writes are only reachable through `ToolRegistry`, which requires
a human approval token. Note text is hashed, never copied into the log.

If a component is absent the method fails loudly rather than returning a made-up
number or an ungrounded narrative.
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import LABS
from .tools import BackendNotConfigured, ToolError
from .transports import Ehr, LocalEhr, LocalPager, Pager

LABEL_COLUMNS = {
    "y_label",
    "is_sepsis_patient",
    "t_sepsis_est",
    "hours_to_onset",
    "onset_censored",
    "split",
}


class PatientNotFound(ToolError):
    """No row for this patient at or before the requested hour."""


def _clean(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _row_to_dict(row: pd.Series) -> dict:
    return {str(k): _clean(v) for k, v in row.items()}


def _append_jsonl(path: Path | None, record: dict, memory: list[dict]) -> None:
    memory.append(record)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _read_jsonl(path: Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


class DataFrameBackend:
    """Serves the tool surface from a Gold frame plus optional components."""

    def __init__(
        self,
        gold: pd.DataFrame,
        alerts: pd.DataFrame | None = None,
        *,
        predictor=None,
        retriever=None,
        generator=None,
        proposals_path: Path | None = None,
        pages_path: Path | None = None,
        notes_path: Path | None = None,
        pager: Pager | None = None,
        ehr: Ehr | None = None,
    ) -> None:
        required = {"patient_id", "icu_hour"}
        missing = required - set(gold.columns)
        if missing:
            raise ValueError(f"Gold frame missing columns: {sorted(missing)}")
        self.gold = gold.sort_values(["patient_id", "icu_hour"]).reset_index(drop=True)
        # Positional index per patient: a score must not scan 1.55M rows.
        self._positions = {
            str(patient_id): np.asarray(positions, dtype=np.int64)
            for patient_id, positions in self.gold.groupby(
                "patient_id", sort=False
            ).indices.items()
        }
        self.alerts = alerts if alerts is not None else pd.DataFrame(
            columns=["patient_id", "alert_id"]
        )
        self.predictor = predictor
        self.retriever = retriever
        self.generator = generator
        self.proposals_path = proposals_path
        self.pages_path = pages_path
        self.notes_path = notes_path
        self.pager = pager or LocalPager()
        self.ehr = ehr or LocalEhr()
        self._proposals: list[dict] = []
        self._pages: list[dict] = []
        self._notes: list[dict] = []

    def _patient(self, patient_id: str) -> pd.DataFrame:
        positions = self._positions.get(str(patient_id))
        if positions is None:
            raise PatientNotFound(f"unknown patient_id: {patient_id}")
        return self.gold.iloc[positions]

    def patient_state(self, patient_id: str, as_of_hour: int) -> dict:
        block = self._patient(patient_id)
        visible = block[block["icu_hour"] <= int(as_of_hour)]
        if visible.empty:
            raise PatientNotFound(
                f"no observations for {patient_id} at or before hour {as_of_hour}"
            )
        row = visible.iloc[-1]
        features = {
            k: v for k, v in _row_to_dict(row).items() if k not in LABEL_COLUMNS
        }
        return {
            "patient_id": patient_id,
            "as_of_hour": int(row["icu_hour"]),
            "features": features,
            "label_hidden": True,
        }

    def recent_labs(
        self, patient_id: str, as_of_hour: int, hours: int = 24
    ) -> dict:
        if hours <= 0:
            raise ValueError("hours must be positive")
        block = self._patient(patient_id)
        low = int(as_of_hour) - int(hours) + 1
        window = block[
            (block["icu_hour"] >= low) & (block["icu_hour"] <= int(as_of_hour))
        ]
        if window.empty:
            raise PatientNotFound(
                f"no observations for {patient_id} in [{low}, {as_of_hour}]"
            )
        columns = ["icu_hour"] + [
            c for c in window.columns if c in LABS or c.endswith("_last")
        ]
        return {
            "patient_id": patient_id,
            "as_of_hour": int(as_of_hour),
            "window_hours": int(hours),
            "rows": [_row_to_dict(r) for _, r in window[columns].iterrows()],
        }

    def alert_history(self, patient_id: str) -> dict:
        records = [
            record
            for record in _read_jsonl(self.proposals_path) + _read_jsonl(self.pages_path)
            if record.get("patient_id") == patient_id
        ]
        if not self.alerts.empty:
            block = self.alerts[self.alerts["patient_id"] == patient_id]
            records.extend(dict(record) for record in block.to_dict(orient="records"))
        return {"patient_id": patient_id, "alerts": records}

    # ------------------------------------------------------------------ model
    def score(self, patient_id: str, as_of_hour: int) -> dict:
        if self.predictor is None:
            raise BackendNotConfigured(
                "no model artifact: train and register the model first "
                "(SPEC.md phase 3, `python -m src.train`). Refusing to invent a risk."
            )
        state = self.patient_state(patient_id, as_of_hour)
        return self.predictor.predict(state["features"])

    # -------------------------------------------------------------- retrieval
    def retrieve(self, query: str, k: int = 5) -> dict:
        if self.retriever is None:
            raise BackendNotConfigured(
                "no guideline index: build it first (SPEC.md phase 4, "
                "`python -m src.rag --build`)."
            )
        results = self.retriever.search(query, k)
        return {
            "query": query,
            "chunks": [result.as_dict() for result in results],
        }

    def draft_sbar(
        self, patient_id: str, as_of_hour: int, chunks: list | None = None
    ) -> dict:
        if self.generator is None:
            raise BackendNotConfigured(
                "no generation pipeline configured (SPEC.md phase 4)."
            )
        score = self.score(patient_id, as_of_hour)
        if chunks is None:
            drivers = " ".join(
                str(item.get("feature", "")) for item in score.get("top_drivers", [])[:3]
            )
            chunks = self.retrieve(f"sepsis {drivers} escalation", 5)["chunks"]
        patient = {
            "patient_id": patient_id,
            "as_of_hour": as_of_hour,
            "risk": score["risk"],
            "threshold": score["threshold"],
            "drivers": score.get("top_drivers", []),
            "facts": score.get("facts", {}),
        }
        return self.generator.draft(patient, chunks)

    # ----------------------------------------------------------------- writes
    def propose_escalation(
        self, patient_id: str, as_of_hour: int, reason: str, citations: list
    ) -> dict:
        record = {
            "alert_id": f"alert-{uuid.uuid4().hex[:12]}",
            "patient_id": patient_id,
            "as_of_hour": int(as_of_hour),
            "reason": reason,
            "citations": list(citations),
            "status": "pending_approval",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        _append_jsonl(self.proposals_path, record, self._proposals)
        return record

    def page_rapid_response(self, alert_id: str, patient_id: str, note: str) -> dict:
        delivery = self.pager.send(
            {"alert_id": alert_id, "patient_id": patient_id, "note": note}
        )
        record = {
            "alert_id": alert_id,
            "patient_id": patient_id,
            "status": "paged" if delivery.delivered else "page_failed",
            "transport": delivery.transport,
            "delivered": delivery.delivered,
            "delivery": delivery.detail,
            "note_sha256": hashlib.sha256(note.encode("utf-8")).hexdigest()[:16],
            "paged_at": datetime.now(timezone.utc).isoformat(),
        }
        _append_jsonl(self.pages_path, record, self._pages)
        return record

    def write_ehr_note(self, alert_id: str, patient_id: str, note: str) -> dict:
        delivery = self.ehr.write_note(
            {"alert_id": alert_id, "patient_id": patient_id, "note": note}
        )
        record = {
            "alert_id": alert_id,
            "patient_id": patient_id,
            "status": "written" if delivery.delivered else "write_failed",
            "transport": delivery.transport,
            "delivered": delivery.delivered,
            "delivery": delivery.detail,
            "note_sha256": hashlib.sha256(note.encode("utf-8")).hexdigest()[:16],
            "written_at": datetime.now(timezone.utc).isoformat(),
        }
        _append_jsonl(self.notes_path, record, self._notes)
        return record


def load_gold(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Gold table not found: {path}. Build it with phase 3 "
            f"(`python -m src.build_gold`) or point --gold at the artifact."
        )
    return pd.read_parquet(path)
