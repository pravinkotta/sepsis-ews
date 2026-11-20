"""Wire the measured model, retriever and SBAR service into one backend."""

from __future__ import annotations

import os
from pathlib import Path

from .approval import ApprovalAuthority
from .config import ARTIFACTS
from .generate import SBARService
from .predict import ModelPredictor
from .rag import Retriever
from .store import DataFrameBackend, load_gold
from .tools import AuditLog, ToolRegistry
from .transports import (
    FhirEhr,
    LocalEhr,
    LocalPager,
    PagerDutyPager,
    WebhookPager,
)


def build_pager():
    mode = os.environ.get("SEPSIS_PAGER", "local").strip().lower()
    if mode == "local":
        return LocalPager()
    if mode == "webhook":
        url = os.environ.get("SEPSIS_PAGER_WEBHOOK_URL", "")
        if not url:
            raise RuntimeError("SEPSIS_PAGER=webhook requires SEPSIS_PAGER_WEBHOOK_URL")
        return WebhookPager(url)
    if mode == "pagerduty":
        key = os.environ.get("SEPSIS_PAGERDUTY_ROUTING_KEY", "")
        if not key:
            raise RuntimeError(
                "SEPSIS_PAGER=pagerduty requires SEPSIS_PAGERDUTY_ROUTING_KEY"
            )
        return PagerDutyPager(key)
    raise RuntimeError(f"unknown SEPSIS_PAGER={mode!r}; use local|webhook|pagerduty")


def build_ehr():
    mode = os.environ.get("SEPSIS_EHR", "local").strip().lower()
    if mode == "local":
        return LocalEhr()
    if mode == "fhir":
        base = os.environ.get("SEPSIS_FHIR_BASE", "")
        if not base:
            raise RuntimeError("SEPSIS_EHR=fhir requires SEPSIS_FHIR_BASE")
        return FhirEhr(
            base,
            token=os.environ.get("SEPSIS_FHIR_TOKEN") or None,
            patient_id=os.environ.get("SEPSIS_FHIR_PATIENT") or None,
            patient_prefix=os.environ.get("SEPSIS_FHIR_PATIENT_PREFIX", ""),
        )
    raise RuntimeError(f"unknown SEPSIS_EHR={mode!r}; use local|fhir")


def load_backend(
    *,
    gold_path: Path | None = None,
    artifacts_dir: Path = ARTIFACTS,
    llm_client=None,
    use_dense: bool = True,
    use_serving: bool = True,
) -> DataFrameBackend:
    artifacts_dir = Path(artifacts_dir)
    if gold_path is None:
        serving = artifacts_dir / "serving.parquet"
        if use_serving and serving.exists():
            gold_path = serving
        else:
            gold_path = artifacts_dir / "gold_hourly.parquet"
    gold = load_gold(gold_path)
    predictor = ModelPredictor(artifacts_dir)
    retriever = Retriever(artifacts_dir, use_dense=use_dense)
    generator = SBARService(client=llm_client)
    return DataFrameBackend(
        gold,
        predictor=predictor,
        retriever=retriever,
        generator=generator,
        proposals_path=artifacts_dir / "proposals.jsonl",
        pages_path=artifacts_dir / "pages.jsonl",
        notes_path=artifacts_dir / "notes.jsonl",
        pager=build_pager(),
        ehr=build_ehr(),
    )


def make_registry(
    backend: DataFrameBackend,
    *,
    secret: bytes,
    artifacts_dir: Path = ARTIFACTS,
) -> tuple[ToolRegistry, ApprovalAuthority]:
    authority = ApprovalAuthority(secret, ttl_seconds=900)
    audit = AuditLog(Path(artifacts_dir) / "audit.jsonl")
    return ToolRegistry(backend, authority, audit), authority
