"""Cited SBAR drafting.

Two modes, explicitly labelled:

* ``llm`` — structured JSON from an OpenAI model, one schema-repair round.
* ``extractive`` — deterministic assembly from the patient facts and retrieved
  chunks. This exists so the demo and tests work without a key. It is **not**
  presented as an LLM narrative.

Both modes pass through `src.validate.validate_sbar`; a rejected draft is
returned as rejected, never silently accepted.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Protocol

from .config import LLM_MODEL
from .env import load_env
from .validate import ValidationResult, validate_sbar

REQUIRED_SECTIONS = ("situation", "background", "assessment", "recommendation")


class GenerationError(RuntimeError):
    """Raised when a draft cannot be produced or parsed."""


@dataclass
class Draft:
    sbar: dict
    generator: str
    validation: ValidationResult
    citations: list[dict]
    refused: bool = False

    def as_dict(self) -> dict:
        return {
            "sbar": self.sbar,
            "generator": self.generator,
            "validation": self.validation.as_dict(),
            "citations": self.citations,
            "refused": self.refused,
        }


class LLMClient(Protocol):
    def complete(self, prompt: str) -> str: ...


class OpenAIClient:
    def __init__(self, model: str = LLM_MODEL, api_key: str | None = None) -> None:
        load_env()
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise GenerationError(
                "OPENAI_API_KEY is not set. Export it, or use the extractive "
                "generator explicitly (`--extractive`). No silent template fallback."
            )
        from openai import OpenAI

        self._client = OpenAI(api_key=key)
        self.model = model

    def complete(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You draft SBAR escalation notes for clinicians. Use ONLY "
                        "the supplied chunks. Cite every assessment and "
                        "recommendation sentence with [S#]. Never write a treatment "
                        "order, never confirm a diagnosis, and say 'insufficient "
                        "guidance' when the chunks do not support a claim. Return "
                        "JSON with keys situation, background, assessment, recommendation."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content or ""


def _citation(chunk: dict) -> str:
    return str(chunk.get("citation_id", ""))


def _chunk_text(chunk: dict) -> str:
    return str(chunk.get("text", ""))


def build_prompt(patient: dict, chunks: list[dict]) -> str:
    facts = patient.get("facts", {})
    drivers = patient.get("drivers", [])
    lines = [
        f"Patient: {patient.get('patient_id')}",
        f"ICU hour: {patient.get('as_of_hour')}",
        f"Calibrated sepsis risk: {patient.get('risk'):.4f}",
        f"Operating threshold: {patient.get('threshold'):.4f}",
        "Top model drivers: " + json.dumps(drivers, default=str),
        "Facts: " + json.dumps(facts, default=str),
        "",
        "Retrieved chunks (cite only these ids):",
    ]
    for chunk in chunks:
        lines.append(
            f"[{_citation(chunk)}] ({chunk.get('doc')}) {_chunk_text(chunk)}"
        )
    lines += [
        "",
        "Return a JSON object with exactly these keys: situation, background, "
        "assessment, recommendation. Keep each section to 1-3 sentences. Use "
        "'consider' language, never orders. If the chunks cannot support an "
        "assessment or recommendation, write 'insufficient guidance'.",
    ]
    return "\n".join(lines)


def _parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise GenerationError("model response contained no JSON object")
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise GenerationError(f"model response was not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise GenerationError("model response JSON is not an object")
    missing = [key for key in REQUIRED_SECTIONS if key not in payload]
    if missing:
        raise GenerationError(f"model response missing sections: {missing}")
    return {key: str(payload[key]) for key in REQUIRED_SECTIONS}


def _fact_line(facts: dict, limit: int = 8) -> str:
    parts = []
    for key, value in facts.items():
        if value is None or value == "":
            continue
        parts.append(f"{key} {value}")
        if len(parts) >= limit:
            break
    return "; ".join(parts) if parts else "no objective observations available"


def _extractive(patient: dict, chunks: list[dict]) -> dict:
    lactate = next((c for c in chunks if "lactate" in _chunk_text(c).lower()), None)
    sepsis = next((c for c in chunks if "sepsis" in _chunk_text(c).lower()), None)
    policy = next(
        (c for c in chunks if c.get("doc") == "local-escalation-policy"), None
    )
    fallback = chunks[0] if chunks else None

    def cite(chunk) -> str:
        return f" [{_citation(chunk)}]" if chunk else ""

    situation = (
        f"{patient.get('patient_id')} at ICU hour {patient.get('as_of_hour')}: "
        f"calibrated sepsis risk {patient.get('risk', 0):.2%} against a "
        f"{patient.get('threshold', 0):.2%} operating threshold."
    )
    background = _fact_line(patient.get("facts", {})) + cite(lactate or fallback)
    assessment = (
        "Concerning for evolving sepsis given the calibrated risk and the "
        "documented observations" + cite(sepsis or fallback) + "."
    )
    recommendation = (
        "Consider clinician review and, per the local escalation policy, "
        "consider rapid-response escalation; a page requires a human approval "
        "token" + cite(policy or fallback) + "."
    )
    return {
        "situation": situation,
        "background": background,
        "assessment": assessment,
        "recommendation": recommendation,
    }


class SBARService:
    """Thin adapter used by the runtime backend."""

    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client

    def draft(self, patient: dict, chunks: list[dict]) -> Draft:
        return draft_sbar(patient, chunks, client=self.client)


def draft_sbar(
    patient: dict,
    chunks: list[dict],
    *,
    client: LLMClient | None = None,
    max_repairs: int = 1,
) -> Draft:
    retrieved_ids = {_citation(chunk) for chunk in chunks if _citation(chunk)}
    if not retrieved_ids:
        raise GenerationError("no retrieved chunks to cite")

    if client is None:
        sbar = _extractive(patient, chunks)
        validation = validate_sbar(sbar, retrieved_ids=retrieved_ids)
        return Draft(sbar, "extractive", validation, chunks)

    prompt = build_prompt(patient, chunks)
    last_error = ""
    for attempt in range(max_repairs + 1):
        raw = client.complete(prompt)
        try:
            sbar = _parse_json(raw)
        except GenerationError as exc:
            last_error = str(exc)
            prompt = (
                build_prompt(patient, chunks)
                + f"\n\nYour previous response was invalid: {last_error}\n"
                "Return only the JSON object."
            )
            continue

        validation = validate_sbar(sbar, retrieved_ids=retrieved_ids)
        if validation.status != "rejected":
            return Draft(sbar, "llm", validation, chunks)

        last_error = f"validator rejected the draft: {validation.flags}"
        prompt = (
            build_prompt(patient, chunks)
            + f"\n\nYour previous draft was rejected: {validation.flags}\n"
            "Cite only supplied ids and remove treatment orders."
        )

    raise GenerationError(f"could not produce a valid draft: {last_error}")
