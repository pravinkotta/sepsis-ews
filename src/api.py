"""FastAPI service for the Sepsis EWS.

Endpoints:
  GET  /health                      model, threshold, index, budget
  POST /score                       calibrated risk + drivers
  POST /agent/run                   assess → retrieve → draft → propose
  GET  /alerts?patient_id=          proposals/pages for a patient
  POST /alerts/{id}/approve         human approval → single-use token (API key)
  POST /alerts/{id}/page            irreversible page (token required)
  GET  /stream                      SSE replay of hourly risk
  GET  /review/                     one-page review UI

The page endpoint is the only write. It cannot succeed without a token minted
by `/approve`, which itself requires the clinician API key. No route exposes a
decision tool.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    RedirectResponse,
    StreamingResponse,
)
from pydantic import BaseModel, Field

from .agent import AgentError, execute_approved, run_agent
from .approval import ApprovalError
from .auth import auth_mode, proxy_user
from .config import ALERT_BUDGET_PER_PATIENT_DAY, ARTIFACTS
from .env import load_env
from .rag import RagError
from .runtime import load_backend, make_registry
from .store import PatientNotFound
from .tools import ApprovalRequired, BackendNotConfigured, ToolError

APPROVAL_TOOL = "page_rapid_response"
EHR_TOOL = "write_ehr_note"
WRITE_TOOLS = {APPROVAL_TOOL, EHR_TOOL}
MAX_STREAM_HOURS = 168

_STATE: dict[str, Any] = {}


class ScoreRequest(BaseModel):
    patient_id: str = Field(min_length=1)
    as_of_hour: int = Field(ge=1)


class ApproveRequest(BaseModel):
    patient_id: str = Field(min_length=1)
    approver: str | None = Field(default=None, min_length=2)
    tool: str = Field(default=APPROVAL_TOOL)


class PageRequest(BaseModel):
    patient_id: str = Field(min_length=1)
    approval_token: str = Field(min_length=10)


def configure(
    *,
    backend,
    authority,
    registry,
    api_key: str,
    model_info: dict | None = None,
) -> None:
    """Inject a runtime (tests) or replace the default one."""
    _STATE.clear()
    _STATE.update(
        {
            "backend": backend,
            "authority": authority,
            "registry": registry,
            "api_key": api_key,
            "model_info": model_info or {"model": "lightgbm", "threshold": None},
            "alerts": {},
            "started_at": time.time(),
        }
    )


def _state() -> dict:
    if _STATE.get("backend") is not None:
        return _STATE
    load_env()
    secret = os.environ.get("SEPSIS_APPROVAL_SECRET", "").encode("utf-8")
    api_key = os.environ.get("SEPSIS_APPROVAL_API_KEY", "")
    if len(secret) < 32:
        raise RuntimeError(
            "SEPSIS_APPROVAL_SECRET must be at least 32 bytes; refusing to serve"
        )
    if not api_key:
        raise RuntimeError(
            "SEPSIS_APPROVAL_API_KEY must be set for the human approval endpoint"
        )
    llm_client = None
    if os.environ.get("SEPSIS_USE_LLM", "0") == "1":
        from .generate import OpenAIClient

        llm_client = OpenAIClient()
    backend = load_backend(llm_client=llm_client)
    registry, authority = make_registry(backend, secret=secret)
    _STATE.update(
        {
            "backend": backend,
            "authority": authority,
            "registry": registry,
            "api_key": api_key,
            "model_info": {
                "model": "lightgbm",
                "threshold": float(backend.predictor.threshold),
            },
            "alerts": {},
            "started_at": time.time(),
        }
    )
    return _STATE


app = FastAPI(title="Sepsis Early-Warning & Escalation Copilot", version="0.5.0")
logger = logging.getLogger("sepsis.api")


@app.exception_handler(Exception)
async def unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
    """Always return JSON, so the workbench never reports 'non-JSON response'."""
    logger.exception("unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": f"internal error: {type(exc).__name__}: {exc}"},
    )


def _service() -> dict:
    try:
        return _state()
    except Exception as exc:  # noqa: BLE001 - health must say why
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict:
    state = _service()
    backend = state["backend"]
    return {
        "status": "ok",
        "model": state["model_info"].get("model", "lightgbm"),
        "threshold": state["model_info"].get("threshold"),
        "rag_index": getattr(backend, "retriever", None) is not None,
        "alert_budget": ALERT_BUDGET_PER_PATIENT_DAY,
        "approval_required": True,
        "auth_mode": auth_mode(),
        "uptime_seconds": round(time.time() - state["started_at"], 1),
    }


@app.get("/me")
def me(request: Request) -> dict:
    mode = auth_mode()
    if mode == "proxy":
        return {"auth_mode": mode, "user": proxy_user(request)}
    return {"auth_mode": mode, "user": None}


@app.post("/score")
def score(request: ScoreRequest) -> dict:
    state = _service()
    try:
        return state["backend"].score(request.patient_id, request.as_of_hour)
    except PatientNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except BackendNotConfigured as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/agent/run")
def agent_run(request: ScoreRequest) -> dict:
    state = _service()
    try:
        result = run_agent(
            request.patient_id, request.as_of_hour, state["backend"]
        )
    except PatientNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RagError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (BackendNotConfigured, ToolError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if result.get("status") == "pending_approval" and result.get("alert_id"):
        state["alerts"][result["alert_id"]] = result
    return result


@app.get("/alerts")
def alerts(patient_id: str = Query(min_length=1)) -> dict:
    state = _service()
    history = state["backend"].alert_history(patient_id)
    memory = [
        {
            "alert_id": alert_id,
            "patient_id": item.get("patient_id"),
            "status": item.get("status"),
        }
        for alert_id, item in state["alerts"].items()
        if item.get("patient_id") == patient_id
    ]
    return {"patient_id": patient_id, "alerts": history.get("alerts", []), "in_memory": memory}


@app.post("/alerts/{alert_id}/approve")
def approve(
    alert_id: str,
    request: ApproveRequest,
    req: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> dict:
    state = _service()
    tool = request.tool
    if tool not in WRITE_TOOLS:
        raise HTTPException(
            status_code=422,
            detail=f"tool must be one of {sorted(WRITE_TOOLS)}",
        )
    if auth_mode() == "proxy":
        approver = proxy_user(req)
    else:
        expected = state.get("api_key")
        if not expected:
            raise HTTPException(status_code=503, detail="approval API key is not configured")
        if not x_api_key or not secrets.compare_digest(x_api_key, expected):
            raise HTTPException(status_code=401, detail="invalid clinician API key")
        if not request.approver:
            raise HTTPException(status_code=422, detail="approver is required in apikey mode")
        approver = request.approver
    alert = state["alerts"].get(alert_id)
    if alert is None or alert.get("patient_id") != request.patient_id:
        raise HTTPException(status_code=404, detail="alert not found for this patient")
    token = state["authority"].approve(
        alert_id=alert_id,
        patient_id=request.patient_id,
        tool=tool,
        approver=approver,
    )
    return {
        "alert_id": alert_id,
        "approval_token": token,
        "approver": approver,
        "expires_in_seconds": 900,
        "tool": tool,
    }


def _verified_actor(
    state: dict,
    req: Request,
    *,
    alert_id: str,
    patient_id: str,
    token: str,
    tool: str,
) -> str:
    if auth_mode() != "proxy":
        return "clinician"
    actor = proxy_user(req)
    claims = state["authority"].verify(
        token,
        alert_id=alert_id,
        patient_id=patient_id,
        tool=tool,
    )
    if claims.approver.lower() != actor:
        raise HTTPException(
            status_code=403,
            detail="approval token was issued to a different clinician",
        )
    return actor


def _execute_write(
    state: dict, alert: dict, body: PageRequest, req: Request, *, tool: str
) -> dict:
    actor = _verified_actor(
        state,
        req,
        alert_id=alert["alert_id"],
        patient_id=body.patient_id,
        token=body.approval_token,
        tool=tool,
    )
    try:
        return execute_approved(
            alert,
            registry=state["registry"],
            approval_token=body.approval_token,
            tool=tool,
            actor=actor,
        )
    except ApprovalRequired as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ApprovalError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except AgentError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/alerts/{alert_id}/page")
def page(alert_id: str, request: PageRequest, req: Request) -> dict:
    state = _service()
    alert = state["alerts"].get(alert_id)
    if alert is None or alert.get("patient_id") != request.patient_id:
        raise HTTPException(status_code=404, detail="alert not found for this patient")
    return _execute_write(state, alert, request, req, tool=APPROVAL_TOOL)


@app.post("/alerts/{alert_id}/ehr-note")
def ehr_note(alert_id: str, request: PageRequest, req: Request) -> dict:
    state = _service()
    alert = state["alerts"].get(alert_id)
    if alert is None or alert.get("patient_id") != request.patient_id:
        raise HTTPException(status_code=404, detail="alert not found for this patient")
    return _execute_write(state, alert, request, req, tool=EHR_TOOL)


@app.get("/stream")
async def stream(
    patient_id: str = Query(min_length=1),
    start_hour: int = Query(ge=1),
    end_hour: int = Query(ge=1),
    delay_ms: int = Query(default=0, ge=0, le=2000),
):
    state = _service()
    backend = state["backend"]
    if end_hour < start_hour:
        raise HTTPException(status_code=422, detail="end_hour must be >= start_hour")
    if end_hour - start_hour + 1 > MAX_STREAM_HOURS:
        raise HTTPException(status_code=422, detail=f"at most {MAX_STREAM_HOURS} hours")
    try:
        backend.score(patient_id, start_hour)
    except PatientNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except BackendNotConfigured as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    async def events():
        for hour in range(start_hour, end_hour + 1):
            try:
                score = backend.score(patient_id, hour)
            except Exception as exc:  # noqa: BLE001 - stream must terminate loudly
                yield f"event: error\ndata: {json.dumps({'hour': hour, 'error': str(exc)})}\n\n"
                return
            payload = {"hour": hour, **score}
            yield f"event: risk\ndata: {json.dumps(payload, default=str)}\n\n"
            if delay_ms:
                await asyncio.sleep(delay_ms / 1000.0)

    return StreamingResponse(events(), media_type="text/event-stream")


@app.get("/review/")
def review_ui():
    path = Path(__file__).resolve().parent / "static" / "review.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="review UI not installed")
    return FileResponse(path)


@app.get("/")
def root():
    return RedirectResponse(url="/review/")
