"""MCP server exposing the sepsis tool surface.

Read tools are open. Write tools take ``approval_token`` — minted by the human
approval UI, never by the model and never by this server. The token is bound to
one alert, one patient and one tool, and is consumed on first use.

Run (stdio):

    set SEPSIS_APPROVAL_SECRET=<32+ random bytes>
    python -m src.mcp_server --artifacts artifacts

The server refuses to start without a Gold artifact or a strong secret.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .approval import ApprovalAuthority, ApprovalError
from .runtime import load_backend
from .tools import AuditLog, ToolError, ToolRegistry


def _new_server(name: str):
    """mcp>=2 renamed FastMCP to MCPServer; support both without pinning."""
    try:
        from mcp.server.mcpserver import MCPServer

        return MCPServer(name)
    except ImportError:
        from mcp.server.fastmcp import FastMCP

        return FastMCP(name)


def build_server(backend, authority: ApprovalAuthority, audit: AuditLog | None = None):
    registry = ToolRegistry(backend, authority, audit or AuditLog())
    server = _new_server("sepsis-ews")

    @server.tool()
    def get_patient_state(patient_id: str, as_of_hour: int) -> dict:
        """Vitals, labs, demographics and features up to as_of_hour (labels hidden)."""
        return registry.call(
            "get_patient_state",
            {"patient_id": patient_id, "as_of_hour": as_of_hour},
        )

    @server.tool()
    def get_recent_labs(patient_id: str, as_of_hour: int, hours: int = 24) -> dict:
        """Lab rows in the last N hours, with hours since measurement."""
        return registry.call(
            "get_recent_labs",
            {"patient_id": patient_id, "as_of_hour": as_of_hour, "hours": hours},
        )

    @server.tool()
    def score_sepsis(patient_id: str, as_of_hour: int) -> dict:
        """Calibrated sepsis risk plus top drivers. Fails loud if no model exists."""
        return registry.call(
            "score_sepsis", {"patient_id": patient_id, "as_of_hour": as_of_hour}
        )

    @server.tool()
    def retrieve_guideline(query: str, k: int = 5) -> dict:
        """Top-k guideline chunks with ids for citation. Fails loud if unbuilt."""
        return registry.call("retrieve_guideline", {"query": query, "k": k})

    @server.tool()
    def get_alert_history(patient_id: str) -> dict:
        """Prior alerts and approval decisions for this patient."""
        return registry.call("get_alert_history", {"patient_id": patient_id})

    @server.tool()
    def draft_sbar(patient_id: str, as_of_hour: int) -> dict:
        """Draft a cited SBAR. Proposes only; pages no one."""
        return registry.call(
            "draft_sbar", {"patient_id": patient_id, "as_of_hour": as_of_hour}
        )

    @server.tool()
    def propose_escalation(
        patient_id: str,
        as_of_hour: int,
        reason: str,
        citations: list[str],
    ) -> dict:
        """Record a proposed escalation for human review. No side effect."""
        return registry.call(
            "propose_escalation",
            {
                "patient_id": patient_id,
                "as_of_hour": as_of_hour,
                "reason": reason,
                "citations": citations,
            },
        )

    @server.tool()
    def page_rapid_response(
        alert_id: str,
        patient_id: str,
        note: str,
        approval_token: str | None = None,
    ) -> dict:
        """Irreversible page. Requires a human approval token bound to this alert.

        The token is supplied by the approval UI after a clinician confirms.
        Never fabricate, reuse, or derive it from the note.
        """
        try:
            return registry.call(
                "page_rapid_response",
                {"alert_id": alert_id, "patient_id": patient_id, "note": note},
                approval_token=approval_token,
            )
        except (ToolError, ApprovalError) as exc:
            return {
                "status": "denied",
                "error": type(exc).__name__,
                "detail": str(exc),
            }

    @server.tool()
    def write_ehr_note(
        alert_id: str,
        patient_id: str,
        note: str,
        approval_token: str | None = None,
    ) -> dict:
        """Write an escalation note. Requires a human approval token."""
        try:
            return registry.call(
                "write_ehr_note",
                {"alert_id": alert_id, "patient_id": patient_id, "note": note},
                approval_token=approval_token,
            )
        except (ToolError, ApprovalError) as exc:
            return {
                "status": "denied",
                "error": type(exc).__name__,
                "detail": str(exc),
            }

    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sepsis EWS MCP server")
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path("artifacts"),
        help="directory holding Gold, model, calibrator and RAG index",
    )
    parser.add_argument("--gold", type=Path, help="override Gold parquet path")
    parser.add_argument(
        "--audit", type=Path, default=Path("artifacts/audit.jsonl"), help="audit log"
    )
    args = parser.parse_args(argv)

    secret = os.environ.get("SEPSIS_APPROVAL_SECRET", "").encode("utf-8")
    if len(secret) < 32:
        raise SystemExit(
            "SEPSIS_APPROVAL_SECRET must be at least 32 bytes. Generate one with:\n"
            "  python -c \"import secrets; print(secrets.token_hex(32))\"\n"
            "and export it before starting the server. Refusing to start without it."
        )

    backend = load_backend(
        gold_path=args.gold,
        artifacts_dir=args.artifacts,
        llm_client=None,  # extractive by default; use src.demo --llm for the LLM path
    )
    authority = ApprovalAuthority(secret, ttl_seconds=900)
    server = build_server(backend, authority, AuditLog(args.audit))
    server.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
