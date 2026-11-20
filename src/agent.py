"""LangGraph escalation agent with an explicit approval interrupt.

The graph can assess, retrieve, draft and *propose*. It cannot page: the graph
stops at ``pending_approval``. A separate `execute_approved` call requires a
single-use token minted by a human approval, and goes through the tool registry
so the authority check is structural.

    assess → retrieve → draft → [rejected?] → propose → END (pending_approval)
"""

from __future__ import annotations

from typing import Protocol, TypedDict

from langgraph.graph import END, StateGraph

from .tools import ToolRegistry


class AgentError(RuntimeError):
    """Raised when the graph is used outside its contract."""


class AgentBackend(Protocol):
    def score(self, patient_id: str, as_of_hour: int) -> dict: ...
    def retrieve(self, query: str, k: int) -> dict: ...
    def draft_sbar(self, patient_id: str, as_of_hour: int, chunks=None) -> dict: ...
    def propose_escalation(
        self, patient_id: str, as_of_hour: int, reason: str, citations: list
    ) -> dict: ...
    def page_rapid_response(self, alert_id: str, patient_id: str, note: str) -> dict: ...


class AgentState(TypedDict, total=False):
    patient_id: str
    as_of_hour: int
    risk: float
    threshold: float
    drivers: list
    facts: dict
    chunks: list
    query: str
    sbar: dict
    validation: dict
    generator: str
    citations: list
    proposal: dict
    alert_id: str
    status: str
    reason: str


def _chunk_dict(chunk) -> dict:
    return chunk.as_dict() if hasattr(chunk, "as_dict") else dict(chunk)


def _draft_payload(draft) -> dict:
    return draft.as_dict() if hasattr(draft, "as_dict") else dict(draft)


def _assess(backend: AgentBackend):
    def node(state: AgentState) -> dict:
        score = backend.score(state["patient_id"], int(state["as_of_hour"]))
        return {
            "risk": float(score["risk"]),
            "threshold": float(score.get("threshold", 0.0)),
            "drivers": list(score.get("top_drivers", [])),
            "facts": dict(score.get("facts", {})),
        }

    return node


def _after_assess(state: AgentState) -> str:
    return "retrieve" if state.get("risk", 0.0) >= state.get("threshold", 1.0) else "no_alert"


def _retrieve(backend: AgentBackend):
    def node(state: AgentState) -> dict:
        drivers = state.get("drivers", [])[:3]
        names = " ".join(
            str(item.get("feature", "")) for item in drivers if isinstance(item, dict)
        )
        query = ("sepsis " + names + " escalation").strip()
        result = backend.retrieve(query, 5)
        raw = result.get("chunks", []) if isinstance(result, dict) else result
        chunks = [_chunk_dict(chunk) for chunk in raw]
        return {"chunks": chunks, "query": query}

    return node


def _after_retrieve(state: AgentState) -> str:
    return "draft" if state.get("chunks") else "no_guidance"


def _draft(backend: AgentBackend):
    def node(state: AgentState) -> dict:
        draft = backend.draft_sbar(
            state["patient_id"], int(state["as_of_hour"]), state.get("chunks", [])
        )
        payload = _draft_payload(draft)
        return {
            "sbar": payload["sbar"],
            "validation": payload["validation"],
            "generator": payload.get("generator", "unknown"),
            "citations": payload.get("citations", state.get("chunks", [])),
        }

    return node


def _after_draft(state: AgentState) -> str:
    validation = state.get("validation", {})
    return "rejected" if validation.get("status") == "rejected" else "propose"


def _propose(backend: AgentBackend):
    def node(state: AgentState) -> dict:
        reason = state.get("sbar", {}).get("assessment", "sepsis risk above threshold")
        citations = [
            str(chunk.get("citation_id", ""))
            for chunk in state.get("chunks", [])
            if chunk.get("citation_id")
        ]
        proposal = backend.propose_escalation(
            state["patient_id"], int(state["as_of_hour"]), reason, citations
        )
        return {
            "proposal": proposal,
            "alert_id": proposal.get("alert_id"),
            "status": "pending_approval",
        }

    return node


def _no_alert(state: AgentState) -> dict:
    return {
        "status": "no_alert",
        "reason": f"risk {state.get('risk', 0):.4f} below threshold "
        f"{state.get('threshold', 0):.4f}",
    }


def _no_guidance(state: AgentState) -> dict:
    return {"status": "no_guidance", "reason": "no chunks retrieved"}


def _rejected(state: AgentState) -> dict:
    return {
        "status": "rejected",
        "reason": "deterministic validator rejected the draft: "
        + str(state.get("validation", {}).get("flags", [])),
    }


def build_graph(backend: AgentBackend):
    graph = StateGraph(AgentState)
    graph.add_node("assess", _assess(backend))
    graph.add_node("retrieve", _retrieve(backend))
    graph.add_node("draft", _draft(backend))
    graph.add_node("propose", _propose(backend))
    graph.add_node("no_alert", _no_alert)
    graph.add_node("no_guidance", _no_guidance)
    graph.add_node("rejected", _rejected)

    graph.set_entry_point("assess")
    graph.add_conditional_edges(
        "assess", _after_assess, {"retrieve": "retrieve", "no_alert": "no_alert"}
    )
    graph.add_conditional_edges(
        "retrieve", _after_retrieve, {"draft": "draft", "no_guidance": "no_guidance"}
    )
    graph.add_conditional_edges(
        "draft", _after_draft, {"propose": "propose", "rejected": "rejected"}
    )
    graph.add_edge("propose", END)
    graph.add_edge("no_alert", END)
    graph.add_edge("no_guidance", END)
    graph.add_edge("rejected", END)
    return graph.compile()


def run_agent(patient_id: str, as_of_hour: int, backend: AgentBackend) -> dict:
    app = build_graph(backend)
    return dict(app.invoke({"patient_id": patient_id, "as_of_hour": int(as_of_hour)}))


def render_note(state: dict) -> str:
    sbar = state.get("sbar", {})
    lines = [
        f"SBAR — {state.get('patient_id')} hour {state.get('as_of_hour')}",
        "",
        f"S: {sbar.get('situation', '')}",
        f"B: {sbar.get('background', '')}",
        f"A: {sbar.get('assessment', '')}",
        f"R: {sbar.get('recommendation', '')}",
        "",
        "Sources:",
    ]
    for chunk in state.get("chunks", []):
        lines.append(
            f"[{chunk.get('citation_id')}] {chunk.get('doc')} — {chunk.get('source')}"
        )
    if state.get("validation", {}).get("flags"):
        lines.append(f"Validator flags: {state['validation']['flags']}")
    return "\n".join(lines)


def execute_approved(
    state: dict,
    *,
    registry: ToolRegistry,
    approval_token: str | None,
    tool: str = "page_rapid_response",
    actor: str = "clinician",
) -> dict:
    """Execute the irreversible action only after a human approval token.

    No token, a stale token, a token for another patient/alert/tool, or a
    replayed token all raise from the approval authority / tool registry.
    """
    if state.get("status") != "pending_approval":
        raise AgentError(
            f"agent state is {state.get('status')!r}, not pending_approval; "
            "there is nothing to execute"
        )
    alert_id = state.get("alert_id")
    if not alert_id:
        raise AgentError("proposal has no alert_id")
    if tool not in ("page_rapid_response", "write_ehr_note"):
        raise AgentError(f"{tool} is not an approval-gated write tool")

    return registry.call(
        tool,
        {
            "alert_id": alert_id,
            "patient_id": state["patient_id"],
            "note": render_note(state),
        },
        approval_token=approval_token,
        actor=actor,
    )
