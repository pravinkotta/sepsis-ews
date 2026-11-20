"""Protocol-level MCP check over stdio.

    python -m src.mcp_check

Spawns the real MCP server, initializes a ClientSession, lists the tools, calls
read tools, and attempts a write without a token. Writes the result to
`artifacts/mcp_tools.json` for the DoD evidence.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .config import ARTIFACTS
from .tools import FORBIDDEN_TOOLS, TOOLS

SECRET = "mcp-check-secret-" + "x" * 24


def _text(result) -> str:
    parts = []
    for item in getattr(result, "content", []) or []:
        text = getattr(item, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts)


def _is_error(result) -> bool:
    return bool(
        getattr(result, "isError", False)
        or getattr(result, "is_error", False)
    )


async def run(artifacts: Path) -> dict:
    env = {**os.environ, "SEPSIS_APPROVAL_SECRET": SECRET}
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "src.mcp_server", "--artifacts", str(artifacts)],
        env=env,
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = await session.list_tools()
            tools = list(listed.tools)
            names = [tool.name for tool in tools]
            descriptions = {tool.name: (tool.description or "") for tool in tools}

            patient = await session.call_tool(
                "get_patient_state",
                {"patient_id": "p000206", "as_of_hour": 15},
            )
            score = await session.call_tool(
                "score_sepsis",
                {"patient_id": "p000206", "as_of_hour": 15},
            )

            blocked = False
            page_detail = ""
            try:
                page = await session.call_tool(
                    "page_rapid_response",
                    {
                        "alert_id": "mcp-check",
                        "patient_id": "p000206",
                        "note": "mcp protocol check",
                    },
                )
                page_detail = _text(page)
                blocked = _is_error(page) or "approval" in page_detail.lower()
            except Exception as exc:  # noqa: BLE001 - a transport-level refusal also counts
                blocked = True
                page_detail = f"{type(exc).__name__}: {exc}"

            return {
                "server": "sepsis-ews",
                "transport": "stdio",
                "tool_count": len(names),
                "tools": names,
                "descriptions": descriptions,
                "registry_count": len(TOOLS),
                "forbidden_present": sorted(set(names) & FORBIDDEN_TOOLS),
                "get_patient_state": {
                    "is_error": _is_error(patient),
                    "text": _text(patient)[:500],
                },
                "score_sepsis": {
                    "is_error": _is_error(score),
                    "text": _text(score)[:500],
                },
                "write_without_token_blocked": blocked,
                "write_detail": page_detail[:500],
                "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MCP protocol check")
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    args = parser.parse_args(argv)

    result = asyncio.run(asyncio.wait_for(run(args.artifacts), timeout=180))
    (args.artifacts / "mcp_tools.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({key: result[key] for key in (
        "tool_count", "tools", "forbidden_present",
        "write_without_token_blocked",
    )}, indent=2))
    print(f"evidence -> {args.artifacts / 'mcp_tools.json'}")

    ok = (
        result["tool_count"] == 9
        and not result["forbidden_present"]
        and result["write_without_token_blocked"]
        and not result["get_patient_state"]["is_error"]
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
