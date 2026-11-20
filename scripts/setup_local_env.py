"""Create/update the local `.env` secrets without printing them.

    python scripts/setup_local_env.py

Preserves `OPENAI_API_KEY`; rotates the approval secret and clinician key only
if they are missing (use --rotate to change them).
"""

from __future__ import annotations

import argparse
import secrets
from pathlib import Path

ENV = Path(__file__).resolve().parents[1] / ".env"


def _upsert(lines: list[str], key: str, value: str) -> None:
    for index, line in enumerate(lines):
        if line.startswith(key + "="):
            lines[index] = f"{key}={value}"
            return
    lines.append(f"{key}={value}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Local .env setup")
    parser.add_argument("--rotate", action="store_true", help="replace existing secrets")
    parser.add_argument("--no-llm", action="store_true", help="use extractive SBAR")
    args = parser.parse_args()

    lines = ENV.read_text(encoding="utf-8").splitlines() if ENV.exists() else []
    existing = {
        line.split("=", 1)[0]: line.split("=", 1)[1]
        for line in lines
        if "=" in line and not line.strip().startswith("#")
    }

    added = []
    if args.rotate or not existing.get("SEPSIS_APPROVAL_SECRET"):
        _upsert(lines, "SEPSIS_APPROVAL_SECRET", secrets.token_hex(32))
        added.append("SEPSIS_APPROVAL_SECRET")
    if args.rotate or not existing.get("SEPSIS_APPROVAL_API_KEY"):
        _upsert(lines, "SEPSIS_APPROVAL_API_KEY", secrets.token_urlsafe(24))
        added.append("SEPSIS_APPROVAL_API_KEY")
    _upsert(lines, "SEPSIS_USE_LLM", "0" if args.no_llm else "1")

    ENV.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"wrote {ENV}")
    print(f"keys present: {', '.join(sorted(existing) + added)}")
    print("values are not printed; read SEPSIS_APPROVAL_API_KEY from .env when approving.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
