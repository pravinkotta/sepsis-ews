"""Load `.env` for CLI and service startup.

Existing environment variables win (`override=False`). Values are never logged;
only the presence of a key is reported by callers.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_env(path: Path | None = None) -> bool:
    env_path = Path(path) if path else ROOT / ".env"
    if not env_path.exists():
        return False
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=False)
        return True
    except ImportError:  # pragma: no cover - python-dotenv is in requirements
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
        return True
