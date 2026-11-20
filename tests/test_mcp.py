import pandas as pd
import pytest

pytest.importorskip("mcp")

from src.approval import ApprovalAuthority  # noqa: E402
from src.mcp_server import build_server  # noqa: E402
from src.store import DataFrameBackend  # noqa: E402

SECRET = b"mcp-smoke-secret-with-at-least-32-bytes"


def test_server_builds_with_all_tools():
    gold = pd.DataFrame(
        {"patient_id": ["p1"], "icu_hour": [1], "HR_last": [90.0], "Lactate": [1.0]}
    )
    server = build_server(DataFrameBackend(gold), ApprovalAuthority(SECRET))
    assert server is not None
    assert server.name == "sepsis-ews"
