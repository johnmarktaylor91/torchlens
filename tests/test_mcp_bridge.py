"""Tests for the MCP bridge (``torchlens.bridge.mcp``) over the agent registry."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.bridge import mcp as tlmcp


@pytest.fixture()
def saved_trace_path(tmp_path: Path) -> Path:
    """Save a small trace artifact and return its path.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.

    Returns
    -------
    Path
        Path of the saved ``.tlspec`` artifact.
    """

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2)).eval()
    log = tl.trace(model, torch.randn(2, 4), save=tl.func("relu"))
    path = tmp_path / "trace.tlspec"
    tl.save(log, str(path))
    return path


def test_tool_specs_are_wellformed_json_schemas() -> None:
    """Every declared tool has a name, description, object schema, annotations."""

    names = [spec["name"] for spec in tlmcp.TOOL_SPECS]
    assert len(names) == len(set(names))
    # Nine registry tools plus the bridge-local experiment-ledger extension.
    ledger_names = {spec["name"] for spec in tlmcp.LEDGER_TOOL_SPECS}
    assert ledger_names == {
        "torchlens_ledger_overview",
        "torchlens_ledger_entry",
        "torchlens_ledger_evidence",
    }
    assert len(names) == 9 + len(ledger_names)
    for spec in tlmcp.TOOL_SPECS:
        assert spec["name"].startswith("torchlens_")
        assert spec["description"]
        schema = spec["input_schema"]
        assert schema["type"] == "object"
        assert schema.get("additionalProperties") is False
        assert spec["annotations"] == {"readOnlyHint": True, "idempotentHint": True}
        # Registry tools serve agent envelopes; the ledger extension serves
        # the raw torchlens.ledger_*_v1 payloads by contract.
        expected_prefix = (
            "torchlens.ledger_" if spec["name"] in ledger_names else "torchlens.agent."
        )
        assert spec["output_schema"].startswith(expected_prefix)
        json.dumps(spec)


def test_call_tool_refusals_teach_the_fix() -> None:
    """Unknown tools, bad args, and missing paths refuse typed with remedies."""

    with pytest.raises(ValueError, match="served tools") as tool_exc:
        tlmcp.call_tool("torchlens_nope")
    assert tool_exc.value.fields["code"] == "agent_tool_unknown"
    with pytest.raises(ValueError, match="requires 'path'") as args_exc:
        tlmcp.call_tool("torchlens_dump", {})
    assert args_exc.value.fields["code"] == "agent_argument_invalid"
    with pytest.raises(ValueError, match="tl.save") as path_exc:
        tlmcp.call_tool("torchlens_dump", {"path": "/nonexistent/file.tlspec"})
    assert path_exc.value.fields["code"] == "agent_artifact_unreadable"


def test_doctor_and_api_map_tools_are_json_safe() -> None:
    """Environment and API-map tools return enveloped JSON-safe payloads."""

    doctor = tlmcp.call_tool("torchlens_doctor")
    assert doctor["schema"] == "torchlens.agent.doctor.v1"
    assert doctor["artifact"] is None
    assert doctor["data"]["checks"]
    assert {"name", "status", "detail"} <= set(doctor["data"]["checks"][0])
    json.dumps(doctor)

    api_map = tlmcp.call_tool("torchlens_api_map")
    assert api_map["schema"] == "torchlens.agent.api_map.v2"
    names = {row["name"] for row in api_map["data"]["names"]}
    assert set(tl.__all__) <= names
    capabilities = api_map["data"]["capabilities"]
    assert capabilities["writes"] is False
    assert capabilities["executes_user_code"] is False
    # The capabilities block declares the agent-registry roster; MCP serves
    # it plus the bridge-local experiment-ledger extension.
    ledger_names = {spec["name"] for spec in tlmcp.LEDGER_TOOL_SPECS}
    assert ledger_names.isdisjoint(capabilities["tools"])
    assert set(capabilities["tools"]) | ledger_names == {spec["name"] for spec in tlmcp.TOOL_SPECS}
    assert "tl.report" in api_map["data"]["submodules_not_in_all"]
    json.dumps(api_map)


def test_api_map_detail_mode_serves_one_full_record() -> None:
    """Per-name detail mode returns the full normalized signature."""

    detail = tlmcp.call_tool("torchlens_api_map", {"name": "trace"})["data"]["detail"]
    assert detail["name"] == "trace"
    assert detail["signature"].startswith("trace(")
    assert detail["example"]


def test_artifact_tools_drive_the_same_public_surface(saved_trace_path: Path) -> None:
    """Overview, dump, and explain tools mirror the in-process spellings."""

    overview = tlmcp.call_tool(
        "torchlens_overview", {"path": str(saved_trace_path), "mode": "folded"}
    )
    assert overview["schema"] == "torchlens.agent.overview_folded.v1"
    assert overview["data"]["capture"]["capture_status"] == "complete"
    # F09 numbers core: "operations" counts every tracked tensor row (5,
    # incl. input/output); "compute_ops" is the 3-op compute grain.
    assert overview["data"]["counts"]["operations"] == 5
    assert overview["data"]["counts"]["compute_ops"] == 3
    assert overview["artifact"]["kind"] == "trace"
    assert overview["artifact"]["id"].startswith("sha256:")

    dump = tlmcp.call_tool(
        "torchlens_dump", {"path": str(saved_trace_path), "view": "graph", "max_rows": 2}
    )
    assert dump["schema"] == "torchlens.agent.dump.v1"
    assert dump["truncation"]["omitted"] == 3  # 5 graph rows (incl. input/output), page of 2
    assert dump["data"]["next"] is not None
    json.dumps(dump)

    report = tlmcp.call_tool(
        "torchlens_explain", {"path": str(saved_trace_path), "max_tokens": 100}
    )
    assert "Capture status" in report["data"]["report"]
    assert "Truncation" in report["data"]["report"]


def test_server_adapter_serves_the_declared_tools(saved_trace_path: Path) -> None:
    """The mcp>=2.0 server exposes exactly the registry and dispatches calls."""

    pytest.importorskip("mcp")
    pytest.importorskip("mcp.server")

    server = tlmcp._build_server()

    async def _drive() -> None:
        """List tools and run one artifact call through the live server."""

        tools = await server.list_tools()
        assert [tool.name for tool in tools] == [spec["name"] for spec in tlmcp.TOOL_SPECS]
        result = await server.call_tool(
            "torchlens_dump", {"path": str(saved_trace_path), "view": "full"}
        )
        assert result.is_error is False
        payload = result.structured_content
        assert payload["schema"] == "torchlens.agent.dump.v1"
        assert payload["data"]["schema"] == "torchlens.agent_trace.v1"

    asyncio.run(_drive())


def test_main_without_mcp_teaches_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing mcp package refuses with the extra named."""

    import builtins

    real_import = builtins.__import__

    def _blocked(name: str, *args: object, **kwargs: object) -> object:
        """Simulate an environment without the mcp package."""

        if name == "mcp.server" or name.startswith("mcp"):
            raise ImportError("No module named 'mcp'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    with pytest.raises(ImportError, match=r"torchlens\[mcp\]"):
        tlmcp.main()
