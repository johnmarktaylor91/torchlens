"""A09 listA row 25b + agent stage-0 items 2/3/5: MCP bounds, manifest-first
load plan, content-digest identity, served-schema parity.

One MCP tool call must never materialize an unbounded artifact (the OOM
vector), every load decision is made from manifest-DECLARED numbers and
disclosed as ``load_plan``, artifact identity is the manifest content digest
(``(path, mtime)`` is only a verified stat hint), and the served input
schemas cannot drift from the pure handlers.
"""

from __future__ import annotations

import inspect
import shutil
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.bridge import mcp

pytestmark = pytest.mark.smoke


@pytest.fixture()
def artifact(tmp_path: Path) -> Path:
    """One small saved Trace artifact."""

    torch.manual_seed(0)
    trace = tl.trace(nn.Sequential(nn.Linear(3, 4), nn.ReLU()), torch.randn(2, 3))
    destination = tmp_path / "trace.tlspec"
    tl.save(trace, destination)
    return destination


@pytest.fixture(autouse=True)
def _fresh_caches() -> None:
    """Isolate the digest cache and hint map per test."""

    mcp._TRACE_CACHE.clear()
    mcp._DIGEST_HINTS.clear()


def test_load_plan_disclosed_on_every_artifact_tool(artifact: Path) -> None:
    """load_overview/agent_dump/explain all disclose the load plan."""

    overview = mcp.call_tool("torchlens_load_overview", {"path": str(artifact)})
    dump = mcp.call_tool("torchlens_agent_dump", {"path": str(artifact)})
    explain_result = mcp.call_tool("torchlens_explain", {"path": str(artifact)})
    for payload in (overview, dump, explain_result):
        plan = payload["load_plan"]
        assert plan["mode"] == "eager"
        assert isinstance(plan["declared_payload_bytes"], int)
        assert plan["declared_payload_count"] > 0
        assert "reason" in plan


def test_declared_bytes_over_threshold_load_lazily(
    artifact: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Above the eager threshold the load goes lazy -- blobs stay on disk."""

    monkeypatch.setattr(mcp, "EAGER_LOAD_MAX_BYTES", 1)
    overview = mcp.call_tool("torchlens_load_overview", {"path": str(artifact)})
    assert overview["load_plan"]["mode"] == "lazy"
    trace, plan = mcp._load_trace(str(artifact))
    assert plan["mode"] == "lazy"
    assert trace.payload_load_status == "loaded_lazy"


def test_memory_shaped_refusal_plant(artifact: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The declared-size gate REFUSES typed before any blob is opened."""

    monkeypatch.setattr(mcp, "LOAD_MAX_PAYLOAD_ENTRIES", 0)
    with pytest.raises(ValueError, match="Refusing to load"):
        mcp.call_tool("torchlens_load_overview", {"path": str(artifact)})
    with pytest.raises(ValueError, match="single-tool-call bound"):
        mcp.call_tool("torchlens_agent_dump", {"path": str(artifact)})


def test_agent_dump_is_bounded_by_default(artifact: Path) -> None:
    """Omitted max_ops means the server default cap, never every op row."""

    dump = mcp.call_tool("torchlens_agent_dump", {"path": str(artifact)})
    # Small artifact: no truncation fires, but the cap was applied.
    assert dump["truncation"] is None
    assert len(dump["ops"]) <= mcp.DUMP_DEFAULT_MAX_OPS


def test_agent_dump_default_cap_truncates_with_disclosure(
    artifact: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When ops exceed the default cap the dump truncates and DISCLOSES."""

    monkeypatch.setattr(mcp, "DUMP_DEFAULT_MAX_OPS", 2)
    dump = mcp.call_tool("torchlens_agent_dump", {"path": str(artifact)})
    assert len(dump["ops"]) == 2
    assert dump["truncation"]["ops_omitted"] > 0


def test_agent_dump_ceiling_refuses_typed(artifact: Path) -> None:
    """An explicit max_ops above the served ceiling refuses with the bound."""

    with pytest.raises(ValueError, match="served ceiling"):
        mcp.call_tool(
            "torchlens_agent_dump",
            {"path": str(artifact), "max_ops": mcp.DUMP_MAX_OPS_CEILING + 1},
        )


def test_artifact_identity_is_content_digest(artifact: Path, tmp_path: Path) -> None:
    """Identical content at a different path hits the same cache entry."""

    copy_path = tmp_path / "copy.tlspec"
    shutil.copytree(artifact, copy_path)
    trace_a, _ = mcp._load_trace(str(artifact))
    trace_b, _ = mcp._load_trace(str(copy_path))
    assert trace_a is trace_b  # digest-keyed, not path-keyed


def test_changed_content_at_same_path_is_not_served_stale(artifact: Path, tmp_path: Path) -> None:
    """A rename-replace at the same path serves the NEW content."""

    trace_a, _ = mcp._load_trace(str(artifact))
    torch.manual_seed(1)
    other = tl.trace(nn.Sequential(nn.Linear(5, 2), nn.Tanh()), torch.randn(2, 5))
    other_path = tmp_path / "other.tlspec"
    tl.save(other, other_path)
    shutil.rmtree(artifact)
    shutil.move(str(other_path), str(artifact))
    trace_b, _ = mcp._load_trace(str(artifact))
    assert trace_b is not trace_a
    assert trace_b.model_class_name == "Sequential"
    assert int(trace_b.num_params) == int(other.num_params)


def test_served_schema_field_parity() -> None:
    """TOOL_SPECS input schemas match the pure handlers field-for-field."""

    assert set(mcp.TOOL_HANDLERS) == {spec["name"] for spec in mcp.TOOL_SPECS}
    for spec in mcp.TOOL_SPECS:
        handler = mcp.TOOL_HANDLERS[spec["name"]]
        parameters = inspect.signature(handler).parameters
        schema = spec["input_schema"]
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        declared = set(schema.get("properties", {}))
        assert declared == set(parameters), (
            f"{spec['name']}: served schema fields {sorted(declared)} != "
            f"handler parameters {sorted(parameters)}"
        )
        required = {
            name
            for name, parameter in parameters.items()
            if parameter.default is inspect.Parameter.empty
        }
        assert set(schema.get("required", [])) == required, (
            f"{spec['name']}: served required fields drifted from the handler"
        )


def test_dump_and_report_are_escape_free(artifact: Path) -> None:
    """MCP responses are publishable data: no ESC bytes anywhere."""

    dump = mcp.call_tool("torchlens_agent_dump", {"path": str(artifact)})
    explain_result = mcp.call_tool("torchlens_explain", {"path": str(artifact)})
    assert "\x1b" not in str(dump)
    assert "\x1b" not in str(explain_result)
