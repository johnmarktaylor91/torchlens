"""A09 listA row 25b + agent stage-0 items 2/3/5 over the F29 registry: MCP
bounds, manifest-first load plan, content-digest identity, served-schema
parity, memory-shaped refusal plant.

One MCP tool call must never materialize an unbounded artifact (the OOM
vector), every load decision is made from manifest-DECLARED numbers and
disclosed as ``load_plan``, artifact identity is the manifest content digest
(``(path, mtime)`` is only a verified stat hint), and the served declarations
cannot drift from the registry (field-level parity, the F29 release gate).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.agent import _artifacts, _budgets, list_tools, tool_specs
from torchlens.bridge import mcp


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

    _artifacts._TRACE_CACHE.clear()
    _artifacts._DIGEST_HINTS.clear()


def test_load_plan_disclosed_on_every_artifact_tool(artifact: Path) -> None:
    """overview/dump/explain all disclose the manifest-first load plan."""

    overview = mcp.call_tool("torchlens_overview", {"path": str(artifact)})
    dump = mcp.call_tool("torchlens_dump", {"path": str(artifact)})
    explain_result = mcp.call_tool("torchlens_explain", {"path": str(artifact)})
    for payload in (overview, dump, explain_result):
        plan = payload["data"]["load_plan"]
        assert plan["mode"] == "eager"
        assert isinstance(plan["declared_payload_bytes"], int)
        assert plan["declared_payload_count"] > 0
        assert "reason" in plan


def test_declared_bytes_over_threshold_load_lazily(
    artifact: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Above the eager threshold the load goes lazy -- blobs stay on disk."""

    monkeypatch.setattr(_artifacts, "EAGER_LOAD_MAX_BYTES", 1)
    overview = mcp.call_tool("torchlens_overview", {"path": str(artifact)})
    assert overview["data"]["load_plan"]["mode"] == "lazy"
    trace, plan, _ = _artifacts.load_trace(str(artifact))
    assert plan["mode"] == "lazy"
    assert trace.payload_load_status == "loaded_lazy"


def test_memory_shaped_refusal_plant(artifact: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The declared-size gate REFUSES typed before any blob is opened."""

    monkeypatch.setattr(_artifacts, "LOAD_MAX_PAYLOAD_ENTRIES", 0)
    with pytest.raises(ValueError, match="Refusing to load") as overview_exc:
        mcp.call_tool("torchlens_overview", {"path": str(artifact)})
    assert overview_exc.value.fields["code"] == "agent_artifact_load_refused"
    with pytest.raises(ValueError, match="single-tool-call bound"):
        mcp.call_tool("torchlens_dump", {"path": str(artifact)})


def test_row_tools_are_bounded_by_default(artifact: Path) -> None:
    """Omitted max_rows means the served default cap, never every row."""

    dump = mcp.call_tool("torchlens_dump", {"path": str(artifact), "view": "graph"})
    assert dump["limits"]["max_rows"] == _budgets.DEFAULT_MAX_ROWS
    assert len(dump["data"]["rows"]) <= _budgets.DEFAULT_MAX_ROWS
    # Small artifact: no truncation fires, but the cap was applied.
    assert dump["truncation"] is None


def test_row_cap_truncates_with_disclosure_and_pages(artifact: Path) -> None:
    """Over-cap graph rows truncate with disclosure and a continuation."""

    dump = mcp.call_tool("torchlens_dump", {"path": str(artifact), "view": "graph", "max_rows": 2})
    assert len(dump["data"]["rows"]) == 2
    assert dump["truncation"]["omitted"] > 0
    assert dump["truncation"]["how_to_get_more"]
    next_struct = dump["data"]["next"]
    assert next_struct["offset"] == 2
    page2 = mcp.call_tool(
        "torchlens_dump",
        {"path": str(artifact), "view": "graph", "max_rows": 2, "continuation": next_struct},
    )
    first_labels = {row["label"] for row in dump["data"]["rows"]}
    assert first_labels.isdisjoint({row["label"] for row in page2["data"]["rows"]})


def test_limit_above_served_ceiling_refuses_typed(artifact: Path) -> None:
    """An explicit max_rows above the served ceiling refuses with the bound."""

    with pytest.raises(ValueError, match="served ceiling") as exc:
        mcp.call_tool(
            "torchlens_dump",
            {"path": str(artifact), "view": "graph", "max_rows": _budgets.DEFAULT_MAX_ROWS + 1},
        )
    assert exc.value.fields["code"] == "agent_limit_invalid"


def test_artifact_identity_is_content_digest(artifact: Path, tmp_path: Path) -> None:
    """Identical content at a different path hits the same cache entry."""

    copy_path = tmp_path / "copy.tlspec"
    shutil.copytree(artifact, copy_path)
    trace_a, _, block_a = _artifacts.load_trace(str(artifact))
    trace_b, _, block_b = _artifacts.load_trace(str(copy_path))
    assert trace_a is trace_b  # digest-keyed, not path-keyed
    assert block_a["id"] == block_b["id"]
    assert block_a["id"].startswith("sha256:")


def test_changed_content_at_same_path_is_not_served_stale(artifact: Path, tmp_path: Path) -> None:
    """A rename-replace at the same path serves the NEW content."""

    trace_a, _, _ = _artifacts.load_trace(str(artifact))
    torch.manual_seed(1)
    other = tl.trace(nn.Sequential(nn.Linear(5, 2), nn.Tanh()), torch.randn(2, 5))
    other_path = tmp_path / "other.tlspec"
    tl.save(other, other_path)
    shutil.rmtree(artifact)
    shutil.move(str(other_path), str(artifact))
    trace_b, _, _ = _artifacts.load_trace(str(artifact))
    assert trace_b is not trace_a
    assert trace_b.model_class_name == "Sequential"
    assert int(trace_b.num_params) == int(other.num_params)


def test_served_schema_field_parity() -> None:
    """Served declarations == registry, field-by-field (the F29 release gate).

    ``list_tools()`` is what MCP serves; the registry specs are the single
    source. Every field the host sees -- name, description, input schema
    (properties, required, additionalProperties), output schema id,
    annotations, CLI verb, default limits -- must match the registry exactly.
    """

    served = {declaration["name"]: declaration for declaration in list_tools()}
    specs = {spec.name: spec for spec in tool_specs()}
    assert set(served) == set(specs)
    # The handler map carries the registry plus the bridge-local
    # experiment-ledger extension (raw torchlens.experiment._mcp handlers).
    ledger_names = {spec["name"] for spec in mcp.LEDGER_TOOL_SPECS}
    assert set(mcp.TOOL_HANDLERS) == set(specs) | ledger_names
    for name, spec in specs.items():
        declaration = served[name]
        assert declaration["description"] == spec.description
        assert declaration["input_schema"] == spec.input_schema
        assert declaration["output_schema"] == spec.output_schema
        assert declaration["annotations"] == {
            "readOnlyHint": spec.read_only,
            "idempotentHint": spec.idempotent,
        }
        assert declaration["cli_verb"] == spec.cli_verb
        assert declaration["default_limits"] == dict(spec.default_limits)
        assert spec.input_schema["additionalProperties"] is False
        assert mcp.TOOL_HANDLERS[name] is spec.handler
    # The bridge's frozen TOOL_SPECS snapshot cannot drift from the registry:
    # the registry rows lead verbatim, the ledger extension trails.
    assert list(mcp.TOOL_SPECS) == list_tools() + list(mcp.LEDGER_TOOL_SPECS)


def test_memory_shaped_stats_reads_do_not_attach(artifact: Path) -> None:
    """200 sequential payload_stats calls keep the lazily loaded ops payload-free.

    The MEMORY-shaped acceptance test (agent memo 3.2 part 5): the analytics
    path reads through the scoped NON-attaching primitive, so repeated stats
    calls against a cached lazy trace never rebuild the eager resident set.
    """

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_artifacts, "EAGER_LOAD_MAX_BYTES", 1)
        trace, plan, _ = _artifacts.load_trace(str(artifact))
        assert plan["mode"] == "lazy"
        for _ in range(200):
            mcp.call_tool("torchlens_payload_stats", {"path": str(artifact), "metrics": ["mean"]})
        resident = sum(
            1 for op in trace.layer_list if callable(op._slot) and op._slot("out") is not None
        )
        assert resident == 0, (
            f"{resident} payloads attached to the cached trace -- the stats "
            "path leaked through the attaching door"
        )


def test_dump_and_report_are_escape_free(artifact: Path) -> None:
    """MCP responses are publishable data: no ESC bytes anywhere."""

    dump = mcp.call_tool("torchlens_dump", {"path": str(artifact), "view": "full"})
    explain_result = mcp.call_tool("torchlens_explain", {"path": str(artifact)})
    assert "\x1b" not in str(dump)
    assert "\x1b" not in str(explain_result)
