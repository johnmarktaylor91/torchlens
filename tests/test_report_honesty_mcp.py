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
    trace.save(destination)
    return destination


@pytest.fixture(autouse=True)
def _fresh_caches() -> None:
    """Isolate the digest cache and hint map per test."""

    _artifacts._TRACE_CACHE.clear()
    _artifacts._DIGEST_HINTS.clear()


# Kept by the shrink-r4 coverage proof: sole branch coverage of _data_classes/_value_repr.py, report/_factcore.py and a sole campaign mutation kill.
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


# Kept by the shrink-r4 coverage proof: sole branch coverage of agent/_artifacts.py, agent/_registry.py.
def test_memory_shaped_refusal_plant(artifact: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The declared-size gate REFUSES typed before any blob is opened."""

    monkeypatch.setattr(_artifacts, "LOAD_MAX_PAYLOAD_ENTRIES", 0)
    with pytest.raises(ValueError, match="Refusing to load") as overview_exc:
        mcp.call_tool("torchlens_overview", {"path": str(artifact)})
    assert overview_exc.value.fields["code"] == "agent_artifact_load_refused"
    with pytest.raises(ValueError, match="single-tool-call bound"):
        mcp.call_tool("torchlens_dump", {"path": str(artifact)})


# Kept by the shrink-r4 coverage proof: sole branch coverage of agent/_registry.py.
def test_limit_above_served_ceiling_refuses_typed(artifact: Path) -> None:
    """An explicit max_rows above the served ceiling refuses with the bound."""

    with pytest.raises(ValueError, match="served ceiling") as exc:
        mcp.call_tool(
            "torchlens_dump",
            {"path": str(artifact), "view": "graph", "max_rows": _budgets.DEFAULT_MAX_ROWS + 1},
        )
    assert exc.value.fields["code"] == "agent_limit_invalid"


# Kept by the shrink-r4 coverage proof: sole branch coverage of _data_classes/_trace_groups_capture.py, _data_classes/trace.py.
def test_changed_content_at_same_path_is_not_served_stale(artifact: Path, tmp_path: Path) -> None:
    """A rename-replace at the same path serves the NEW content."""

    trace_a, _, _ = _artifacts.load_trace(str(artifact))
    torch.manual_seed(1)
    other = tl.trace(nn.Sequential(nn.Linear(5, 2), nn.Tanh()), torch.randn(2, 5))
    other_path = tmp_path / "other.tlspec"
    other.save(other_path)
    shutil.rmtree(artifact)
    shutil.move(str(other_path), str(artifact))
    trace_b, _, _ = _artifacts.load_trace(str(artifact))
    assert trace_b is not trace_a
    assert trace_b.source.class_name == "Sequential"
    assert int(trace_b.num_params) == int(other.num_params)


# Kept by the shrink-r4 coverage proof: sole branch coverage of agent/__init__.py.
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
    # The handler map carries exactly the registry.
    assert set(mcp.TOOL_HANDLERS) == set(specs)
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
    # The bridge's frozen TOOL_SPECS snapshot cannot drift from the registry.
    assert list(mcp.TOOL_SPECS) == list_tools()


# Kept by the shrink-r4 coverage proof: sole branch coverage of _io/payload_reader.py, agent/_stats.py.
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
            1 for op in trace._layer_list if callable(op._slot) and op._slot("out") is not None
        )
        assert resident == 0, (
            f"{resident} payloads attached to the cached trace -- the stats "
            "path leaked through the attaching door"
        )
