"""Module-granularity netron export (lane F14, netron memo D-11/D-12, ruling N2).

Per-call FunctionProtos with recursive nesting, depth-1 interim default,
single-op inlining with FunctionProto DELETION, non-empty root, per-function
valueInfo, signature-style formal input names, per-call keying on reused
modules (compo row C7), and the extent budget as WARN-plus-remedy.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl


class _Block(nn.Module):
    """Multi-op child block (linear + relu)."""

    def __init__(self) -> None:
        """Build the child linear layer."""

        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Linear then relu."""

        return torch.relu(self.fc(x))


class _Deep(nn.Module):
    """Two-level hierarchy: outer wraps two blocks plus a root op."""

    def __init__(self) -> None:
        """Build both blocks."""

        super().__init__()
        self.first = _Block()
        self.second = _Block()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Chain the blocks and add a root-level op."""

        return self.first(x) + self.second(x)


def _export(log: Any, path: Path, **kwargs: Any) -> dict[str, Any]:
    """Export and parse one artifact."""

    return json.loads(tl.export.netron(log, path, **kwargs).read_text(encoding="utf-8"))


def test_module_is_the_file_export_default(tmp_path: Path) -> None:
    """Ruling N2: the bare file export defaults to the module projection."""

    log = tl.trace(_Deep(), torch.randn(2, 4))
    payload = _export(log, tmp_path / "default.json")
    props = {row["key"]: row["value"] for row in payload["metadataProps"]}
    assert props["torchlens.granularity"] == "module"
    assert props["torchlens.module_depth"] == "1"


def test_per_call_functions_and_drill_down_shapes(tmp_path: Path) -> None:
    """Each multi-op call gets its own FunctionProto with typed body values."""

    log = tl.trace(_Deep(), torch.randn(2, 4))
    payload = _export(log, tmp_path / "module.json")
    functions = {fn["name"]: fn for fn in payload.get("functions", [])}
    assert set(functions) == {"first", "second"}
    for fn in functions.values():
        assert fn["node"], "no empty FunctionProtos (a blank drill-down room is noise)"
        assert fn["input"] and fn["input"][0] == "x", "formal names are signatures"
        typed = {row["name"] for row in fn.get("valueInfo", [])}
        for node in fn["node"]:
            assert node["output"][0] in typed, "shapes render inside drill-downs"
    root_ops = {node["name"] for node in payload["graph"]["node"]}
    assert "add_1_5" in root_ops, "root-level op stays at root"
    assert {"first", "second"} <= root_ops, "call sites are root nodes"


def test_single_op_module_is_inlined_and_function_deleted(tmp_path: Path) -> None:
    """A one-op module inlines; its FunctionProto is DELETED, not emptied."""

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    log = tl.trace(model, torch.randn(2, 4))
    payload = _export(log, tmp_path / "inline.json")
    assert payload.get("functions", []) == []
    names = [node["name"] for node in payload["graph"]["node"]]
    assert names == ["linear_1_1", "relu_1_2"]


def test_reused_module_gets_per_call_functions(tmp_path: Path) -> None:
    """Compo row C7: one module called twice keys per call, pass-qualified."""

    class _Reuse(nn.Module):
        """Calls one block twice in a single forward."""

        def __init__(self) -> None:
            """Build the shared block."""

            super().__init__()
            self.block = _Block()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Apply the same block twice."""

            return self.block(self.block(x))

    log = tl.trace(_Reuse(), torch.randn(2, 4))
    payload = _export(log, tmp_path / "reuse.json")
    functions = [fn["name"] for fn in payload.get("functions", [])]
    assert functions == ["block:1", "block:2"], "per-call keying, pass-qualified"
    for fn in payload["functions"]:
        assert fn["node"], "zero empty FunctionProtos"
    calls = [n for n in payload["graph"]["node"] if n["domain"] == "ai.torchlens.module"]
    assert [call["name"] for call in calls] == ["block:1", "block:2"]
    assert calls[1]["input"] == calls[0]["output"], "second call consumes the first's output"


def test_depth_two_dissolves_the_top_level(tmp_path: Path) -> None:
    """depth=2 flattens level-1 calls into the root and retains level 2."""

    class _Wrapper(nn.Module):
        """Wraps _Deep one level down."""

        def __init__(self) -> None:
            """Build the wrapped model."""

            super().__init__()
            self.inner = _Deep()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Delegate to the wrapped model."""

            return self.inner(x)

    log = tl.trace(_Wrapper(), torch.randn(2, 4))
    shallow = _export(log, tmp_path / "d1.json", depth=1)
    deep = _export(log, tmp_path / "d2.json", depth=2)
    # depth=1: ONE root box (inner) with the blocks nested inside it.
    shallow_root_calls = [
        n["name"] for n in shallow["graph"]["node"] if n["domain"] == "ai.torchlens.module"
    ]
    assert shallow_root_calls == ["inner"]
    assert {fn["name"] for fn in shallow.get("functions", [])} == {
        "inner",
        "inner.first",
        "inner.second",
    }
    # depth=2: the top level dissolves; the blocks become the root boxes.
    deep_root_calls = [
        n["name"] for n in deep["graph"]["node"] if n["domain"] == "ai.torchlens.module"
    ]
    assert set(deep_root_calls) == {"inner.first", "inner.second"}
    assert {fn["name"] for fn in deep.get("functions", [])} == {"inner.first", "inner.second"}
    props = {row["key"]: row["value"] for row in deep["metadataProps"]}
    assert props["torchlens.module_depth"] == "2", "the chosen depth is recorded"


def test_nested_functions_resolve_one_click_deep(tmp_path: Path) -> None:
    """A retained call inside a retained call nests as a function-body call site."""

    class _Outer2(nn.Module):
        """Outer module whose single child itself contains a block."""

        def __init__(self) -> None:
            """Build the nested hierarchy."""

            super().__init__()
            self.mid = _Deep()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Delegate plus one outer op."""

            return torch.sigmoid(self.mid(x))

    log = tl.trace(_Outer2(), torch.randn(2, 4))
    payload = _export(log, tmp_path / "nested.json")
    functions = {fn["name"]: fn for fn in payload.get("functions", [])}
    assert "mid" in functions
    nested_calls = [
        node["opType"]
        for node in functions["mid"]["node"]
        if node["domain"] == "ai.torchlens.module"
    ]
    assert set(nested_calls) == {"mid.first", "mid.second"}
    assert set(nested_calls) <= set(functions), "nested call sites resolve to functions"


def test_module_attrs_say_inclusive(tmp_path: Path) -> None:
    """Module-call metrics are labelled inclusive (wording contract D-17)."""

    log = tl.trace(_Deep(), torch.randn(2, 4))
    payload = _export(log, tmp_path / "attrs.json")
    call = next(n for n in payload["graph"]["node"] if n["domain"] == "ai.torchlens.module")
    names = [attr["name"] for attr in call.get("attribute", [])]
    assert "module_path" in names
    assert any(name.endswith("_inclusive_us") for name in names)
    assert names == sorted(names, key=str.lower)


def test_extent_budget_warns_with_remedy(tmp_path: Path) -> None:
    """Past ~27 ranks the export warns and names the remedy (memo D-12)."""

    class _Rope(nn.Module):
        """A 40-op serial chain -- a rope, not a diagram."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Chain forty adds."""

            for _ in range(40):
                x = x + 1.0
            return x

    log = tl.trace(_Rope(), torch.randn(2))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tl.export.netron(log, tmp_path / "rope.json", granularity="op")
    codes = [getattr(w.message, "fields", {}).get("code") for w in caught]
    assert "netron_extent_budget_exceeded" in codes
    message = str(next(w.message for w in caught if getattr(w.message, "fields", {}).get("code")))
    assert "granularity='module'" in message


def test_root_graph_never_empty(tmp_path: Path) -> None:
    """The root graph is asserted non-empty (netron would open a function)."""

    log = tl.trace(_Deep(), torch.randn(2, 4))
    payload = _export(log, tmp_path / "root.json")
    assert payload["graph"]["node"], "non-empty root graph"


def test_cycle_fail_soft_provokes_the_fallback_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The DAG belt fail-softs to the op projection with the coded warning.

    Per-call keying makes real function-call cycles structurally impossible,
    so the belt is provoked by forcing the named assertion to report a cycle
    -- the warning code ``netron_module_projection_fallback`` and the
    ``torchlens.granularity_fallback_reason`` metadata are the contract.
    """

    import json

    from torchlens.export import _netron

    monkeypatch.setattr(_netron, "_assert_function_dag", lambda projection: False)
    log = tl.trace(_Deep(), torch.randn(2, 4))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        path = tl.export.netron(log, tmp_path / "fallback.json")
    codes = [getattr(w.message, "fields", {}).get("code") for w in caught]
    assert "netron_module_projection_fallback" in codes
    payload = json.loads(path.read_text(encoding="utf-8"))
    props = {row["key"]: row["value"] for row in payload["metadataProps"]}
    assert props["torchlens.granularity_fallback_reason"] == "module_call_cycle"
    assert props["torchlens.function_count"] == "0", "the fallback IS the op projection"
