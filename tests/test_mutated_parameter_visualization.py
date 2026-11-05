"""Forward renders draw a source node for an in-place-mutated ``nn.Parameter``.

Capture records ``with torch.no_grad(): self.temp.clamp_(0.001, 0.5)`` (MIX-HIC) as an
op whose parameter input is ``temp``, and binds later reads of ``temp`` to that op. The
render adds one cylinder node per mutated Parameter, filled with the parameter grey and
placed in its owning module's cluster, with an edge to every op that reads the
pre-mutation value. Parameters that are never mutated get no node and leave the DOT
unchanged.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._visualization.themes import resolve_theme

#: Parameter fills as the DEFAULT skin renders them.
TRAINABLE_PARAMS_BG_COLOR = resolve_theme("torchlens").semantic_palette["params_trainable"]
FROZEN_PARAMS_BG_COLOR = resolve_theme("torchlens").semantic_palette["params_frozen"]

pytest.importorskip("graphviz")


class _MixHicInner(nn.Module):
    """MIX-HIC-style temperature: clamp a scalar Parameter, then divide by it."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(3, 3)
        self.temp = nn.Parameter(0.07 * torch.ones([]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.lin(x)
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)
        return y / self.temp


class _MixHicToy(nn.Module):
    """Wrap the temperature module so the Parameter has a non-root owner."""

    def __init__(self) -> None:
        super().__init__()
        self.inner = _MixHicInner()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x) + 1


class _ReadBeforeAndAfter(nn.Module):
    """Read a Parameter, mutate it in place, then read it again."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.full((3,), 2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        before = x * self.scale
        with torch.no_grad():
            self.scale.mul_(3.0)
        return before + x * self.scale


class _FrozenMutated(nn.Module):
    """Mutate a frozen Parameter in place (no ``no_grad`` needed)."""

    def __init__(self) -> None:
        super().__init__()
        self.temp = nn.Parameter(torch.ones([]), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.temp.clamp_(0.001, 0.5)
        return x / self.temp


class _NeverMutated(nn.Module):
    """The MIX-HIC shape without the in-place op: Parameters only read."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(3, 3)
        self.temp = nn.Parameter(0.07 * torch.ones([]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x) / self.temp.clamp(0.001, 0.5)


class _Mid(nn.Module):
    """Middle module wrapping the MIX-HIC-style temperature module."""

    def __init__(self) -> None:
        super().__init__()
        self.leaf = _MixHicInner()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.leaf(x) * 2


class _Clamper(nn.Module):
    """Clamp the Parameter it is handed; owns no Parameter of its own."""

    def forward(self, x: torch.Tensor, param: nn.Parameter) -> torch.Tensor:
        with torch.no_grad():
            param.clamp_(0.001, 0.5)
        return x + 0


class _RootParamMutatedInChild(nn.Module):
    """A root-owned Parameter mutated inside child ``mut``."""

    def __init__(self) -> None:
        super().__init__()
        self.temp = nn.Parameter(0.07 * torch.ones([]))
        self.mut = _Clamper()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mut(x, self.temp) / self.temp


class _Owner(nn.Module):
    """Owns ``p`` and reads it (a submodule keeps it a cluster, not an atomic box)."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(3, 3)
        self.p = nn.Parameter(torch.full((3,), 2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x) * self.p


class _Mutator(nn.Module):
    """Mutates a Parameter it does not own."""

    def forward(self, x: torch.Tensor, param: nn.Parameter) -> torch.Tensor:
        with torch.no_grad():
            param.mul_(3.0)
        return x + 0


class _SharedMutatedOutsideOwner(nn.Module):
    """``owner.p`` read in ``owner``, mutated in ``mut``, read in ``owner`` again."""

    def __init__(self) -> None:
        super().__init__()
        self.owner = _Owner()
        self.mut = _Mutator()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.owner(x)
        y = self.mut(y, self.owner.p)
        return self.owner(y)


def _trace(model: nn.Module) -> tl.Trace:
    """Capture ``model`` on a fixed input."""

    torch.manual_seed(0)
    return tl.trace(model, torch.randn(2, 3))


def _dot(trace: tl.Trace, tmp_path: Path, name: str, **kwargs: object) -> str:
    """Render ``trace`` and return the DOT source."""

    kwargs.setdefault("layout", "dot")
    return str(
        trace.draw(
            file_format="svg",
            path=str(tmp_path / name),
            **kwargs,
        )
    )


def _node_name(trace: tl.Trace, func_name: str, occurrence: int = 0) -> str:
    """Return the DOT name of the ``occurrence``-th op calling ``func_name``."""

    ops = [op for op in trace._layer_list if op.func_name == func_name]
    return str(ops[occurrence].label).replace(":", "pass")


def _node_line(dot: str, name: str) -> str:
    """Return the DOT declaration line of node ``name``."""

    lines = [line for line in dot.splitlines() if line.strip().startswith(f"{name} [")]
    assert len(lines) == 1, (name, lines)
    return lines[0]


def _edges(dot: str) -> set[tuple[str, str]]:
    """Return every ``tail -> head`` pair in the DOT source."""

    return set(re.findall(r"^\s*(\S+) -> (\S+) \[", dot, flags=re.MULTILINE))


def _cluster_body(dot: str, cluster: str) -> str:
    """Return the text of subgraph ``cluster`` including nested subgraphs."""

    start = dot.index(f"subgraph {cluster} {{")
    depth = 0
    for index in range(start, len(dot)):
        if dot[index] == "{":
            depth += 1
        elif dot[index] == "}":
            depth -= 1
            if depth == 0:
                return dot[start : index + 1]
    raise AssertionError(f"unterminated subgraph {cluster}")


def test_frozen_mutated_parameter_uses_frozen_parameter_grey(tmp_path: Path) -> None:
    """A frozen Parameter's node takes the frozen grey and the frozen shape notation."""

    trace = _trace(_FrozenMutated())
    try:
        dot = _dot(trace, tmp_path, "frozen")
    finally:
        trace.cleanup()

    line = _node_line(dot, "mutatedparam_temp")
    assert f'fillcolor="{FROZEN_PARAMS_BG_COLOR}"' in line
    assert ">[]<" in line


def test_rolled_view_draws_the_parameter_node(tmp_path: Path) -> None:
    """The rolled view draws the same node in the pass-free owner cluster."""

    trace = _trace(_MixHicToy())
    try:
        dot = _dot(trace, tmp_path, "rolled", rolled=True)
        clamp_layer = next(op.layer_label for op in trace._layer_list if op.func_name == "clamp_")
    finally:
        trace.cleanup()

    param_node = '"mutatedparam_inner.temp"'
    assert "shape=cylinder" in _node_line(dot, param_node)
    assert (param_node, clamp_layer) in _edges(dot)
    assert f"{param_node} [" in _cluster_body(dot, "cluster_inner")


def _cluster_names(dot: str) -> list[str]:
    """Return every subgraph cluster name in the DOT source."""

    return re.findall(r"subgraph (\S*cluster\S*) \{", dot)


def test_module_focus_excluding_the_owner_draws_no_node(tmp_path: Path) -> None:
    """A focus on the mutating child of a root-owned Parameter draws the op, not the node."""

    trace = _trace(_RootParamMutatedInChild())
    try:
        full = _dot(trace, tmp_path, "focus_full")
        focused = _dot(trace, tmp_path, "focus_mut", module="mut")
    finally:
        trace.cleanup()

    assert "mutatedparam_temp [" in full
    assert "clamp_" in focused
    assert "mutatedparam_" not in focused


def test_rank_engine_draws_the_parameter_node_and_legend_row(tmp_path: Path) -> None:
    """The rank layout engine positions the node, its edge, and the legend row."""

    trace = _trace(_MixHicToy())
    plain = _trace(_NeverMutated())
    try:
        dot = _dot(trace, tmp_path, "rank", layout="rank", show_legend=True)
        plain_dot = _dot(plain, tmp_path, "rank_plain", layout="rank", show_legend=True)
        clamp = _node_name(trace, "clamp_")
    finally:
        trace.cleanup()
        plain.cleanup()

    param_node = '"mutatedparam_inner.temp"'
    line = _node_line(dot, param_node)
    assert 'shape="cylinder"' in line or "shape=cylinder" in line
    assert "parameter temp" in line
    assert "pos=" in line  # positioned for neato -n like every rank node
    assert f"{param_node} [" in _cluster_body(dot, "cluster_inner_pass1")
    assert (param_node, clamp) in _edges(dot)
    assert ">mutated parameter<" in dot
    assert "mutated parameter" not in plain_dot
    assert "mutatedparam_" not in plain_dot


def test_shared_parameter_mutated_outside_owner_sits_in_owner_cluster(tmp_path: Path) -> None:
    """A Parameter mutated in another module still sits in its owner's box."""

    trace = _trace(_SharedMutatedOutsideOwner())
    try:
        dot = _dot(trace, tmp_path, "shared")
        mul_ = _node_name(trace, "mul_")
        reads = [
            str(op.label).replace(":", "pass")
            for op in trace._layer_list
            if op.func_name == "__mul__"
        ]
    finally:
        trace.cleanup()

    param_node = '"mutatedparam_owner.p"'
    owner_clusters = [name for name in _cluster_names(dot) if "owner" in name]
    assert owner_clusters
    assert f"{param_node} [" in _cluster_body(dot, owner_clusters[0])
    edges = _edges(dot)
    assert (param_node, reads[0]) in edges
    assert (param_node, mul_) in edges
    assert (mul_, reads[1]) in edges
    assert (param_node, reads[1]) not in edges
