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
from torchlens.visualization._render_common import (
    FROZEN_PARAMS_BG_COLOR,
    TRAINABLE_PARAMS_BG_COLOR,
)

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


class _MutatedTwice(nn.Module):
    """Mutate a Parameter twice, then read it."""

    def __init__(self) -> None:
        super().__init__()
        self.temp = nn.Parameter(torch.ones([]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.temp.mul_(2.0)
            self.temp.add_(1.0)
        return x * self.temp


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


class _ParamReadByInplaceOp(nn.Module):
    """In-place ops that READ Parameters without mutating them, plus a receiver."""

    def __init__(self) -> None:
        super().__init__()
        self.temp = nn.Parameter(torch.ones(3))
        self.w = nn.Parameter(torch.full((3,), 0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.clone()
        with torch.no_grad():
            y.add_(self.w)
            self.temp.add_(self.w)
        return y * self.temp


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


class _ThreeLevel(nn.Module):
    """Root -> ``mid`` -> ``mid.leaf``, which owns and clamps ``temp``."""

    def __init__(self) -> None:
        super().__init__()
        self.mid = _Mid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mid(x) + 1


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


class _HoldsB(nn.Module):
    """Owns ``b`` and clamps it."""

    def __init__(self) -> None:
        super().__init__()
        self.b = nn.Parameter(torch.ones([]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.b.clamp_(0.0, 2.0)
        return x * self.b


class _DotUnderscoreCollision(nn.Module):
    """``a.b`` and root ``a_b``: addresses equal after ``.`` -> ``_`` folding."""

    def __init__(self) -> None:
        super().__init__()
        self.a = _HoldsB()
        self.a_b = nn.Parameter(torch.ones([]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.a_b.clamp_(0.0, 3.0)
        return self.a(x) * self.a_b


class _Owner(nn.Module):
    """Owns ``p`` and reads it."""

    def __init__(self) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.full((3,), 2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.p


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

    kwargs.setdefault("vis_node_placement", "dot")
    return str(
        trace.draw(
            vis_save_only=True,
            vis_fileformat="svg",
            vis_outpath=str(tmp_path / name),
            **kwargs,
        )
    )


def _node_name(trace: tl.Trace, func_name: str, occurrence: int = 0) -> str:
    """Return the DOT name of the ``occurrence``-th op calling ``func_name``."""

    ops = [op for op in trace.layer_list if op.func_name == func_name]
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


def test_mixhic_toy_draws_parameter_cylinder_in_owner_cluster(tmp_path: Path) -> None:
    """The clamped Parameter is a grey cylinder inside ``@inner`` feeding the clamp op."""

    trace = _trace(_MixHicToy())
    try:
        dot = _dot(trace, tmp_path, "mixhic")
        clamp = _node_name(trace, "clamp_")
        truediv = _node_name(trace, "__truediv__")
    finally:
        trace.cleanup()

    param_node = '"mutatedparam_inner.temp"'
    line = _node_line(dot, param_node)
    assert "shape=cylinder" in line
    assert f'fillcolor="{TRAINABLE_PARAMS_BG_COLOR}"' in line
    assert "<B>parameter temp</B>" in line
    assert ">()<" in line  # the scalar's shape row
    assert f"{param_node} [" in _cluster_body(dot, "cluster_inner_pass1")

    edges = _edges(dot)
    assert (param_node, clamp) in edges
    # The later read takes its edge from the in-place op, never the Parameter node.
    assert (clamp, truediv) in edges
    assert (param_node, truediv) not in edges
    assert sum(1 for tail, _ in edges if tail == param_node) == 1


def test_parameter_mutated_twice_chains_through_both_ops(tmp_path: Path) -> None:
    """The node feeds the first mutation; the second chains on it; the read on the last."""

    trace = _trace(_MutatedTwice())
    try:
        dot = _dot(trace, tmp_path, "twice")
        first = _node_name(trace, "mul_")
        second = _node_name(trace, "add_")
        read = _node_name(trace, "__mul__")
    finally:
        trace.cleanup()

    param_node = "mutatedparam_temp"
    assert "shape=cylinder" in _node_line(dot, param_node)
    edges = _edges(dot)
    assert (param_node, first) in edges
    assert (first, second) in edges
    assert (second, read) in edges
    assert {head for tail, head in edges if tail == param_node} == {first}
    # A root-owned Parameter sits at top level, outside every cluster.
    assert "subgraph cluster" not in dot


def test_reads_before_and_after_mutation_bind_to_the_right_value(tmp_path: Path) -> None:
    """A read before the mutation hangs off the Parameter node; a read after, off the op."""

    trace = _trace(_ReadBeforeAndAfter())
    try:
        dot = _dot(trace, tmp_path, "before_after")
        before = _node_name(trace, "__mul__", 0)
        mutation = _node_name(trace, "mul_")
        after = _node_name(trace, "__mul__", 1)
    finally:
        trace.cleanup()

    param_node = "mutatedparam_scale"
    edges = _edges(dot)
    assert {head for tail, head in edges if tail == param_node} == {before, mutation}
    assert (mutation, after) in edges
    assert (param_node, after) not in edges


def test_legend_lists_mutated_parameter_only_when_drawn(tmp_path: Path) -> None:
    """The legend row follows the buffer row's form and is gated on a drawn node."""

    mutated = _trace(_MixHicToy())
    plain = _trace(_NeverMutated())
    try:
        mutated_dot = _dot(mutated, tmp_path, "legend_mutated", show_legend=True)
        plain_dot = _dot(plain, tmp_path, "legend_plain", show_legend=True)
    finally:
        mutated.cleanup()
        plain.cleanup()

    assert "mutated parameter (cylinder)" in mutated_dot
    assert "buffer (cylinder)" in mutated_dot
    assert "mutated parameter" not in plain_dot
    assert "buffer (cylinder)" in plain_dot


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
        dot = _dot(trace, tmp_path, "rolled", vis_mode="rolled")
        clamp_layer = next(op.layer_label for op in trace.layer_list if op.func_name == "clamp_")
    finally:
        trace.cleanup()

    param_node = '"mutatedparam_inner.temp"'
    assert "shape=cylinder" in _node_line(dot, param_node)
    assert (param_node, clamp_layer) in _edges(dot)
    assert f"{param_node} [" in _cluster_body(dot, "cluster_inner")


def test_only_the_receiver_parameter_gets_a_node(tmp_path: Path) -> None:
    """A Parameter read by an in-place op is not mutated; only the receiver is."""

    from torchlens.visualization import _mutated_params

    trace = _trace(_ParamReadByInplaceOp())
    try:
        sources = _mutated_params.find_mutated_parameter_sources(trace)
        dot = _dot(trace, tmp_path, "receiver")
    finally:
        trace.cleanup()

    assert [source.param.address for source in sources] == ["temp"]
    assert "mutatedparam_temp [" in dot
    assert "mutatedparam_w" not in dot


def test_never_mutated_parameters_get_no_node_and_unchanged_dot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a mutation there is no node, and the DOT equals the feature-free render."""

    from torchlens.visualization import _mutated_params

    trace = _trace(_NeverMutated())
    try:
        assert _mutated_params.find_mutated_parameter_sources(trace) == ()
        dot = _dot(trace, tmp_path, "plain", show_legend=True)
        monkeypatch.setattr(_mutated_params, "find_mutated_parameter_sources", lambda _trace: ())
        baseline = _dot(trace, tmp_path, "plain_baseline", show_legend=True)
    finally:
        trace.cleanup()

    assert "mutatedparam_" not in dot
    assert "shape=cylinder" not in dot
    assert dot == baseline


def _cluster_names(dot: str) -> list[str]:
    """Return every subgraph cluster name in the DOT source."""

    return re.findall(r"subgraph (\S*cluster\S*) \{", dot)


@pytest.mark.parametrize("focus", ["mid.leaf", "mid"])
def test_module_focus_draws_the_parameter_node(tmp_path: Path, focus: str) -> None:
    """A focus containing the owner draws the node inside the owner cluster."""

    trace = _trace(_ThreeLevel())
    try:
        dot = _dot(trace, tmp_path, f"focus_{focus}", module=focus)
        clamp = next(op for op in trace.layer_list if op.func_name == "clamp_")
        truediv = next(op for op in trace.layer_list if op.func_name == "__truediv__")
    finally:
        trace.cleanup()

    param_node = '"mutatedparam_mid.leaf.temp"'
    line = _node_line(dot, param_node)
    assert "shape=cylinder" in line
    assert "<B>parameter temp</B>" in line
    edges = _edges(dot)
    clamp_name = next(head for tail, head in edges if tail == param_node)
    assert clamp_name in {clamp.layer_label, str(clamp.label).replace(":", "pass")}
    assert {head for tail, head in edges if tail == param_node} == {clamp_name}
    truediv_names = {truediv.layer_label, str(truediv.label).replace(":", "pass")}
    assert any(tail == clamp_name and head in truediv_names for tail, head in edges)
    leaf_cluster = next(name for name in _cluster_names(dot) if "leaf" in name)
    assert f"{param_node} [" in _cluster_body(dot, leaf_cluster)


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


def test_dot_and_underscore_addresses_get_distinct_nodes(tmp_path: Path) -> None:
    """``a.b`` and ``a_b`` keep separate node declarations and separate edges."""

    from torchlens.visualization._mutated_params import mutated_parameter_node_name

    trace = _trace(_DotUnderscoreCollision())
    try:
        dot = _dot(trace, tmp_path, "collision")
        clamps = [
            str(op.label).replace(":", "pass")
            for op in trace.layer_list
            if op.func_name == "clamp_"
        ]
        receivers = [
            op._param_logs[0].address for op in trace.layer_list if op.func_name == "clamp_"
        ]
    finally:
        trace.cleanup()

    assert mutated_parameter_node_name("a.b") != mutated_parameter_node_name("a_b")
    dotted, flat = '"mutatedparam_a.b"', "mutatedparam_a_b"
    assert "<B>parameter b</B>" in _node_line(dot, dotted)
    assert "<B>parameter a_b</B>" in _node_line(dot, flat)
    clamp_of = dict(zip(receivers, clamps))
    edges = _edges(dot)
    assert {head for tail, head in edges if tail == dotted} == {clamp_of["a.b"]}
    assert {head for tail, head in edges if tail == flat} == {clamp_of["a_b"]}


def test_node_names_are_injective_and_port_free() -> None:
    """Distinct addresses never share a node name, and no name carries a DOT port colon."""

    from torchlens.visualization._mutated_params import mutated_parameter_node_name

    addresses = ["a.b", "a_b", "a%2Eb", "a:b", "a-b", 'a"b', "a b", "a.b.c", "a_b.c"]
    names = [mutated_parameter_node_name(address) for address in addresses]
    assert len(set(names)) == len(addresses)
    assert all(":" not in name and '"' not in name and " " not in name for name in names)


def test_rank_engine_draws_the_parameter_node_and_legend_row(tmp_path: Path) -> None:
    """The rank layout engine positions the node, its edge, and the legend row."""

    trace = _trace(_MixHicToy())
    plain = _trace(_NeverMutated())
    try:
        dot = _dot(trace, tmp_path, "rank", vis_node_placement="rank", show_legend=True)
        plain_dot = _dot(plain, tmp_path, "rank_plain", vis_node_placement="rank", show_legend=True)
        clamp = _node_name(trace, "clamp_")
    finally:
        trace.cleanup()
        plain.cleanup()

    param_node = '"mutatedparam_inner.temp"'
    line = _node_line(dot, param_node)
    assert 'shape="cylinder"' in line or "shape=cylinder" in line
    assert "parameter temp" in line
    assert "pos=" in line  # positioned for neato -n like every rank node
    assert (param_node, clamp) in _edges(dot)
    assert "mutated parameter (cylinder)" in dot
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
            for op in trace.layer_list
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
