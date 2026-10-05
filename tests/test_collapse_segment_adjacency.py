"""Segment boxes may only group siblings that are adjacent in execution order.

The collapse reference promises that a dashed segment box asserts adjacency: its
members are one consecutive run of siblings joined member to member. A segment
that groups non-adjacent siblings (``b1, b2, b3`` around ``t1, t2``) or hides a
connector it does not absorb draws edges out of the box and back in, a cycle the
traced network does not have.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from itertools import pairwise

import pytest
import torch

import torchlens as tl
from tests.test_collapse_optimizer import UniqueWideFanModel
from torchlens.visualization.auto_collapse import analyze_collapse
from torchlens.visualization.collapse_optimizer import _segment_is_legal, select_collapse_plan
from torchlens.visualization.collapse_plan import ChildSegment, RenderContext

_DOT_EDGE = re.compile(r'^\s*"?([^"\s\[]+)"?\s*->\s*"?([^"\s\[;]+)"?', re.MULTILINE)


class TinyBlock(torch.nn.Module):
    """Three-op block; every instance has the same role."""

    def __init__(self, width: int = 8) -> None:
        """Initialize the block."""

        super().__init__()
        self.fc1 = torch.nn.Linear(width, width)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the block."""

        return self.fc2(self.relu(self.fc1(x)))


class Transition(torch.nn.Module):
    """Two-op block of a second role, interleaved between ``TinyBlock`` s."""

    def __init__(self, width: int = 8) -> None:
        """Initialize the transition."""

        super().__init__()
        self.fc = torch.nn.Linear(width, width)
        self.act = torch.nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the transition."""

        return self.act(self.fc(x))


class AlternatingSiblings(torch.nn.Module):
    """DenseNet-shaped sibling order: block, transition, block, ..., block."""

    def __init__(self, width: int = 8) -> None:
        """Initialize the alternating siblings."""

        super().__init__()
        self.stem = torch.nn.Linear(width, width)
        self.b1 = TinyBlock(width)
        self.t1 = Transition(width)
        self.b2 = TinyBlock(width)
        self.t2 = Transition(width)
        self.b3 = TinyBlock(width)
        self.t3 = Transition(width)
        self.b4 = TinyBlock(width)
        self.head = torch.nn.Linear(width, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the siblings in alternating order."""

        x = self.stem(x)
        for name in ("b1", "t1", "b2", "t2", "b3", "t3", "b4"):
            x = getattr(self, name)(x)
        return self.head(x)


class ParentOpConnectors(torch.nn.Module):
    """Same-role siblings joined through parent-owned ops, not directly."""

    def __init__(self, width: int = 8) -> None:
        """Initialize the connector chain."""

        super().__init__()
        self.b0 = TinyBlock(width)
        self.b1 = TinyBlock(width)
        self.b2 = TinyBlock(width)
        self.b3 = TinyBlock(width)
        self.head = torch.nn.Linear(width, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the blocks with a parent-owned op between each pair."""

        x = self.b0(x)
        x = torch.sigmoid(x)
        x = self.b1(x)
        x = torch.sigmoid(x)
        x = self.b2(x)
        x = torch.sigmoid(x)
        x = self.b3(x)
        return self.head(x)


def _trace(model: torch.nn.Module, x: torch.Tensor) -> tl.Trace:
    """Trace a model in eval mode without autograd."""

    model.eval()
    with torch.no_grad():
        return tl.trace(model, x)


def _assert_segments_flow_adjacent(trace: tl.Trace, mode: str | float) -> list[ChildSegment]:
    """Assert every child segment in the ``mode`` plan is a direct sibling chain."""

    analysis = analyze_collapse(trace)
    result = select_collapse_plan(trace, RenderContext(), mode=mode)
    segments = [node for node in result.plan.nodes if isinstance(node, ChildSegment)]
    for segment in segments:
        first = segment.members[0]
        parent = first.rsplit(".", 1)[0] if "." in first else "self"
        graph = analysis.child_flow_graphs[parent]
        order = {address: index for index, address in enumerate(graph.flow_children)}
        positions = [order[member] for member in segment.members]
        assert positions == list(range(positions[0], positions[0] + len(positions))), (
            f"{mode!r} segment {segment.members} groups siblings that are not adjacent in "
            f"execution order {graph.flow_children}"
        )
        edges = set(graph.edges)
        for left, right in pairwise(segment.members):
            assert (left, right) in edges, (
                f"{mode!r} segment {segment.members} joins {left} -> {right} through a "
                "connector the segment does not absorb"
            )
    return segments


def _dot_cycle(source: str) -> list[str] | None:
    """Return one directed cycle in the DOT edge list, or ``None`` if acyclic."""

    successors: dict[str, set[str]] = {}
    for tail, head in _DOT_EDGE.findall(source):
        successors.setdefault(tail.split(":")[0], set()).add(head.split(":")[0])
    state: dict[str, int] = {}
    for root in sorted(successors):
        if state.get(root):
            continue
        path = [root]
        stack = [iter(sorted(successors.get(root, ())))]
        state[root] = 1
        while stack:
            nxt = next(stack[-1], None)
            if nxt is None:
                state[path.pop()] = 2
                stack.pop()
                continue
            if state.get(nxt) == 1:
                return [*path[path.index(nxt) :], nxt]
            if not state.get(nxt):
                state[nxt] = 1
                path.append(nxt)
                stack.append(iter(sorted(successors.get(nxt, ()))))
    return None


def _max_dot_source(trace: tl.Trace, tmp_path: object) -> str:
    """Render ``collapse="max"`` and return the DOT source."""

    return str(
        trace.draw(
            vis_outpath=str(tmp_path / "max"),  # type: ignore[operator]
            vis_save_only=True,
            vis_fileformat="svg",
            vis_node_placement="dot",
            collapse="max",
        )
    )


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("factory", "shape"),
    [
        (AlternatingSiblings, (2, 8)),
        (ParentOpConnectors, (2, 8)),
        (UniqueWideFanModel, (1, 4, 8, 8)),
    ],
    ids=["alternating", "parent_op_connectors", "siblings_around_a_fan"],
)
def test_max_segments_group_only_adjacent_siblings(
    factory: Callable[[], torch.nn.Module],
    shape: tuple[int, ...],
    tmp_path: object,
) -> None:
    """Max-mode segment boxes cover consecutive, directly joined siblings only."""

    trace = _trace(factory(), torch.randn(*shape))
    try:
        for mode in ("max", 1.0, 0.75, "auto"):
            _assert_segments_flow_adjacent(trace, mode)
        source = _max_dot_source(trace, tmp_path)
        assert _dot_cycle(source) is None, (
            f"collapse='max' draws a cycle the feedforward trace lacks: {_dot_cycle(source)}"
        )
    finally:
        trace.cleanup()


@pytest.mark.smoke
def test_segment_legality_rejects_skipped_sibling() -> None:
    """Same-role siblings separated by another sibling are not a legal segment."""

    trace = _trace(AlternatingSiblings(), torch.randn(2, 8))
    try:
        graph = analyze_collapse(trace).child_flow_graphs["self"]
        assert not _segment_is_legal(("b1", "b2"), graph)
        assert not _segment_is_legal(("b1", "b2", "b3"), graph)
        assert _segment_is_legal(("b1", "t1"), graph)
        assert _segment_is_legal(("b1", "t1", "b2"), graph)
    finally:
        trace.cleanup()


@pytest.mark.smoke
def test_segment_legality_rejects_parent_op_connector() -> None:
    """Siblings joined only through a parent-owned op are not a legal segment."""

    trace = _trace(ParentOpConnectors(), torch.randn(2, 8))
    try:
        graph = analyze_collapse(trace).child_flow_graphs["self"]
        assert not _segment_is_legal(("b0", "b1"), graph)
        assert not _segment_is_legal(("b0", "b1", "b2", "b3"), graph)
    finally:
        trace.cleanup()


@pytest.mark.heavy
def test_densenet121_max_segments_respect_block_transition_order(tmp_path: object) -> None:
    """DenseNet-121 max never groups denseblocks across their transitions."""

    tvm = pytest.importorskip("torchvision.models")
    trace = _trace(tvm.densenet121(weights=None), torch.randn(1, 3, 64, 64))
    try:
        segments = _assert_segments_flow_adjacent(trace, "max")
        for segment in segments:
            kinds = {member.rsplit(".", 1)[-1].rstrip("0123456789") for member in segment.members}
            assert kinds != {"denseblock"} and kinds != {"transition"}, (
                f"segment {segment.members} groups one role across interleaved siblings"
            )
        source = _max_dot_source(trace, tmp_path)
        assert _dot_cycle(source) is None, (
            f"collapse='max' draws a cycle the feedforward trace lacks: {_dot_cycle(source)}"
        )
    finally:
        trace.cleanup()
