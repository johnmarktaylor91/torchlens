"""Regression tests for loop-rolling and module-rolling reconciliation."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._visualization._pass_label_placement import place_pass_labels
from torchlens._visualization._render_nodes import compute_default_node_lines

# Session-private output root (conftest exports it before collection); the
# repo-local fallback raced across parallel worktree lanes (b2p2 R77).
OUTPUT_DIR = (
    Path(os.environ.get("TORCHLENS_TEST_OUTPUTS_DIR", Path(__file__).parent / "generated_outputs"))
    / "visualizations"
    / "loop_module_rolling"
)


class ReusedReluLoop(nn.Module):
    """Single-op module reused in a loop."""

    def __init__(self) -> None:
        """Initialize the reused ReLU."""

        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the same ReLU module four times."""

        for _ in range(4):
            x = self.relu(x)
        return x


class InsideOutsideRelu(nn.Module):
    """ReLU module fed in a continuous chain (outside call feeds the loop)."""

    def __init__(self) -> None:
        """Initialize the reused ReLU."""

        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Chain ReLU once outside the loop into four calls inside it."""

        y = self.relu(x)
        for _ in range(4):
            y = self.relu(y)
        return y


class InsideOutsideReluSeparable(nn.Module):
    """Atomic ReLU module used at a separable outside site plus a loop."""

    def __init__(self) -> None:
        """Initialize the reused ReLU."""

        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Call ReLU once on its own, then three times on an independent stream."""

        outside = self.relu(x)
        inside = x + 1
        for _ in range(3):
            inside = self.relu(inside)
        return outside + inside


class ParallelFanout(nn.Module):
    """Shared projection used as parallel fan-out into stack."""

    def __init__(self) -> None:
        """Initialize the shared projection."""

        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Stack four independent calls to the same projection."""

        return torch.stack([self.proj(x) for _ in range(4)])


class TwoLinearBlock(nn.Module):
    """Block that calls one shared layer at two structural sites."""

    def __init__(self) -> None:
        """Initialize the shared linear layer."""

        super().__init__()
        self.lin = nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Call the same linear layer twice."""

        return self.lin(self.lin(x))


class WrappedTwoSiteLoop(nn.Module):
    """Module-wrapped two-site recurrent layer."""

    def __init__(self) -> None:
        """Initialize the reusable block."""

        super().__init__()
        self.block = TwoLinearBlock()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Call the block three times."""

        for _ in range(3):
            x = self.block(x)
        return x


class RNNCellLoop(nn.Module):
    """RNN-cell-style loop with direct hidden-state recurrence."""

    def __init__(self) -> None:
        """Initialize the recurrent cell."""

        super().__init__()
        self.cell = nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the same cell to the hidden state four times."""

        h = x
        for _ in range(4):
            h = self.cell(h)
        return h


class ActivationBlock(nn.Module):
    """Two-op block whose output feeds the next block call."""

    def __init__(self) -> None:
        """Initialize the linear and activation layers."""

        super().__init__()
        self.lin = nn.Linear(4, 4, bias=False)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear layer and activation."""

        return self.act(self.lin(x))


class MlpBlock(nn.Module):
    """Three-op feed-forward block."""

    def __init__(self) -> None:
        """Initialize the block layers."""

        super().__init__()
        self.lin1 = nn.Linear(4, 4, bias=False)
        self.act = nn.ReLU()
        self.lin2 = nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Linear -> ReLU -> Linear."""

        return self.lin2(self.act(self.lin1(x)))


class CollapsedBlockRecurrence(nn.Module):
    """Repeated block whose collapsed module box carries recurrence."""

    def __init__(self) -> None:
        """Initialize the recurrent block."""

        super().__init__()
        self.block = ActivationBlock()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Call the same block three times in a loop."""

        for _ in range(3):
            x = self.block(x)
        return x


class InsideOutsideBlock(nn.Module):
    """Multi-op block called once before a loop and repeatedly inside it."""

    def __init__(self) -> None:
        """Initialize the reused block."""

        super().__init__()
        self.block = ActivationBlock()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Call the block once outside the loop, then four times inside it."""

        y = self.block(x)
        for _ in range(4):
            y = self.block(y)
        return y


class BranchingLoopBody(nn.Module):
    """Recurrent cell whose last pass takes an extra branch op."""

    def __init__(self) -> None:
        """Initialize the hidden projection."""

        super().__init__()
        self.lin = nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a tanh cell three times and double the state on the last pass."""

        h = x
        for step in range(3):
            h = torch.tanh(self.lin(h))
            if step == 2:
                h = h * 2
        return h


class DeepLoopBody(nn.Module):
    """Loop body with three operations."""

    def __init__(self) -> None:
        """Initialize the body layers."""

        super().__init__()
        self.fc = nn.Linear(4, 4, bias=False)
        self.act = nn.ReLU()
        self.norm = nn.LayerNorm(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a three-op body three times."""

        for _ in range(3):
            x = self.norm(self.act(self.fc(x)))
        return x


class TanhRNNCellLoop(nn.Module):
    """Small recurrent cell with a multi-op loop body."""

    def __init__(self) -> None:
        """Initialize the hidden projection."""

        super().__init__()
        self.lin = nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a recurrent tanh cell four times."""

        h = x
        for _ in range(4):
            h = torch.tanh(self.lin(h))
        return h


class RepeatedBlockStack(nn.Module):
    """Canonical repeated block stack."""

    def __init__(self) -> None:
        """Initialize the repeated MLP block."""

        super().__init__()
        self.block = MlpBlock()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Call the same three-op block four times."""

        for _ in range(4):
            x = self.block(x)
        return x


class SharedTwoSiteBlock(nn.Module):
    """Block that applies one shared layer to two recurrent streams."""

    def __init__(self) -> None:
        """Initialize the shared linear layer."""

        super().__init__()
        self.lin = nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the shared layer independently to two streams."""

        return self.lin(x), self.lin(y)


class SharedTwoSiteRecurrences(nn.Module):
    """Two interleaved recurrent chains through the same internal layer."""

    def __init__(self) -> None:
        """Initialize the shared two-site recurrent block."""

        super().__init__()
        self.block = SharedTwoSiteBlock()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run three block calls carrying two separate recurrent streams."""

        y = x + 1
        for _ in range(3):
            x, y = self.block(x, y)
        return x + y


class TwoDistinctLoops(nn.Module):
    """Shared module used by two distinct loops."""

    def __init__(self) -> None:
        """Initialize the shared module."""

        super().__init__()
        self.shared = nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run two separate loops that reuse the same module."""

        first = x
        for _ in range(2):
            first = self.shared(first)
        second = x + 1
        for _ in range(3):
            second = self.shared(second)
        return first + second


class BufferRewriteLoops(nn.Module):
    """Buffer read and rewritten on every pass of one loop body."""

    def __init__(self) -> None:
        """Initialize the state buffer."""

        super().__init__()
        self.register_buffer("state", torch.zeros(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Read and rewrite the buffer five times from one loop body.

        One loop, so the read and the rewrite are each one layer with five passes
        (two loops would be two places in the code, so two layers each).
        """

        y = x
        for _ in range(5):
            y = y + self.state
            self.state = self.state + 1
        return y


class MixedDependency(nn.Module):
    """Shared projection with mixed fan-out and chain dependency."""

    def __init__(self) -> None:
        """Initialize the shared projection."""

        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Call two independent projections, then one dependent projection."""

        a = self.proj(x)
        b = self.proj(x)
        return self.proj(a + b)


class InsideOutsideLoop(nn.Module):
    """Shared layer called once outside and then inside a loop."""

    def __init__(self) -> None:
        """Initialize the shared layer."""

        super().__init__()
        self.lin = nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Call the layer outside the loop and then three times inside it."""

        outside = self.lin(x)
        inside = x + 1
        for _ in range(3):
            inside = self.lin(inside)
        return outside + inside


class ParallelSiblingsLoop(nn.Module):
    """Shared layer used at two sibling sites in each loop body."""

    def __init__(self) -> None:
        """Initialize the shared linear layer."""

        super().__init__()
        self.lin = nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run four loop bodies with two parallel sibling calls each."""

        for _ in range(4):
            a = self.lin(x)
            b = self.lin(x)
            x = a + b
        return x


class NestedLoopBlock(nn.Module):
    """Nested loop reusing the same projection."""

    def __init__(self) -> None:
        """Initialize the reused projection."""

        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run two outer iterations with two inner projection calls each."""

        for _ in range(2):
            residual = x
            for _ in range(2):
                x = self.proj(x)
            x = x + residual
        return x


class VariableLinearBlock(nn.Module):
    """Block that applies a shared linear layer ``n`` times."""

    def __init__(self) -> None:
        """Initialize the shared linear layer."""

        super().__init__()
        self.lin = nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor, n: int) -> torch.Tensor:
        """Apply the linear layer ``n`` times."""

        for _ in range(n):
            x = self.lin(x)
        return x


class RaggedBlockLoop(nn.Module):
    """Loop whose block expands to a ragged number of linear passes."""

    def __init__(self) -> None:
        """Initialize the variable block."""

        super().__init__()
        self.block = VariableLinearBlock()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Call the block with a 2, 2, 3 pass pattern."""

        for n in (2, 2, 3):
            x = self.block(x, n)
        return x


class IntegralRaggedBlocks(nn.Module):
    """Two block calls with non-rectangular 3 and 1 pass bodies."""

    def __init__(self) -> None:
        """Initialize the variable block."""

        super().__init__()
        self.block = VariableLinearBlock()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Call the block with uneven body sizes."""

        x = self.block(x, 3)
        return self.block(x, 1)


class ModeDependentBlock(nn.Module):
    """Block whose two calls have equal cardinality but different structure."""

    def __init__(self) -> None:
        """Initialize the shared linear layer."""

        super().__init__()
        self.lin = nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """Run either a parallel or chained two-call body."""

        if mode == "parallel":
            return self.lin(x) + self.lin(x)
        return self.lin(self.lin(x))


class NonUniformBodyCalls(nn.Module):
    """Two equal-sized block calls with different separation signatures."""

    def __init__(self) -> None:
        """Initialize the mode-dependent block."""

        super().__init__()
        self.block = ModeDependentBlock()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a parallel body followed by a chain body."""

        x = self.block(x, "parallel")
        return self.block(x, "chain")


class MixedWithSites(nn.Module):
    """Mixed dependency that still has separable structural sites."""

    def __init__(self) -> None:
        """Initialize the shared linear layer."""

        super().__init__()
        self.lin = nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Create a chain plus an independent sibling before a join."""

        a = self.lin(x)
        b = self.lin(a)
        c = self.lin(x)
        return b + c


class StackFanoutForDeepArgMutation(nn.Module):
    """Stack fan-out used to mutate a consumer path beyond renderer support."""

    def __init__(self) -> None:
        """Initialize the shared linear layer."""

        super().__init__()
        self.lin = nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Stack two calls so both feed one consumer op."""

        return torch.stack([self.lin(x), self.lin(x)])


class BareFunctionalTanhChain(nn.Module):
    """Two adjacent bare functional tanh calls with no reused module or buffer."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ``torch.tanh`` twice in a straight-line chain."""

        return torch.tanh(torch.tanh(x))


def test_bare_functional_op_chain_is_not_a_loop() -> None:
    """Two adjacent bare functional ops are a chain, not a param-free two-pass loop.

    A parameter-free adjacency merge with a single-op body and no reused persistent identity
    (no submodule call, no buffer) is the historical false positive. Each ``tanh`` must stay
    its own single-pass layer instead of collapsing into a spurious ``tanh (x2)`` loop.
    """

    trace = _trace(BareFunctionalTanhChain())
    tanh_layers = [label for label in trace.layer_labels if label.startswith("tanh")]
    assert len(tanh_layers) == 2
    for label in tanh_layers:
        assert trace.layers[label].num_passes == 1


def test_reused_single_op_module_stays_a_loop() -> None:
    """A reused single-op module is still grouped as recurrence despite the >=2-op guard.

    Its ops are recurrence-anchored (module-bound), so the parameter-free body-size guard
    does not apply -- the four ``self.relu`` calls remain one four-pass layer.
    """

    trace = _trace(ReusedReluLoop())
    relu_layers = [label for label in trace.layer_labels if label.startswith("relu")]
    assert len(relu_layers) == 1
    assert trace.layers[relu_layers[0]].num_passes == 4


def _trace(model: nn.Module) -> tl.Trace:
    """Trace a demo model deterministically."""

    torch.manual_seed(0)
    return tl.trace(model, torch.randn(1, 4), save="none")


def _first_layer(trace: tl.Trace, layer_type: str) -> tl.Layer:
    """Return the first rendered layer of a given type."""

    for layer in trace._layer_logs.values():
        if layer.layer_type == layer_type:
            return layer
    raise AssertionError(f"No layer of type {layer_type!r}.")


def _normalized_title(trace: tl.Trace, layer_type: str, display_name: str) -> str:
    """Return a display title normalized to the proposal's short op name."""

    layer = _first_layer(trace, layer_type)
    title = compute_default_node_lines(layer, vis_mode="rolled")[0]
    return re.sub(rf"^{layer_type}_\d+_\d+", display_name, title)


def _render_dot(trace: tl.Trace, tmp_path: Path, name: str, **kwargs: object) -> str:
    """Render DOT for assertions."""

    return _render_source(trace, tmp_path, name, "dot", **kwargs)


def _render_source(
    trace: tl.Trace,
    tmp_path: Path,
    name: str,
    file_format: str,
    **kwargs: object,
) -> str:
    """Render a graph format and return the emitted Graphviz source."""

    return trace.draw(
        rolled=True,
        file_format=file_format,
        path=str(tmp_path / name),
        **kwargs,
    ).dot


def _dot_self_edge_attrs(dot: str) -> list[str]:
    """Return DOT attributes for explicit ``name -> name`` self-edges."""

    return [
        match.group("attrs")
        for match in re.finditer(
            r"(?m)^\s*(?P<name>[A-Za-z0-9_.]+) -> (?P=name) \[(?P<attrs>[^\]]*)\]",
            dot,
        )
    ]


def _dot_edge_attrs(dot: str, tail: str, head: str) -> list[str]:
    """Return DOT attributes for explicit ``tail -> head`` edges."""

    return [
        match.group("attrs")
        for match in re.finditer(
            rf"(?m)^\s*{re.escape(tail)} -> {re.escape(head)} \[(?P<attrs>[^\]]*)\]",
            dot,
        )
    ]


def _html_label_lines(attrs: str, kind: str) -> list[str] | None:
    """Return the visible text lines of an HTML ``kind=<...>`` edge label.

    ``kind`` is ``"label"``, ``"xlabel"``, ``"headlabel"``, or ``"taillabel"``.  Returns None
    when the edge carries no such HTML label.  Tags (including ``<BR/>`` line
    breaks) are treated as line separators, so a two-line combined label yields
    two entries.
    """

    match = re.search(rf"(?<![a-z]){kind}=<<TABLE[^>]*>(?P<body>.*?)</TABLE>>", attrs)
    if match is None:
        return None
    text = re.sub(r"<[^>]+>", "\n", match.group("body"))
    return [line.strip() for line in text.split("\n") if line.strip()]


def _dot_node_shapes(dot: str) -> dict[str, str]:
    """Map each DOT node title (``<B>...</B>``) to its graphviz ``shape``."""

    shapes: dict[str, str] = {}
    for line in dot.splitlines():
        if "shape=" not in line or "->" in line:
            continue
        shape_match = re.search(r"shape=(\w+)", line)
        title_match = re.search(r"<B>(.*?)</B>", line)
        if shape_match and title_match:
            shapes[title_match.group(1)] = shape_match.group(1)
    return shapes


def test_reused_single_op_module_count_is_depth_invariant(tmp_path: Path) -> None:
    """A reused single-op module keeps the honest count at shallow and deep depth."""

    trace = _trace(ReusedReluLoop())
    assert _normalized_title(trace, "relu", "relu") == "relu (x4)"

    shallow_dot = _render_dot(trace, tmp_path, "relu_shallow", depth=1)
    deep_dot = _render_dot(trace, tmp_path, "relu_deep", depth=1000)

    assert "relu_1_1 (x4)" in shallow_dot
    assert "relu_1_1 (x4)" in deep_dot


def test_multi_op_loop_body_keeps_plain_counts(tmp_path: Path) -> None:
    """A Linear -> ReLU loop body shows honest counts without site suffixes."""

    trace = _trace(CollapsedBlockRecurrence())
    assert _normalized_title(trace, "linear", "lin") == "lin (x3)"
    assert _normalized_title(trace, "relu", "relu") == "relu (x3)"

    dot = _render_dot(trace, tmp_path, "multiop_loop", depth=1000)
    assert "sites" not in dot
    assert "×" not in dot
    assert 'label="↻"' not in dot


def test_colon_call_ranges_disambiguate_split_regions(tmp_path: Path) -> None:
    """Reused modules in split regions use compact colon ranges."""

    distinct_trace = _trace(TwoDistinctLoops())
    assert _normalized_title(distinct_trace, "linear", "lin") == "lin:1-2,3-5"
    distinct_dot = _render_dot(distinct_trace, tmp_path, "two_loops", depth=1)
    assert "@shared:1-2,3-5" in distinct_dot
    assert "calls" not in distinct_dot

    inside_outside_trace = _trace(InsideOutsideLoop())
    assert _normalized_title(inside_outside_trace, "linear", "lin") == "lin:1,2-4"


def test_continuous_relu_chain_is_one_atomic_rectangle(tmp_path: Path) -> None:
    """A continuous relu chain from one loop line rolls to one atomic-module rectangle.

    ``ReusedReluLoop`` feeds each relu call into the next from one loop line, so all
    four calls form one recurrent chain at a single site -- it rolls to one node,
    marked as the atomic ``@relu`` module (a rectangle), with a plain ``(x4)`` count
    and no split range.
    """

    trace = _trace(ReusedReluLoop())
    assert _normalized_title(trace, "relu", "relu") == "relu (x4)"

    dot = _render_dot(trace, tmp_path, "relu_loop_chain", depth=1)
    assert "relu_1_1 (x4)" in dot
    # Atomic single-op module is marked as a module and rendered as a rectangle.
    assert "@relu" in dot
    assert _dot_node_shapes(dot).get("relu_1_1 (x4)") == "box"
    # Single contiguous site -> no split call range on the marking.
    assert "@relu:" not in dot


def test_inside_outside_relu_chain_splits_at_the_outside_line(tmp_path: Path) -> None:
    """A stateless relu called once before its loop is two layers, two atomic rectangles.

    ``InsideOutsideRelu`` chains the outside call into four loop calls. ``nn.ReLU``
    owns no state, so its calls group by call site (``callsite_v2``): the outside
    line is one layer and the loop line a 4-pass layer, and the markings name the
    calls each rectangle holds (``@relu:1`` and ``@relu:2-5``).
    """

    trace = _trace(InsideOutsideRelu())
    dot = _render_dot(trace, tmp_path, "inside_outside_relu", depth=1)
    shapes = _dot_node_shapes(dot)
    assert shapes.get("relu_1_1") == "box"
    assert shapes.get("relu_2_2") == "box"
    assert "@relu:1" in dot
    assert "@relu:2-5" in dot
    assert "relu_1_1 (x" not in dot


def test_separable_atomic_module_marks_distinct_calls(tmp_path: Path) -> None:
    """An atomic module reused at separable outside+inside sites labels its calls.

    Unlike the continuous chain, ``InsideOutsideReluSeparable`` uses the relu at an
    independent outside site plus a loop, so it renders as two atomic rectangles
    whose markings disambiguate the calls (``@relu:1`` and ``@relu:2-4``). The
    rectangles are identical at shallow and deep call depth -- an atomic module is
    already maximally collapsed.
    """

    # ``nn.ReLU`` owns no state, so its calls group by call site (``callsite_v2``):
    # the outside line is one layer and the loop line a 3-pass layer.
    trace = _trace(InsideOutsideReluSeparable())
    shallow_dot = _render_dot(trace, tmp_path, "relu_sep_shallow", depth=1)
    deep_dot = _render_dot(trace, tmp_path, "relu_sep_deep", depth=1000)

    for dot in (shallow_dot, deep_dot):
        shapes = _dot_node_shapes(dot)
        assert shapes.get("relu_1_1") == "box"
        assert shapes.get("relu_2_3") == "box"
        assert "@relu:1" in dot
        assert "@relu:2-4" in dot
        # The range lives on the marking, never duplicated as a title count.
        assert "relu_1_1 (x" not in dot
        # Atomic modules never collapse into a box3d module summary.
        assert "box3d" not in dot


def test_inside_outside_block_module_uses_plain_count(tmp_path: Path) -> None:
    """A collapsed multi-op block used outside and inside one loop shows plain counts."""

    trace = _trace(InsideOutsideBlock())
    expanded_dot = _render_dot(trace, tmp_path, "inside_outside_block_expanded", depth=1000)
    collapsed_dot = _render_dot(trace, tmp_path, "inside_outside_block_collapsed", depth=1)

    assert "linear_1_1 (x5)" in expanded_dot
    assert "relu_1_2 (x5)" in expanded_dot
    assert "@block (x5)" in collapsed_dot
    assert "@block:1,2-5" not in collapsed_dot


def test_clean_reused_block_keeps_plain_counts(tmp_path: Path) -> None:
    """A reused multi-op block in one loop keeps plain module and op counts."""

    trace = _trace(CollapsedBlockRecurrence())
    expanded_dot = _render_dot(trace, tmp_path, "block_expanded", depth=1000)
    collapsed_dot = _render_dot(trace, tmp_path, "block_collapsed", depth=1)

    assert "linear_1_1 (x3)" in expanded_dot
    assert "relu_1_2 (x3)" in expanded_dot
    assert "@block (x3)" in collapsed_dot
    assert 'label="↻"' not in expanded_dot
    assert 'label="↻"' not in collapsed_dot


def test_buffer_versions_use_colon_ranges(tmp_path: Path) -> None:
    """Rolled buffer nodes use the module-address colon range form."""

    trace = _trace(BufferRewriteLoops())
    dot = _render_dot(trace, tmp_path, "buffers", show_buffers="always")
    assert "@state:1-6" in dot
    assert "@state v1-6" not in dot
    assert "buffer_versions" not in dot


def test_recurrent_self_loop_merges_in_out_into_one_label_placed_at_its_node(
    tmp_path: Path,
) -> None:
    """A rolled recurrence self-loop merges In/Out into one label beside its node.

    Both ends of a self-loop are one node, so its pass ranges stay one two-line HTML
    label, ``In`` above ``Out`` for the default bottom-up layout.  It is the loop's
    tail label, so the post-layout chooser places it beside the node clear of the
    loop and its neighbours, as it places every rolled pass label; Graphviz's own
    midpoint spot for a self-loop label crossed nearby edges and nodes in sideways
    drawings.  Parallel fan-out has no self-edge at all.
    """

    recurrent_trace = _trace(ReusedReluLoop())
    drawn = _render_dot(recurrent_trace, tmp_path, "relu_self_edge", depth=1000)
    assert "labelangle=" not in drawn  # placement goes only into the image's source
    recurrent_dot = place_pass_labels(drawn, engine="dot", timeout=60)
    recurrent_self_edges = _dot_self_edge_attrs(recurrent_dot)
    assert len(recurrent_self_edges) == 1
    recurrent_attrs = recurrent_self_edges[0]
    assert "headlabel" not in recurrent_attrs
    assert _html_label_lines(recurrent_attrs, "label") is None
    assert _html_label_lines(recurrent_attrs, "taillabel") == ["In 2-4", "Out 1-3"]
    assert "labelangle=" in recurrent_attrs and "minlen" not in recurrent_attrs
    assert "↻" not in recurrent_attrs

    fanout_trace = _trace(ParallelFanout())
    fanout_dot = _render_dot(fanout_trace, tmp_path, "parallel_fanout", depth=1000)
    assert _dot_self_edge_attrs(fanout_dot) == []


def test_self_loop_label_order_flips_with_layout_direction(tmp_path: Path) -> None:
    """The self-loop's In/Out order follows flow: In above Out only bottom-up."""

    trace = _trace(ReusedReluLoop())
    topdown_dot = _render_dot(trace, tmp_path, "relu_topdown", depth=1000, direction="topdown")
    self_edges = _dot_self_edge_attrs(topdown_dot)
    assert len(self_edges) == 1
    # Top-down flow points down, so Out (tail) is placed above In (head).
    assert _html_label_lines(self_edges[0], "taillabel") == ["Out 1-3", "In 2-4"]


def test_recurrence_back_edge_puts_out_at_its_tail_and_in_at_its_head(tmp_path: Path) -> None:
    """A rolled recurrence back-edge labels each end at its own node.

    Restored behaviour (JMT, 3 Nov 2026): ``Out`` names the passes the edge leaves
    its tail on and sits at the tail, ``In`` names the passes it enters its head on
    and sits at the head; a merged two-line label floated between the nodes and
    belonged to neither.  ``TanhRNNCellLoop`` rolls to a two-op cycle whose
    back-edge (``tanh_1_2 -> linear_1_1``) is the return arc.  The forward edge runs
    on every pass at both ends, so it carries no pass label, only a blank midpoint
    slot that keeps the two curves apart.
    """

    dot = _render_dot(_trace(TanhRNNCellLoop()), tmp_path, "rnn_back_edge", depth=1000)

    back_edges = _dot_edge_attrs(dot, "tanh_1_2", "linear_1_1")
    assert len(back_edges) == 1
    back_attrs = back_edges[0]
    assert _html_label_lines(back_attrs, "taillabel") == ["Out 1-3"]
    assert _html_label_lines(back_attrs, "headlabel") == ["In 2-4"]
    assert _html_label_lines(back_attrs, "label") is None
    assert _html_label_lines(back_attrs, "xlabel") is None

    forward_edges = _dot_edge_attrs(dot, "linear_1_1", "tanh_1_2")
    assert len(forward_edges) == 1
    forward_attrs = forward_edges[0]
    assert "headlabel" not in forward_attrs
    assert "taillabel" not in forward_attrs
    assert _html_label_lines(forward_attrs, "label") == []
    # Graphviz 2.43 rejects an empty FONT element as a syntax error.
    assert re.search(r"<FONT[^>]*></FONT>", dot, flags=re.IGNORECASE) is None


def test_buffer_edges_label_each_end_at_its_node(tmp_path: Path) -> None:
    """Buffer-incident edges put each pass annotation at the node it describes.

    Restored head/tail placement (JMT, 3 Nov 2026): each edge labels only the ends
    that a strict subset of their node's passes use, ``Out`` at the tail and ``In``
    at the head, never as a midpoint label between the two nodes.
    """

    dot = _render_dot(_trace(BufferRewriteLoops()), tmp_path, "buffer_edges", show_buffers="always")

    read_edges = _dot_edge_attrs(dot, "buffer_1", "add_1_1")
    assert len(read_edges) == 1
    # Every pass of the read uses the edge, so only the buffer end is labelled.
    assert _html_label_lines(read_edges[0], "taillabel") == ["Out 1-5"]

    fanout_read_edges = _dot_edge_attrs(dot, "buffer_1", "add_2_2")
    assert len(fanout_read_edges) == 1
    assert _html_label_lines(fanout_read_edges[0], "taillabel") == ["Out 1-5"]

    write_edges = _dot_edge_attrs(dot, "add_2_2", "buffer_1")
    assert len(write_edges) == 1
    # The write edge closes the loop: its pass range sits at the buffer it enters.
    assert _html_label_lines(write_edges[0], "headlabel") == ["In 2-6"]

    for attrs in (read_edges[0], fanout_read_edges[0], write_edges[0]):
        assert _html_label_lines(attrs, "label") in (None, [])
        assert _html_label_lines(attrs, "xlabel") is None


def test_varying_forward_edges_label_their_partial_ends_at_the_node(tmp_path: Path) -> None:
    """Rolled edges label only the ends a strict subset of passes use, at that end.

    ``DeepLoopBody`` is a three-op cycle. The body edges run on every pass at
    both ends, so they carry no pass label; the input edge (pass 1 only) and the
    back-edge (passes 1-2 into 2-3) keep theirs, each beside its node: the input
    edge's ``In 1`` as the head label at the loop's first op, the back-edge's
    ``Out 1-2`` at its tail and ``In 2-3`` at its head (restored head/tail
    placement, JMT 3 Nov 2026).  Only pass-labelled edges get a chosen
    labeldistance/labelangle; unlabelled body edges carry none.
    """

    drawn = _render_dot(_trace(DeepLoopBody()), tmp_path, "deep_forward", depth=1000)
    assert "labelangle=" not in drawn  # placement goes only into the image's source
    dot = place_pass_labels(drawn, engine="dot", timeout=60)

    body_out = _dot_edge_attrs(dot, "linear_1_1", "relu_1_2")
    assert len(body_out) == 1
    assert _html_label_lines(body_out[0], "label") is None

    body_in = _dot_edge_attrs(dot, "relu_1_2", "layernorm_1_3")
    assert len(body_in) == 1
    assert _html_label_lines(body_in[0], "label") is None

    input_edge = _dot_edge_attrs(dot, "input_1", "linear_1_1")
    assert len(input_edge) == 1
    assert _html_label_lines(input_edge[0], "headlabel") == ["In 1"]
    assert _html_label_lines(input_edge[0], "label") is None

    for attrs in (body_out[0], body_in[0], input_edge[0]):
        assert "taillabel" not in attrs
    for attrs in (body_out[0], body_in[0]):
        assert "headlabel" not in attrs
        assert "labeldistance=" not in attrs
        assert "labelangle=" not in attrs
    assert "labeldistance=" in input_edge[0] and "labelangle=" in input_edge[0]

    back_edge = _dot_edge_attrs(dot, "layernorm_1_3", "linear_1_1")
    assert len(back_edge) == 1
    assert _html_label_lines(back_edge[0], "taillabel") == ["Out 1-2"]
    assert _html_label_lines(back_edge[0], "headlabel") == ["In 2-3"]
    assert _html_label_lines(back_edge[0], "xlabel") is None


def test_rolled_edges_label_only_ends_a_strict_subset_of_passes_uses(tmp_path: Path) -> None:
    """An edge end that every pass of its node uses carries no pass label.

    In ``BranchingLoopBody`` the body edge ``linear -> tanh`` runs on all three
    passes at both ends and carries no text (it used to read ``In 1-3 / Out
    1-3``); the input edge (pass 1), the back-edge (passes 1-2 into 2-3) and the
    branch edge (pass 3 only) keep exactly their partial ends.
    """

    dot = _render_dot(_trace(BranchingLoopBody()), tmp_path, "branching_loop", depth=1000)
    texts: set[str] = set()
    for line in dot.splitlines():
        if " -> " in line:
            for kind in ("label", "xlabel", "headlabel", "taillabel"):
                texts.update(_html_label_lines(line, kind) or [])
    assert texts == {"In 1", "In 2-3", "Out 1-2", "Out 3"}, sorted(texts)
    body = _dot_edge_attrs(dot, "linear_1_1", "tanh_1_2")
    assert len(body) == 1
    # Blank, not absent: the back-edge joins the same two nodes, and the blank
    # slot keeps Graphviz from bundling the loop's two splines into one line.
    assert _html_label_lines(body[0], "label") == []


def test_atomic_single_op_module_collapse_preserves_op_render(tmp_path: Path) -> None:
    """Single-op modules render as the same atomic rectangle at every depth.

    An atomic module is already maximally collapsed: it renders as a rectangle
    marked ``@relu`` (never an ellipse op, never a ``box3d`` collapsed-module
    summary, never a ``cluster_relu`` box), identically at shallow and deep depth.
    """

    trace = _trace(ReusedReluLoop())
    expanded_dot = _render_dot(trace, tmp_path, "relu_expanded", depth=1000)
    collapsed_dot = _render_dot(trace, tmp_path, "relu_collapsed", depth=1)

    for dot in (expanded_dot, collapsed_dot):
        assert "cluster_relu" not in dot
        assert "box3d" not in dot
        assert "relu_1_1 (x4)" in dot
        # Marked as a module and rendered as a rectangle (not an ellipse op).
        assert "@relu" in dot
        assert _dot_node_shapes(dot).get("relu_1_1 (x4)") == "box"
        assert 'label="↻"' not in dot


@pytest.mark.heavy
def test_render_loop_module_rolling_demos() -> None:
    """Render SVG, PDF, and PNG demos into the committed test-output folder."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    demos: list[tuple[str, nn.Module, dict[str, object], str]] = [
        ("inside_outside_relu", InsideOutsideRelu(), {"depth": 1}, "init+forward"),
        (
            "inside_outside_relu_separable",
            InsideOutsideReluSeparable(),
            {"depth": 1},
            "init+forward",
        ),
        (
            "inside_outside_block_collapsed",
            InsideOutsideBlock(),
            {"depth": 1},
            "init+forward",
        ),
        (
            "inside_outside_block_expanded",
            InsideOutsideBlock(),
            {"depth": 1000},
            "init+forward",
        ),
        ("deep_loop_body", DeepLoopBody(), {"depth": 1000}, "init+forward"),
        ("rnn_cell", TanhRNNCellLoop(), {"depth": 1000}, "init+forward"),
        (
            "repeated_block_stack_collapsed",
            RepeatedBlockStack(),
            {"depth": 1},
            "init+forward",
        ),
        (
            "repeated_block_stack_expanded",
            RepeatedBlockStack(),
            {"depth": 1000},
            "init+forward",
        ),
        ("two_distinct_loops", TwoDistinctLoops(), {"depth": 1}, "init+forward"),
        (
            "buffer_loop",
            BufferRewriteLoops(),
            {"show_buffers": "always"},
            "init+forward",
        ),
        ("nested_loop", NestedLoopBlock(), {"depth": 1}, "init+forward"),
        ("parallel_fanout", ParallelFanout(), {"depth": 1}, "init+forward"),
    ]
    # Magic-byte / markup signatures so a no-op renderer (no file, or an empty
    # one) cannot pass: PDF starts with %PDF, PNG with the 8-byte PNG header,
    # and SVG payloads contain an <svg element.
    _format_signature = {"pdf": b"%PDF", "png": b"\x89PNG\r\n\x1a\n"}
    for name, model, kwargs, code_panel in demos:
        trace = _trace(model)
        for file_format in ("svg", "pdf", "png"):
            artifact = OUTPUT_DIR / f"{name}.{file_format}"
            artifact.unlink(missing_ok=True)
            dot = trace.draw(
                rolled=True,
                file_format=file_format,
                path=str(OUTPUT_DIR / name),
                code_panel=code_panel,
                **kwargs,
            ).dot
            assert isinstance(dot, str) and "digraph" in dot, (
                f"expected non-empty DOT source for {name} ({file_format})"
            )
            assert artifact.exists(), f"expected render artifact at {artifact}"
            blob = artifact.read_bytes()
            assert blob, f"expected non-empty artifact at {artifact}"
            if file_format in _format_signature:
                assert blob.startswith(_format_signature[file_format]), (
                    f"expected {artifact} to be a valid {file_format} artifact"
                )
            else:  # svg
                assert b"<svg" in blob, f"expected {artifact} to contain SVG markup"
