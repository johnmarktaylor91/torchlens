"""Regression tests for loop-rolling and module-rolling reconciliation."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.visualization.rendering import compute_default_node_lines


OUTPUT_DIR = Path(__file__).parent / "test_outputs" / "visualizations" / "loop_module_rolling"


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
    """Buffer rewritten across repeated loop bodies."""

    def __init__(self) -> None:
        """Initialize the state buffer."""

        super().__init__()
        self.register_buffer("state", torch.zeros(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Read and rewrite the buffer across two loops."""

        y = x
        for _ in range(2):
            y = y + self.state
            self.state = self.state + 1
        for _ in range(3):
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


def _trace(model: nn.Module) -> tl.Trace:
    """Trace a demo model deterministically."""

    torch.manual_seed(0)
    return tl.trace(model, torch.randn(1, 4), layers_to_save="none")


def _first_layer(trace: tl.Trace, layer_type: str) -> tl.Layer:
    """Return the first rendered layer of a given type."""

    for layer in trace.layer_logs.values():
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
        vis_mode="rolled",
        vis_save_only=True,
        vis_fileformat=file_format,
        vis_outpath=str(tmp_path / name),
        **kwargs,
    )


def _dot_self_edge_attrs(dot: str) -> list[str]:
    """Return DOT attributes for explicit ``name -> name`` self-edges."""

    return [
        match.group("attrs")
        for match in re.finditer(
            r"(?m)^\s*(?P<name>[A-Za-z0-9_.]+) -> (?P=name) \[(?P<attrs>[^\]]*)\]",
            dot,
        )
    ]


@pytest.mark.smoke
def test_reused_single_op_module_count_is_depth_invariant(tmp_path: Path) -> None:
    """A reused single-op module keeps the honest count at shallow and deep depth."""

    trace = _trace(ReusedReluLoop())
    assert _normalized_title(trace, "relu", "relu") == "relu (x4)"

    shallow_dot = _render_dot(trace, tmp_path, "relu_shallow", vis_call_depth=1)
    deep_dot = _render_dot(trace, tmp_path, "relu_deep", vis_call_depth=1000)

    assert "relu_1_1 (x4)" in shallow_dot
    assert "relu_1_1 (x4)" in deep_dot


def test_multi_op_loop_body_keeps_plain_counts(tmp_path: Path) -> None:
    """A Linear -> ReLU loop body shows honest counts without site suffixes."""

    trace = _trace(CollapsedBlockRecurrence())
    assert _normalized_title(trace, "linear", "lin") == "lin (x3)"
    assert _normalized_title(trace, "relu", "relu") == "relu (x3)"

    dot = _render_dot(trace, tmp_path, "multiop_loop", vis_call_depth=1000)
    assert "sites" not in dot
    assert "×" not in dot
    assert 'label="↻"' not in dot


def test_colon_call_ranges_disambiguate_split_regions(tmp_path: Path) -> None:
    """Reused modules in split regions use compact colon ranges."""

    distinct_trace = _trace(TwoDistinctLoops())
    assert _normalized_title(distinct_trace, "linear", "lin") == "lin:1-2,3-5"
    distinct_dot = _render_dot(distinct_trace, tmp_path, "two_loops", vis_call_depth=1)
    assert "@shared:1-2,3-5" in distinct_dot
    assert "calls" not in distinct_dot

    inside_outside_trace = _trace(InsideOutsideLoop())
    assert _normalized_title(inside_outside_trace, "linear", "lin") == "lin:1,2-4"


def test_clean_reused_block_keeps_plain_counts(tmp_path: Path) -> None:
    """A reused multi-op block in one loop keeps plain module and op counts."""

    trace = _trace(CollapsedBlockRecurrence())
    expanded_dot = _render_dot(trace, tmp_path, "block_expanded", vis_call_depth=1000)
    collapsed_dot = _render_dot(trace, tmp_path, "block_collapsed", vis_call_depth=1)

    assert "linear_1_1 (x3)" in expanded_dot
    assert "relu_1_2 (x3)" in expanded_dot
    assert "@block (x3)" in collapsed_dot
    assert 'label="↻"' not in expanded_dot
    assert 'label="↻"' not in collapsed_dot


def test_buffer_versions_use_colon_ranges(tmp_path: Path) -> None:
    """Rolled buffer nodes use the module-address colon range form."""

    trace = _trace(BufferRewriteLoops())
    dot = _render_dot(trace, tmp_path, "buffers", show_buffer_layers="always")
    assert "@state:1-6" in dot
    assert "@state v1-6" not in dot
    assert "buffer_versions" not in dot


def test_recurrent_self_loop_is_clean_and_fanout_has_none(tmp_path: Path) -> None:
    """Rolled recurrence keeps one clean self-edge; parallel fan-out has none."""

    recurrent_trace = _trace(ReusedReluLoop())
    recurrent_dot = _render_dot(recurrent_trace, tmp_path, "relu_self_edge", vis_call_depth=1000)
    recurrent_self_edges = _dot_self_edge_attrs(recurrent_dot)
    assert len(recurrent_self_edges) == 1
    recurrent_attrs = recurrent_self_edges[0]
    assert "label=" not in recurrent_attrs
    assert "headlabel=" not in recurrent_attrs
    assert "taillabel=" not in recurrent_attrs
    assert "↻" not in recurrent_attrs
    assert " In " not in recurrent_attrs
    assert " Out " not in recurrent_attrs

    fanout_trace = _trace(ParallelFanout())
    fanout_dot = _render_dot(fanout_trace, tmp_path, "parallel_fanout", vis_call_depth=1000)
    assert _dot_self_edge_attrs(fanout_dot) == []


def test_atomic_single_op_module_collapse_preserves_op_render(tmp_path: Path) -> None:
    """Single-op modules render identically at expanded and collapsed depths."""

    trace = _trace(ReusedReluLoop())
    expanded_dot = _render_dot(trace, tmp_path, "relu_expanded", vis_call_depth=1000)
    collapsed_dot = _render_dot(trace, tmp_path, "relu_collapsed", vis_call_depth=1)

    assert "cluster_relu" not in expanded_dot
    assert "cluster_relu" not in collapsed_dot
    assert "relu_1_1 (x4)" in expanded_dot
    assert "relu_1_1 (x4)" in collapsed_dot
    assert "@relu (x4)" not in collapsed_dot
    assert 'label="↻"' not in expanded_dot
    assert 'label="↻"' not in collapsed_dot


def test_render_loop_module_rolling_demos() -> None:
    """Render SVG and PNG demos into the committed test-output folder."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    demos: list[tuple[str, nn.Module, dict[str, object]]] = [
        ("multiop_loop_body_expanded", CollapsedBlockRecurrence(), {"vis_call_depth": 1000}),
        ("reused_module_inside_outside_loop", InsideOutsideLoop(), {}),
        ("reused_block_loop_expanded", CollapsedBlockRecurrence(), {"vis_call_depth": 1000}),
        ("reused_block_loop_collapsed", CollapsedBlockRecurrence(), {"vis_call_depth": 1}),
        ("buffer_rewrite_loops", BufferRewriteLoops(), {"show_buffer_layers": "always"}),
        ("two_distinct_loops", TwoDistinctLoops(), {"vis_call_depth": 1}),
    ]
    for name, model, kwargs in demos:
        trace = _trace(model)
        for file_format in ("svg", "png"):
            trace.draw(
                vis_mode="rolled",
                vis_save_only=True,
                vis_fileformat=file_format,
                vis_outpath=str(OUTPUT_DIR / name),
                **kwargs,
            )
