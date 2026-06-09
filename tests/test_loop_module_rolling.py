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


@pytest.mark.smoke
def test_reused_single_op_module_count_is_depth_invariant(tmp_path: Path) -> None:
    """A reused single-op module keeps the honest count at shallow and deep depth."""

    trace = _trace(ReusedReluLoop())
    assert _normalized_title(trace, "relu", "relu") == "relu (x4)"

    shallow_dot = _render_dot(trace, tmp_path, "relu_shallow", vis_call_depth=1)
    deep_dot = _render_dot(trace, tmp_path, "relu_deep", vis_call_depth=1000)

    assert "@relu (x4)" in shallow_dot
    assert "relu_1_1 (x4)" in deep_dot


def test_parallel_fanout_is_one_site_with_fanout_metadata() -> None:
    """Parallel stack fan-out keeps one site and exposes facet metadata."""

    trace = _trace(ParallelFanout())
    assert _normalized_title(trace, "linear", "proj") == "proj (x4)"
    annotation = compute_default_node_lines(_first_layer(trace, "linear"), vis_mode="rolled")
    assert annotation[0] == "linear_1_1 (x4)"


def test_wrapped_two_site_loop_gets_module_certified_multiplier() -> None:
    """A wrapped two-site layer receives ``2 sites ×3``."""

    trace = _trace(WrappedTwoSiteLoop())
    assert _normalized_title(trace, "linear", "lin") == "lin (x6 · 2 sites ×3)"


def test_two_distinct_loops_are_partitioned_not_summed(tmp_path: Path) -> None:
    """Two loops sharing one module show call partitions on layer and box labels."""

    trace = _trace(TwoDistinctLoops())
    assert _normalized_title(trace, "linear", "lin") == "lin (x5 · calls 1-2,3-5)"

    dot = _render_dot(trace, tmp_path, "two_loops", vis_call_depth=1)
    assert "@shared (x5 · calls 1-2,3-5)" in dot
    assert "@shared (x5)</B>" not in dot


def test_buffer_versions_are_surfaced(tmp_path: Path) -> None:
    """Rolled buffer nodes show flat version sets instead of a bare address."""

    trace = _trace(BufferRewriteLoops())
    dot = _render_dot(trace, tmp_path, "buffers", show_buffer_layers="always")
    assert "@state v1-6" in dot
    assert "buffer_versions=1-6" in dot


def test_mixed_chain_fanout_keeps_clean_count_and_metadata(tmp_path: Path) -> None:
    """Mixed dependency keeps ``proj (x3)`` and exposes ``facet=mixed`` metadata."""

    trace = _trace(MixedDependency())
    assert _normalized_title(trace, "linear", "proj") == "proj (x3)"

    dot = _render_dot(trace, tmp_path, "mixed")
    assert "facet=mixed" in dot


def test_inside_outside_loop_shows_call_partition() -> None:
    """A layer called inside and outside a loop shows the honest call partition."""

    trace = _trace(InsideOutsideLoop())
    assert _normalized_title(trace, "linear", "lin") == "lin (x4 · calls 1,2-4)"


def test_parallel_siblings_loop_keeps_two_sites_without_multiplier() -> None:
    """Parallel siblings in a loop show two sites and no fabricated multiplier."""

    trace = _trace(ParallelSiblingsLoop())
    assert _normalized_title(trace, "linear", "lin") == "lin (x8 · 2 sites)"


def test_integral_ragged_module_body_is_uncertain_without_multiplier() -> None:
    """Uneven per-module body sizes show honest site uncertainty and no ``×P``."""

    trace = _trace(IntegralRaggedBlocks())
    title = _normalized_title(trace, "linear", "lin")
    assert title == "lin (x4 · sites?)"
    assert "×" not in title


def test_non_uniform_equal_cardinality_body_rejects_multiplier() -> None:
    """Equal-sized module calls with different site signatures do not certify ``×P``."""

    trace = _trace(NonUniformBodyCalls())
    title = _normalized_title(trace, "linear", "lin")
    assert title == "lin (x4 · 3 sites)"
    assert "×2" not in title


def test_ragged_block_loop_surfaces_site_uncertainty() -> None:
    """A 2, 2, 3 repeated block body shows ``sites?`` instead of a fabricated count."""

    trace = _trace(RaggedBlockLoop())
    assert _normalized_title(trace, "linear", "lin") == "lin (x7 · sites?)"


def test_mixed_facet_does_not_collapse_sites(tmp_path: Path) -> None:
    """Mixed facet remains metadata and does not force the site partition to one."""

    trace = _trace(MixedWithSites())
    assert _normalized_title(trace, "linear", "lin") == "lin (x3 · 2 sites)"

    dot = _render_dot(trace, tmp_path, "mixed_with_sites")
    assert "facet=mixed" in dot
    assert "linear_1_1 (x3 · 2 sites · mixed)" in dot


def test_too_deep_consumer_container_path_surfaces_site_uncertainty() -> None:
    """A consumer arg path beyond the two-level renderer scheme shows ``sites?``."""

    trace = _trace(StackFanoutForDeepArgMutation())
    layer = _first_layer(trace, "linear")
    first_op = next(iter(layer.ops.values()))
    stack_label = first_op.children[0]
    stack_op = trace.layer_dict_all_keys[stack_label]
    stack_op.parent_arg_positions["args"] = {
        (0, 0, 0): "linear_1_1:1",
        (0, 0, 1): "linear_1_1:2",
    }

    assert _normalized_title(trace, "linear", "lin") == "lin (x2 · sites?)"


def test_static_targets_promote_parallel_and_mixed_facets(tmp_path: Path) -> None:
    """Static face labels include static-only facet tags for non-chain facets."""

    fanout_dot = _render_dot(_trace(ParallelFanout()), tmp_path, "static_parallel")
    mixed_dot = _render_dot(_trace(MixedDependency()), tmp_path, "static_mixed")
    fanout_svg = _render_source(
        _trace(ParallelFanout()),
        tmp_path,
        "static_parallel_svg",
        "svg",
    )
    mixed_pdf = _render_source(
        _trace(MixedDependency()),
        tmp_path,
        "static_mixed_pdf",
        "pdf",
    )

    assert "linear_1_1 (x4 · parallel)" in fanout_dot
    assert "linear_1_1 (x3 · mixed)" in mixed_dot
    assert "linear_1_1 (x4 · parallel)" in fanout_svg
    assert "linear_1_1 (x3 · mixed)" in mixed_pdf


def test_render_cache_does_not_mutate_save_or_fork_trace(tmp_path: Path) -> None:
    """Rendering memoization stays off the trace state and forks."""

    trace = _trace(ParallelSiblingsLoop())
    _render_dot(trace, tmp_path, "cache_parent")

    assert "_tl_rendering_cache" not in trace.__dict__
    assert "_tl_rendering_cache" not in trace.__getstate__()

    fork = trace.fork("cache_fork")
    assert "_tl_rendering_cache" not in fork.__dict__
    assert "_tl_rendering_cache" not in fork.__getstate__()


def test_render_cache_is_fresh_after_same_trace_structural_mutation(tmp_path: Path) -> None:
    """A second render sees in-place structural edits on the same trace."""

    trace = _trace(StackFanoutForDeepArgMutation())
    layer = _first_layer(trace, "linear")

    before_dot = _render_dot(trace, tmp_path, "cache_before_mutation", vis_call_depth=1000)
    assert f"{layer.layer_label} (x2" in before_dot
    assert f"{layer.layer_label} (x2 · sites?" not in before_dot

    first_op = next(iter(layer.ops.values()))
    stack_label = first_op.children[0]
    stack_op = trace.layer_dict_all_keys[stack_label]
    stack_op.parent_arg_positions["args"] = {
        (0, 0, 0): "linear_1_1:1",
        (0, 0, 1): "linear_1_1:2",
    }

    after_dot = _render_dot(trace, tmp_path, "cache_after_mutation", vis_call_depth=1000)
    assert f"{layer.layer_label} (x2 · sites?" in after_dot


def test_rolled_recurrent_layer_draws_marked_self_edge(tmp_path: Path) -> None:
    """A recurrent rolled layer keeps its same-layer self-edge as ``↻``."""

    dot = _render_dot(_trace(RNNCellLoop()), tmp_path, "rnn_cell_layer", vis_call_depth=1000)

    assert "linear_1_1 -> linear_1_1" in dot
    assert 'label="↻"' in dot
    assert "TorchLens recurrence: endpoint=linear_1_1;" in dot
    assert "pass_pairs=1->2,2->3,3->4; count=3" in dot


def test_collapsed_module_recurrence_draws_marked_self_edge(tmp_path: Path) -> None:
    """A collapsed recurrent module box keeps its internal self-edge as ``↻``."""

    dot = _render_dot(
        _trace(CollapsedBlockRecurrence()), tmp_path, "block_recurrence", vis_call_depth=1
    )

    assert "block -> block" in dot
    assert 'label="↻"' in dot
    assert "TorchLens recurrence: endpoint=relu_1_2->linear_1_1@block;" in dot
    assert "pass_pairs=1->2,2->3" in dot
    assert "count=2" in dot


def test_reused_module_without_loop_carried_dep_has_no_recurrence_self_edge(
    tmp_path: Path,
) -> None:
    """A reused non-recurrent module does not receive a spurious ``↻`` self-edge."""

    dot = _render_dot(_trace(ParallelFanout()), tmp_path, "fanout_no_recurrence", vis_call_depth=1)

    assert "proj -> proj" not in dot
    assert 'label="↻"' not in dot


def test_interleaved_same_layer_recurrences_keep_distinct_chains(tmp_path: Path) -> None:
    """Two same-layer recurrence sites remain distinct in deep and collapsed views."""

    trace = _trace(SharedTwoSiteRecurrences())
    deep_dot = _render_dot(trace, tmp_path, "two_site_recurrence_deep", vis_call_depth=1000)
    collapsed_dot = _render_dot(trace, tmp_path, "two_site_recurrence_collapsed", vis_call_depth=1)

    for dot in (deep_dot, collapsed_dot):
        assert 'label="↻"' in dot
        assert "pass_pairs=1->3,2->4,3->5,4->6" in dot
        assert "chains=1->3->5,2->4->6" in dot
        assert "pass_pairs=1->2,2->3" not in dot

    layer = _first_layer(trace, "linear")
    assert f"{layer.layer_label} -> {layer.layer_label}" in deep_dot
    assert "block -> block" in collapsed_dot


def test_render_loop_module_rolling_demos() -> None:
    """Render SVG and PDF demos into the committed test-output folder."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    demos: list[tuple[str, nn.Module, dict[str, object]]] = [
        ("reused_relu_loop", ReusedReluLoop(), {}),
        ("parallel_fanout", ParallelFanout(), {}),
        ("wrapped_two_site_loop", WrappedTwoSiteLoop(), {}),
        ("rnn_cell_recurrence", RNNCellLoop(), {"vis_call_depth": 1}),
        ("two_distinct_loops", TwoDistinctLoops(), {"vis_call_depth": 1}),
        ("buffer_rewrite_loops", BufferRewriteLoops(), {"show_buffer_layers": "always"}),
        ("mixed_dependency", MixedDependency(), {}),
        ("inside_outside_loop", InsideOutsideLoop(), {}),
    ]
    for name, model, kwargs in demos:
        trace = _trace(model)
        for file_format in ("svg", "pdf"):
            trace.draw(
                vis_mode="rolled",
                vis_save_only=True,
                vis_fileformat=file_format,
                vis_outpath=str(OUTPUT_DIR / name),
                **kwargs,
            )
