"""Regression tests for loop-rolling and module-rolling reconciliation."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.visualization.rendering import compute_default_node_lines


OUTPUT_DIR = Path(__file__).parent / "test_outputs" / "loop_module_rolling"


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

    return trace.draw(
        vis_mode="rolled",
        vis_save_only=True,
        vis_fileformat="dot",
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


def test_render_loop_module_rolling_demos() -> None:
    """Render SVG and PDF demos into the committed test-output folder."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    demos: list[tuple[str, nn.Module, dict[str, object]]] = [
        ("reused_relu_loop", ReusedReluLoop(), {}),
        ("parallel_fanout", ParallelFanout(), {}),
        ("wrapped_two_site_loop", WrappedTwoSiteLoop(), {}),
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
