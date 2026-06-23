"""Regression coverage for smart module auto-collapse."""

from __future__ import annotations

import re
import time
from pathlib import Path

import torch

import torchlens as tl
from torchlens.visualization.auto_collapse import analyze_collapse


class ResidualBlock(torch.nn.Module):
    """Small residual block with enough internal structure to collapse."""

    def __init__(self, width: int = 8) -> None:
        """Initialize the block."""

        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(width, width),
            torch.nn.ReLU(),
            torch.nn.Linear(width, width),
            torch.nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the residual block."""

        return self.net(x) + x


class RepeatedResidual(torch.nn.Module):
    """Repeated residual blocks for peer and budget tests."""

    def __init__(self, depth: int = 8, width: int = 8) -> None:
        """Initialize repeated residual blocks."""

        super().__init__()
        self.blocks = torch.nn.ModuleList([ResidualBlock(width) for _ in range(depth)])
        self.out = torch.nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the repeated residual model."""

        for block in self.blocks:
            x = block(x)
        return self.out(x)


class ResidualBody(torch.nn.Module):
    """Residual branch body whose join lives in the parent module."""

    def __init__(self, width: int = 8) -> None:
        """Initialize residual body layers."""

        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(width, width),
            torch.nn.ReLU(),
            torch.nn.Linear(width, width),
            torch.nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the residual branch body."""

        return self.net(x)


class ParentJoinResidual(torch.nn.Module):
    """Repeated residual stages with parent-level add junctions."""

    def __init__(self, depth: int = 5, width: int = 8) -> None:
        """Initialize repeated residual stages."""

        super().__init__()
        self.blocks = torch.nn.ModuleList([ResidualBody(width) for _ in range(depth)])
        self.out = torch.nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the residual stack."""

        for block in self.blocks:
            x = block(x) + x
        return self.out(x)


class ConvReluBlock(torch.nn.Module):
    """Small convolutional block without buffer-only bookkeeping."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize convolutional layers."""

        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, 3, padding=1),
            torch.nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the block."""

        return self.net(x)


class SkipConcatUNet(torch.nn.Module):
    """Tiny U-Net-style model with a cross-module concat junction."""

    def __init__(self) -> None:
        """Initialize encoder and decoder blocks."""

        super().__init__()
        self.enc1 = ConvReluBlock(4, 4)
        self.enc2 = ConvReluBlock(4, 4)
        self.dec1 = ConvReluBlock(8, 4)
        self.out = torch.nn.Conv2d(4, 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the skip-concat model."""

        skip = self.enc1(x)
        deep = torch.nn.functional.avg_pool2d(self.enc2(skip), 2)
        up = torch.nn.functional.interpolate(deep, scale_factor=2, mode="nearest")
        return self.out(self.dec1(torch.cat([skip, up], dim=1)))


class BranchConcat(torch.nn.Module):
    """Inception-like parallel branches with a parent-level concat."""

    def __init__(self) -> None:
        """Initialize branch modules."""

        super().__init__()
        self.a = torch.nn.Sequential(
            torch.nn.Conv2d(4, 4, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(4, 4, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(4, 4, 1),
        )
        self.b = torch.nn.Sequential(
            torch.nn.Conv2d(4, 4, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(4, 4, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(4, 4, 1),
        )
        self.c = torch.nn.Sequential(
            torch.nn.AvgPool2d(3, stride=1, padding=1),
            torch.nn.Conv2d(4, 4, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(4, 4, 1),
            torch.nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run all branches and concatenate their outputs."""

        return torch.cat([self.a(x), self.b(x), self.c(x)], dim=1)


class RecurrentWrapper(torch.nn.Module):
    """GRU wrapper for recurrent module collapse coverage."""

    def __init__(self) -> None:
        """Initialize recurrent layers."""

        super().__init__()
        self.rnn = torch.nn.GRU(8, 8, batch_first=True)
        self.head = torch.nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run recurrent model."""

        y, _ = self.rnn(x)
        return self.head(y[:, -1])


class TrivialSingle(torch.nn.Module):
    """Single-op model whose only submodule should not be collapse-eligible."""

    def __init__(self) -> None:
        """Initialize the single-op module."""

        super().__init__()
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the trivial model."""

        return self.relu(x)


class LongFunctional(torch.nn.Module):
    """Large op-count model for signal-tally latency coverage."""

    def __init__(self, depth: int = 3000) -> None:
        """Initialize the operation depth."""

        super().__init__()
        self.depth = depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run many functional operations."""

        for _ in range(self.depth):
            x = torch.relu(x + 1.0)
        return x


def _trace(model: torch.nn.Module, x: torch.Tensor) -> tl.Trace:
    """Capture ``model`` under ``torch.no_grad``."""

    with torch.no_grad():
        return tl.trace(model.eval(), x)


def _draw_source(trace: tl.Trace, tmp_path: Path, name: str, collapse: str) -> str:
    """Render a trace to SVG and return DOT source."""

    return str(
        trace.draw(
            vis_outpath=str(tmp_path / name),
            vis_save_only=True,
            vis_fileformat="svg",
            vis_node_placement="dot",
            collapse=collapse,
        )
    )


def _box_count(source: str) -> int:
    """Return collapsed module node count from DOT source."""

    return source.count("shape=box3d")


def _dot_node_count(source: str) -> int:
    """Return an approximate rendered node count from DOT source."""

    names = re.findall(
        r'^\s*("[^"]+"|[A-Za-z0-9_.]+(?:pass\d+)?) \[',
        source,
        flags=re.MULTILINE,
    )
    return len([name for name in names if name not in {"graph", "node", "edge"}])


def _has_visible_node(source: str, prefix: str) -> bool:
    """Return whether DOT source contains a visible node with ``prefix``.

    Parameters
    ----------
    source:
        DOT source emitted by a render.
    prefix:
        Node identifier prefix to find.

    Returns
    -------
    bool
        True when an explicit node declaration starts with ``prefix``.
    """

    pattern = rf'^\s*"?{re.escape(prefix)}[A-Za-z0-9_.]*"? \['
    return re.search(pattern, source, flags=re.MULTILINE) is not None


def test_auto_collapse_budget_boxes_grain_and_determinism(tmp_path: Path) -> None:
    """Auto collapse hits the overview budget and renders deterministically."""

    trace = _trace(RepeatedResidual(depth=8), torch.randn(2, 8))
    try:
        none_source = _draw_source(trace, tmp_path, "none", "none")
        auto_source = _draw_source(trace, tmp_path, "auto1", "auto")
        auto_source_again = _draw_source(trace, tmp_path, "auto2", "auto")
        max_source = _draw_source(trace, tmp_path, "max", "max")

        assert auto_source == auto_source_again
        assert _box_count(none_source) == 0
        assert _box_count(auto_source) >= 1
        assert _box_count(max_source) >= _box_count(auto_source)
        assert _dot_node_count(auto_source) <= _dot_node_count(none_source)
        assert 8 <= _dot_node_count(auto_source) <= 40

        collapsed_scores = [score for _, score in trace.module_collapse_order if score > 0]
        assert collapsed_scores
        assert trace.module_collapse_order == sorted(
            trace.module_collapse_order,
            key=lambda item: (-item[1], item[0]),
        )

        selected_sizes = [
            analyze_collapse(trace).signals[address].hidden_ops
            for address, score in trace.module_collapse_order
            if score > 0
        ]
        assert max(selected_sizes) - min(selected_sizes) <= max(selected_sizes)
        assert "input_" in auto_source
        assert "output_" in auto_source
    finally:
        trace.cleanup()


def test_repeat_capture_and_trivial_collapse_score() -> None:
    """Scores are deterministic across captures, and trivial modules score zero."""

    first = _trace(RepeatedResidual(depth=4), torch.randn(2, 8))
    second = _trace(RepeatedResidual(depth=4), torch.randn(2, 8))
    trivial = _trace(TrivialSingle(), torch.randn(2, 8))
    try:
        assert first.module_collapse_order == second.module_collapse_order
        assert trivial.modules["relu"].collapse_score == 0.0
    finally:
        first.cleanup()
        second.cleanup()
        trivial.cleanup()


def test_auto_collapse_residual_peer_bodies_keep_parent_joins(tmp_path: Path) -> None:
    """Auto folds repeated residual bodies while keeping add junction nodes visible."""

    trace = _trace(ParentJoinResidual(depth=5), torch.randn(2, 8))
    try:
        none_source = _draw_source(trace, tmp_path, "parent_residual_none", "none")
        auto_source = _draw_source(trace, tmp_path, "parent_residual_auto", "auto")

        assert _dot_node_count(none_source) > 15
        assert _dot_node_count(auto_source) <= int(_dot_node_count(none_source) * 0.7)
        assert _box_count(auto_source) >= 5
        assert _has_visible_node(auto_source, "add_")
        assert auto_source != none_source
    finally:
        trace.cleanup()


def test_auto_collapse_skip_concat_keeps_cat_junction(tmp_path: Path) -> None:
    """Auto folds U-Net-style blocks while keeping the skip concat visible."""

    trace = _trace(SkipConcatUNet(), torch.randn(1, 4, 16, 16))
    try:
        none_source = _draw_source(trace, tmp_path, "skip_concat_none", "none")
        auto_source = _draw_source(trace, tmp_path, "skip_concat_auto", "auto")
        max_source = _draw_source(trace, tmp_path, "skip_concat_max", "max")

        assert _dot_node_count(none_source) > 15
        assert _dot_node_count(auto_source) <= int(_dot_node_count(none_source) * 0.7)
        assert _box_count(auto_source) >= 2
        assert _has_visible_node(auto_source, "cat_")
        assert auto_source != none_source
        assert auto_source != max_source
    finally:
        trace.cleanup()


def test_auto_collapse_branch_concat_keeps_cat_junction(tmp_path: Path) -> None:
    """Auto folds parallel branches while keeping the branch concat visible."""

    trace = _trace(BranchConcat(), torch.randn(1, 4, 16, 16))
    try:
        none_source = _draw_source(trace, tmp_path, "branch_concat_none", "none")
        auto_source = _draw_source(trace, tmp_path, "branch_concat_auto", "auto")

        assert _dot_node_count(none_source) > 15
        assert _dot_node_count(auto_source) <= int(_dot_node_count(none_source) * 0.7)
        assert _box_count(auto_source) >= 3
        assert _has_visible_node(auto_source, "cat_")
        assert auto_source != none_source
    finally:
        trace.cleanup()


def test_auto_collapse_recurrent_module_is_not_noop(tmp_path: Path) -> None:
    """Auto folds recurrent modules instead of treating recurrence as a veto."""

    trace = _trace(RecurrentWrapper(), torch.randn(1, 6, 8))
    try:
        none_source = _draw_source(trace, tmp_path, "recurrent_none", "none")
        auto_source = _draw_source(trace, tmp_path, "recurrent_auto", "auto")

        assert _dot_node_count(auto_source) < _dot_node_count(none_source)
        assert _box_count(auto_source) >= 1
        assert "@rnn" in auto_source
        assert auto_source != none_source
    finally:
        trace.cleanup()


def test_signal_tally_latency_under_budget() -> None:
    """Signal tally stays under the 50 ms per 3k-node budget."""

    trace = _trace(LongFunctional(depth=1500), torch.randn(1, 8))
    try:
        start = time.perf_counter()
        analyze_collapse(trace)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        scaled_ms = elapsed_ms * (3000.0 / max(1, len(trace.ops)))
        assert scaled_ms < 50.0
    finally:
        trace.cleanup()
