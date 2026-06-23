"""Regression coverage for smart module auto-collapse."""

from __future__ import annotations

import re
import time
from pathlib import Path

import torch

import torchlens as tl
from torchlens.visualization.auto_collapse import (
    analyze_collapse,
    resolve_collapse_fn,
    resolve_run_folds,
)


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


class DimStepBlock(torch.nn.Module):
    """Convolutional block whose channel dimensions may change."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize a dim-stepping convolutional block."""

        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the block."""

        return self.net(x)


class DimStepRun(torch.nn.Module):
    """Run of structurally identical blocks with varying channel widths."""

    def __init__(self, depth: int = 4, start_width: int = 4) -> None:
        """Initialize the dim-stepping run."""

        super().__init__()
        widths = list(range(start_width, start_width + depth + 1))
        self.blocks = torch.nn.ModuleList(
            DimStepBlock(in_channels, out_channels)
            for in_channels, out_channels in zip(widths[:-1], widths[1:], strict=True)
        )
        self.head = torch.nn.Conv2d(widths[-1], widths[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run all dim-stepping blocks."""

        for block in self.blocks:
            x = block(x)
        return self.head(x)


class StageUnit(torch.nn.Module):
    """Small convolutional unit used to build synthetic stages."""

    def __init__(self, width: int) -> None:
        """Initialize the unit."""

        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(width, width, 3, padding=1),
            torch.nn.BatchNorm2d(width),
            torch.nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the unit."""

        return self.net(x)


class DepthStage(torch.nn.Module):
    """Same-class stage whose depth is constructor-controlled."""

    def __init__(self, width: int, depth: int) -> None:
        """Initialize a stage with ``depth`` units."""

        super().__init__()
        self.blocks = torch.nn.ModuleList([StageUnit(width) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run all units in the stage."""

        for block in self.blocks:
            x = block(x)
        return x


class UnevenDepthStages(torch.nn.Module):
    """Same-class sibling stages with different internal depths."""

    def __init__(self, depths: tuple[int, ...] = (2, 3, 4)) -> None:
        """Initialize stages of depths two, three, and four."""

        super().__init__()
        self.stem = torch.nn.Conv2d(4, 4, 1)
        self.stages = torch.nn.ModuleList([DepthStage(4, depth) for depth in depths])
        self.head = torch.nn.Conv2d(4, 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the uneven-depth stages."""

        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return self.head(x)


class SpatialStepBlock(torch.nn.Module):
    """Structurally identical block that changes spatial resolution."""

    def __init__(self, width: int) -> None:
        """Initialize the spatial-step block."""

        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(width, width, 3, padding=1),
            torch.nn.BatchNorm2d(width),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the spatial-step block."""

        return self.net(x)


class SpatialStepRun(torch.nn.Module):
    """Run that should not fold because it crosses spatial scales."""

    def __init__(self) -> None:
        """Initialize repeated spatial-step blocks."""

        super().__init__()
        self.blocks = torch.nn.ModuleList([SpatialStepBlock(4) for _ in range(3)])
        self.head = torch.nn.Conv2d(4, 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run all spatial-step blocks."""

        for block in self.blocks:
            x = block(x)
        return self.head(x)


class VggBnFeatures(torch.nn.Module):
    """Flat VGG-style conv-bn-relu-pool features container."""

    def __init__(self) -> None:
        """Initialize a small VGG-BN-style feature extractor."""

        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, 3, padding=1),
            torch.nn.BatchNorm2d(4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(4, 4, 3, padding=1),
            torch.nn.BatchNorm2d(4),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(4, 8, 3, padding=1),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 8, 3, padding=1),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )
        self.classifier = torch.nn.Linear(8 * 4 * 4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run features and classifier."""

        x = self.features(x)
        return self.classifier(torch.flatten(x, 1))


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


class NestedStage(torch.nn.Module):
    """Stage block used inside a generic nested backbone container."""

    def __init__(self, width: int) -> None:
        """Initialize a two-convolution stage."""

        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(width, width, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(width, width, 3, padding=1),
            torch.nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the stage."""

        return self.net(x)


class NestedStageBackbone(torch.nn.Module):
    """Model whose repeated stages live under a generic container."""

    def __init__(self) -> None:
        """Initialize a stem, nested backbone, and head."""

        super().__init__()
        self.stem = torch.nn.Conv2d(4, 4, 1)
        self.backbone = torch.nn.Sequential(*(NestedStage(4) for _ in range(4)))
        self.head = torch.nn.Conv2d(4, 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the nested-stage backbone."""

        return self.head(self.backbone(self.stem(x)))


class StemStageBackbone(torch.nn.Module):
    """Backbone with a standalone leaf-block stem and repeated stage blocks."""

    def __init__(self) -> None:
        """Initialize a leaf-block stem, stages, and leaf-block head."""

        super().__init__()
        self.stem = ConvNormRelu(4, 4, 3)
        self.stages = torch.nn.ModuleList([ConvNormRelu(4, 4, 3) for _ in range(4)])
        self.head = ConvNormRelu(4, 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the stem, repeated stages, and head."""

        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return self.head(x)


class ConvNormRelu(torch.nn.Module):
    """Fixed conv-batchnorm-relu leaf chain."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        """Initialize the chain."""

        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        self.bn = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run convolution, normalization, and activation."""

        return torch.relu(self.bn(self.conv(x)))


class UniqueMixed(torch.nn.Module):
    """Inception-like module with unique branch internals."""

    def __init__(self, width: int, kernel_size: int) -> None:
        """Initialize parallel branches with varying structure."""

        super().__init__()
        self.a = ConvNormRelu(width, width, 1)
        self.b = torch.nn.Sequential(
            ConvNormRelu(width, width, 1),
            ConvNormRelu(width, width, kernel_size),
        )
        self.c = torch.nn.Sequential(
            torch.nn.AvgPool2d(3, stride=1, padding=1),
            ConvNormRelu(width, width, 1),
        )
        self.project = torch.nn.Conv2d(width * 3, width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the parallel branches and project the concatenation."""

        return self.project(torch.cat([self.a(x), self.b(x), self.c(x)], dim=1))


class UniqueParallelStack(torch.nn.Module):
    """Stack of same-role but structurally non-identical mixed modules."""

    def __init__(self) -> None:
        """Initialize mixed modules with different kernels."""

        super().__init__()
        self.mixed_5b = UniqueMixed(4, 3)
        self.mixed_5c = UniqueMixed(4, 5)
        self.mixed_5d = UniqueMixed(4, 3)
        self.mixed_6a = UniqueMixed(4, 5)
        self.head = torch.nn.Conv2d(4, 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the mixed-module stack."""

        x = self.mixed_5b(x)
        x = self.mixed_5c(x)
        x = self.mixed_5d(x)
        x = self.mixed_6a(x)
        return self.head(x)


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


def _collapsed_label_count(source: str, prefix: str) -> int:
    """Return collapsed module label count with ``prefix``."""

    return source.count(f"<B>@{prefix}")


def _collapsed_exact_label_count(source: str, address: str) -> int:
    """Return collapsed module label count for exactly ``address``."""

    return len(
        [
            line
            for line in source.splitlines()
            if "shape=box3d" in line and f"<B>@{address}</B>" in line
        ]
    )


def _run_fold_ellipsis_name(address: str) -> str:
    """Return the deterministic run-fold ellipsis node name for ``address``."""

    return f"{address}pass1___runfoldellipsis"


def _run_fold_ellipsis_count(source: str, multiplicity: int) -> int:
    """Return count of run-fold ellipsis labels for ``multiplicity`` folded modules."""

    return source.count(f'label="... +{multiplicity - 1} more of this type"')


def _has_run_fold_multiplicity_label(source: str, multiplicity: int) -> bool:
    """Return whether DOT source contains the old run-fold ``xN`` label."""

    return bool(re.search(rf"\bx{multiplicity}\b", source))


def _select_first_stage_unit(module: object) -> bool:
    """Return whether ``module`` is the first unit inside a synthetic stage."""

    return re.match(r"^stages\.\d+\.blocks\.0$", str(getattr(module, "address", ""))) is not None


def _select_blocks_child(module: object) -> bool:
    """Return whether ``module`` is a direct child of a ``blocks`` container."""

    return re.match(r"^blocks\.\d+$", str(getattr(module, "address", ""))) is not None


def _edge_count(source: str, tail: str, head: str) -> int:
    """Return Graphviz edge count between two rendered node names."""

    return len(
        re.findall(
            rf'^\s*"?{re.escape(tail)}"? -> "?{re.escape(head)}"?\s+\[',
            source,
            flags=re.MULTILINE,
        )
    )


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
        assert 4 <= _dot_node_count(auto_source) <= 40

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


def test_auto_collapse_run_fold_collapses_nodes_and_edges(tmp_path: Path) -> None:
    """Auto folds an unreadable consecutive identical run through an ellipsis node."""

    trace = _trace(RepeatedResidual(depth=24), torch.randn(2, 8))
    try:
        auto_source = _draw_source(trace, tmp_path, "run_fold_auto", "auto")
        ellipsis_name = _run_fold_ellipsis_name("blocks.0")

        assert _collapsed_exact_label_count(auto_source, "blocks.0") == 1
        assert _run_fold_ellipsis_count(auto_source, 24) == 1
        assert not _has_run_fold_multiplicity_label(auto_source, 24)
        assert _collapsed_exact_label_count(auto_source, "blocks.1") == 0
        assert _collapsed_exact_label_count(auto_source, "blocks.23") == 0
        assert _edge_count(auto_source, "blocks.0pass1", ellipsis_name) == 1
        assert _edge_count(auto_source, "blocks.0pass1", "blocks.0pass1") == 0
    finally:
        trace.cleanup()


def test_auto_collapse_run_fold_ignores_dimension_steps(tmp_path: Path) -> None:
    """Auto folds structurally identical runs even when channel dimensions vary."""

    trace = _trace(DimStepRun(depth=24, start_width=32), torch.randn(1, 32, 8, 8))
    try:
        auto_source = _draw_source(trace, tmp_path, "dim_step_auto", "auto")
        folds = resolve_run_folds(trace, _select_blocks_child)
        ellipsis_name = _run_fold_ellipsis_name("blocks.1")

        assert _collapsed_exact_label_count(auto_source, "blocks.1") == 1
        assert _run_fold_ellipsis_count(auto_source, 23) == 1
        assert _collapsed_exact_label_count(auto_source, "blocks.23") == 0
        assert "shapes " in auto_source
        assert _edge_count(auto_source, "blocks.1pass1", ellipsis_name) == 1
        assert folds["blocks.1"].addresses == tuple(f"blocks.{index}" for index in range(1, 24))
    finally:
        trace.cleanup()


def test_auto_collapse_run_fold_keeps_different_depth_stages_separate(tmp_path: Path) -> None:
    """Run-fold does not merge same-class sibling stages with different depths."""

    trace = _trace(UnevenDepthStages(depths=(2, 3, 4) * 8), torch.randn(1, 4, 16, 16))
    try:
        auto_source = _draw_source(trace, tmp_path, "uneven_depth_stages_auto", "auto")
        folds = resolve_run_folds(trace, _select_first_stage_unit)

        assert _run_fold_ellipsis_count(auto_source, 3) == 0
        assert "stages.0" not in folds
        assert "stages.1" not in folds
        assert "stages.2" not in folds
    finally:
        trace.cleanup()


def test_auto_collapse_run_fold_rejects_spatial_span() -> None:
    """Run-fold does not create a box across spatial-resolution changes."""

    trace = _trace(SpatialStepRun(), torch.randn(1, 4, 32, 32))
    try:
        folds = resolve_run_folds(trace, _select_blocks_child)

        assert "blocks.0" not in folds
        assert "blocks.1" not in folds
        assert "blocks.2" not in folds
    finally:
        trace.cleanup()


def test_auto_collapse_run_fold_skips_readable_stack(tmp_path: Path) -> None:
    """Run-fold does not elide a stack whose collapsed render is readable."""

    trace = _trace(DimStepRun(depth=12, start_width=16), torch.randn(1, 16, 8, 8))
    try:
        collapse_fn = resolve_collapse_fn(trace, "auto", "unrolled")
        folds = resolve_run_folds(trace, collapse_fn)
        auto_source = _draw_source(trace, tmp_path, "run_fold_readable_auto", "auto")

        assert folds == {}
        assert "runfoldellipsis" not in auto_source
        assert "more of this type" not in auto_source
        assert _collapsed_exact_label_count(auto_source, "blocks.0") == 1
        assert _collapsed_exact_label_count(auto_source, "blocks.11") == 1
    finally:
        trace.cleanup()


def test_auto_collapse_run_fold_fires_when_stack_exceeds_band(tmp_path: Path) -> None:
    """Run-fold elides the longest stack when collapsed render exceeds the band."""

    trace = _trace(RepeatedResidual(depth=24), torch.randn(2, 8))
    try:
        collapse_fn = resolve_collapse_fn(trace, "auto", "unrolled")
        folds = resolve_run_folds(trace, collapse_fn)
        auto_source = _draw_source(trace, tmp_path, "run_fold_over_band_auto", "auto")

        assert folds["blocks.0"].addresses == tuple(f"blocks.{index}" for index in range(24))
        assert _run_fold_ellipsis_count(auto_source, 24) == 1
        assert "... +23 more of this type" in auto_source
    finally:
        trace.cleanup()


def test_auto_collapse_flat_vgg_bn_features_container_folds(tmp_path: Path) -> None:
    """Auto folds a flat VGG-BN-style features Sequential as one container."""

    trace = _trace(VggBnFeatures(), torch.randn(1, 3, 16, 16))
    try:
        auto_source = _draw_source(trace, tmp_path, "vgg_bn_auto", "auto")

        assert _collapsed_exact_label_count(auto_source, "features") == 1
        assert _dot_node_count(auto_source) <= 8
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


def test_max_collapse_is_never_less_collapsed_than_auto(tmp_path: Path) -> None:
    """Max collapse renders no more nodes than auto on representative toy models."""

    cases: list[tuple[str, torch.nn.Module, torch.Tensor]] = [
        ("repeated", RepeatedResidual(depth=5), torch.randn(2, 8)),
        ("parent_residual", ParentJoinResidual(depth=4), torch.randn(2, 8)),
        ("skip_concat", SkipConcatUNet(), torch.randn(1, 4, 16, 16)),
        ("branch_concat", BranchConcat(), torch.randn(1, 4, 16, 16)),
        ("stem_stage", StemStageBackbone(), torch.randn(1, 4, 16, 16)),
        ("recurrent", RecurrentWrapper(), torch.randn(1, 6, 8)),
    ]
    for name, model, x in cases:
        trace = _trace(model, x)
        try:
            auto_source = _draw_source(trace, tmp_path, f"{name}_auto", "auto")
            max_source = _draw_source(trace, tmp_path, f"{name}_max", "max")

            assert _dot_node_count(max_source) <= _dot_node_count(auto_source)
        finally:
            trace.cleanup()


def test_auto_collapse_residual_peer_bodies_keep_parent_joins(tmp_path: Path) -> None:
    """Auto elides repeated residual bodies while keeping add junction nodes visible."""

    trace = _trace(ParentJoinResidual(depth=24), torch.randn(2, 8))
    try:
        none_source = _draw_source(trace, tmp_path, "parent_residual_none", "none")
        auto_source = _draw_source(trace, tmp_path, "parent_residual_auto", "auto")

        assert _dot_node_count(none_source) > 15
        assert _dot_node_count(auto_source) <= int(_dot_node_count(none_source) * 0.7)
        assert _box_count(auto_source) >= 1
        assert _collapsed_exact_label_count(auto_source, "blocks.0") == 1
        assert _run_fold_ellipsis_count(auto_source, 24) == 1
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
        assert _dot_node_count(max_source) <= _dot_node_count(auto_source)
        assert _box_count(auto_source) >= 2
        assert _has_visible_node(auto_source, "cat_")
        assert auto_source != none_source
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


def test_auto_collapse_leaf_blocks_match_stage_grain(tmp_path: Path) -> None:
    """Auto folds standalone leaf blocks at the same grain as stage siblings."""

    trace = _trace(StemStageBackbone(), torch.randn(1, 4, 16, 16))
    try:
        none_source = _draw_source(trace, tmp_path, "stem_stage_none", "none")
        auto_source = _draw_source(trace, tmp_path, "stem_stage_auto", "auto")
        max_source = _draw_source(trace, tmp_path, "stem_stage_max", "max")

        assert _dot_node_count(auto_source) < _dot_node_count(none_source)
        assert _dot_node_count(max_source) <= _dot_node_count(auto_source)
        assert _collapsed_exact_label_count(auto_source, "stem") == 1
        assert _collapsed_label_count(auto_source, "stages.") >= 4
        assert _collapsed_exact_label_count(auto_source, "head") == 1
    finally:
        trace.cleanup()


def test_auto_collapse_descends_into_nested_stage_container(tmp_path: Path) -> None:
    """Auto keeps nested stage boxes instead of swallowing the whole container."""

    trace = _trace(NestedStageBackbone(), torch.randn(1, 4, 16, 16))
    try:
        auto_source = _draw_source(trace, tmp_path, "nested_stage_auto", "auto")
        max_source = _draw_source(trace, tmp_path, "nested_stage_max", "max")

        assert _collapsed_label_count(auto_source, "backbone.") >= 4
        assert "<B>@backbone</B></TD>" not in auto_source
        assert _dot_node_count(max_source) <= _dot_node_count(auto_source)
        assert auto_source != max_source
    finally:
        trace.cleanup()


def test_auto_collapse_groups_structurally_similar_unique_parallel_modules(
    tmp_path: Path,
) -> None:
    """Auto groups same-role unique mixed modules and avoids node-wall renders."""

    trace = _trace(UniqueParallelStack(), torch.randn(1, 4, 16, 16))
    try:
        none_source = _draw_source(trace, tmp_path, "unique_parallel_none", "none")
        auto_source = _draw_source(trace, tmp_path, "unique_parallel_auto", "auto")

        assert _dot_node_count(none_source) > 40
        assert _collapsed_label_count(auto_source, "mixed_") >= 4
        assert _dot_node_count(auto_source) < 30
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
