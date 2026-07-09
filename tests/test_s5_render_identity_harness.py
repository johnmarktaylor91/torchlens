"""S5 byte-identity tripwire for render DOT and collapse-plan plumbing."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.visualization.collapse_plan import RenderContext
from torchlens.visualization.render_ir import (
    RenderIR,
    RenderIREdge,
    RenderIRCluster,
    build_render_ir,
    projected_antiparallel_endpoint_pairs,
)

tvm = pytest.importorskip("torchvision.models")

_MANIFEST_PATH = Path(__file__).parent / "fixtures" / "s5_render_golden_manifest.json"


class S5TinyBlock(nn.Module):
    """Small MLP block used by the render identity harness."""

    def __init__(self, width: int = 4) -> None:
        """Initialize the block."""

        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the block."""

        return self.fc2(self.relu(self.fc1(x)))


class S5UniformStack(nn.Module):
    """Stack of repeated MLP blocks."""

    def __init__(self, depth: int = 5, width: int = 4) -> None:
        """Initialize the stack."""

        super().__init__()
        self.blocks = nn.ModuleList(S5TinyBlock(width) for _ in range(depth))
        self.out = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the stack."""

        for block in self.blocks:
            x = block(x)
        return self.out(x)


class S5ResidualBlock(nn.Module):
    """Small residual MLP block."""

    def __init__(self, width: int = 4) -> None:
        """Initialize the block."""

        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the block."""

        return x + self.net(x)


class S5RepeatedResidual(nn.Module):
    """Repeated residual fixture."""

    def __init__(self, depth: int = 4, width: int = 4) -> None:
        """Initialize the fixture."""

        super().__init__()
        self.blocks = nn.Sequential(*(S5ResidualBlock(width) for _ in range(depth)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the fixture."""

        return self.blocks(x)


class S5ConvPlateauBlock(nn.Module):
    """Depthwise-style residual convolution block."""

    def __init__(self, channels: int = 4) -> None:
        """Initialize the block."""

        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the block."""

        return x + self.net(x)


class S5ConvPlateau(nn.Module):
    """Synthetic MobileNet-like plateau fixture."""

    def __init__(self) -> None:
        """Initialize the fixture."""

        super().__init__()
        self.stem = nn.Conv2d(3, 4, 1)
        self.features = nn.ModuleList(S5ConvPlateauBlock(4) for _ in range(4))
        self.head = nn.Conv2d(4, 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the fixture."""

        x = self.stem(x)
        for block in self.features:
            x = block(x)
        return self.head(x)


def _sha256_text(text: str) -> str:
    """Return the SHA-256 digest of a text value."""

    return hashlib.sha256(text.encode()).hexdigest()


def _manifest() -> dict[str, Any]:
    """Load the checked-in S5 golden manifest."""

    return json.loads(_MANIFEST_PATH.read_text())


def _cases() -> dict[str, tuple[Callable[[], nn.Module], torch.Tensor]]:
    """Return deterministic trace fixtures for the identity harness."""

    return {
        "uniform_stack": (S5UniformStack, torch.randn(2, 4)),
        "repeated_residual": (S5RepeatedResidual, torch.randn(2, 4)),
        "conv_plateau": (S5ConvPlateau, torch.randn(1, 3, 8, 8)),
        "torchvision_resnet18": (lambda: tvm.resnet18(weights=None), torch.randn(1, 3, 64, 64)),
    }


def _snapshot_case(trace: tl.Trace, tmp_path: Path, case_name: str) -> dict[str, dict[str, Any]]:
    """Render one trace across the S5 identity modes."""

    items: dict[str, dict[str, Any]] = {}
    for collapse in ("none", "auto", "max"):
        for fold_runs in (None, True, False):
            if collapse == "none" and fold_runs is None:
                key = "none_default"
            elif collapse == "none":
                key = f"none_fold_{fold_runs}"
            elif fold_runs is None:
                key = collapse
            else:
                continue
            source = str(
                trace.draw(
                    vis_outpath=str(tmp_path / f"{case_name}_{key}"),
                    vis_save_only=True,
                    vis_fileformat="svg",
                    vis_node_placement="dot",
                    collapse=collapse,
                    fold_runs=fold_runs,
                    order_siblings=False,
                )
            )
            items[key] = {"dot_len": len(source), "dot_sha256": _sha256_text(source)}
            if collapse in {"auto", "max"}:
                plan_repr = repr(trace.collapse_plan(mode=collapse))
                items[key]["plan_repr_len"] = len(plan_repr)
                items[key]["plan_sha256"] = _sha256_text(plan_repr)
    return items


def _assert_ir_shape(trace: tl.Trace) -> None:
    """Assert the render-IR adapter exposes resolved node data."""

    render_ir = build_render_ir(
        trace,
        collapse_fn=None,
        run_folds=None,
        context=RenderContext(),
    )

    assert isinstance(render_ir, RenderIR)
    assert render_ir.nodes
    assert render_ir.edges
    assert render_ir.clusters
    assert len(render_ir.nodes) == len(render_ir.node_emissions)
    assert all(node.name for node in render_ir.nodes)
    assert all(isinstance(edge, RenderIREdge) for edge in render_ir.edges)
    assert all(isinstance(cluster, RenderIRCluster) for cluster in render_ir.clusters)
    assert {node.kind for node in render_ir.nodes}.issuperset({"raw_op"})
    assert {edge.projection_reason for edge in render_ir.edges}.issuperset({"direct"})


def test_s5_projected_antiparallel_pairs_require_projected_edges() -> None:
    """Projected anti-parallel styling is driven by IR projection reasons."""

    render_ir = RenderIR(
        context=RenderContext(),
        nodes=(),
        edges=(
            RenderIREdge(
                source_unit="fold_a",
                target_unit="fold_b",
                source_originals=("a_1",),
                target_originals=("b_1",),
                owner_cluster=None,
                occurrence_key=("edge", "a_1", "b_1"),
                projection_reason="source_projected",
            ),
            RenderIREdge(
                source_unit="fold_b",
                target_unit="fold_a",
                source_originals=("b_2",),
                target_originals=("a_2",),
                owner_cluster=None,
                occurrence_key=("edge", "b_2", "a_2"),
                projection_reason="both_projected",
            ),
            RenderIREdge(
                source_unit="raw_a",
                target_unit="raw_b",
                source_originals=("raw_a",),
                target_originals=("raw_b",),
                owner_cluster=None,
                occurrence_key=("edge", "raw_a", "raw_b"),
                projection_reason="direct",
            ),
            RenderIREdge(
                source_unit="fold_self",
                target_unit="fold_self",
                source_originals=("self_1",),
                target_originals=("self_2",),
                owner_cluster=None,
                occurrence_key=("edge", "self_1", "self_2"),
                projection_reason="both_projected",
            ),
        ),
        clusters=(),
        node_emissions=(),
    )

    assert projected_antiparallel_endpoint_pairs(render_ir) == frozenset(
        {("fold_a", "fold_b"), ("fold_b", "fold_a")}
    )


@pytest.mark.slow
def test_s5_render_dot_and_plan_hashes_match_manifest(tmp_path: Path) -> None:
    """DOT and collapse-plan hashes stay byte-identical to the S5 manifest."""

    manifest = _manifest()
    actual: dict[str, dict[str, dict[str, Any]]] = {}
    for case_name, (builder, x) in _cases().items():
        with torch.no_grad():
            trace = tl.trace(builder().eval(), x)
        try:
            _assert_ir_shape(trace)
            actual[case_name] = _snapshot_case(trace, tmp_path, case_name)
        finally:
            trace.cleanup()

    assert actual == manifest["cases"]
