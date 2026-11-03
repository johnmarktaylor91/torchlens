"""Tests for large graph rendering, rank layout, and RandomGraphModel."""

from __future__ import annotations

import os
import re
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
import torch
from example_models import RandomGraphModel
from torch import nn

from torchlens import trace as trace_fn
from torchlens._user_funcs import validate_forward_pass
from torchlens._visualization._rank_layout_internal import layout as rank_layout
from torchlens._visualization._rank_layout_internal.layout import (
    _choose_spline_mode,
    _compute_topological_layout,
    _rescale_dot_for_rtree,
)
from torchlens.errors._base import TorchLensWarning

# Session-private output root (conftest exports it before collection); the
# repo-local fallback raced across parallel worktree lanes (b2p2 R77).
VIS_OUTPUT_DIR = os.path.join(
    os.environ.get(
        "TORCHLENS_TEST_OUTPUTS_DIR",
        os.path.join(os.path.dirname(__file__), "generated_outputs"),
    ),
    "visualizations",
    "large",
)


# Graphs of 100k+ nodes take hours or exhaust memory on ordinary machines and
# CI runners, so they run only on explicit opt-in. The dispatched
# ``large-graphs.yml`` workflow sets the variable, with its own job timeout.
LARGE_GRAPH_OPT_IN_ENV = "TORCHLENS_RUN_LARGE_GRAPHS"
requires_large_graph_opt_in = pytest.mark.skipif(
    os.environ.get(LARGE_GRAPH_OPT_IN_ENV) != "1",
    reason=(
        "100k+ node graphs run only on explicit opt-in: "
        f"{LARGE_GRAPH_OPT_IN_ENV}=1 pytest tests/test_large_graphs.py -m rare "
        "(CI: the dispatched large-graphs workflow)"
    ),
)


@pytest.fixture(autouse=True)
def _ensure_output_dir() -> Iterator[None]:
    """Create the visualization output directory for tests that render files."""

    os.makedirs(VIS_OUTPUT_DIR, exist_ok=True)
    yield


@contextmanager
def _tolerate_op_count_disclosure() -> Iterator[None]:
    """Allow the intentional >=50k-op capture disclosure (R60-1) to pass.

    These tests deliberately build >=50k-op ``RandomGraphModel`` instances to
    exercise TorchLens at scale, so the capture-side op-count disclosure
    (``torchlens/_ir/capture_events.py``, fired once per journal at
    ``OP_COUNT_DISCLOSURE_THRESHOLD``) is expected here, not a regression;
    the default ``error::UserWarning:torchlens`` filter would otherwise turn
    it into a test failure. ``tests/test_scaling_ladder.py`` owns the
    dedicated regression coverage for the disclosure itself.
    """

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "default",
            message=r"TorchLens has recorded \d+ ops in this forward.*",
            category=TorchLensWarning,
        )
        yield


def _count_nodes(model: nn.Module, x: torch.Tensor) -> int:
    """Log a forward pass and return the number of captured layers."""

    with _tolerate_op_count_disclosure():
        trace = trace_fn(model, x)
    count = len(trace._layer_list)
    trace.cleanup()
    return count


def _draw_model(model: nn.Module, x: torch.Tensor, **kwargs: Any) -> str:
    """Trace a model, draw it, and return the generated DOT source."""

    trace = trace_fn(model, x)
    try:
        return str(trace.draw(**kwargs).dot)
    finally:
        trace.cleanup()


def _chain_edges(num_nodes: int) -> list[tuple[str, str]]:
    """Build local chain edges for estimator tests."""

    return [(f"n{i}", f"n{i + 1}") for i in range(num_nodes - 1)]


def _synthetic_node_data(num_nodes: int) -> dict[str, dict[str, Any]]:
    """Build minimal node data for rank-layout scale tests."""

    return {
        f"n{i}": {
            "attrs": {"label": f"n{i}"},
            "node_label": f"n{i}",
        }
        for i in range(num_nodes)
    }


def _synthetic_rank_edges(num_nodes: int) -> list[dict[str, str]]:
    """Build minimal rank-layout edge dictionaries for a chain."""

    return [
        {"tail_name": source, "head_name": target} for source, target in _chain_edges(num_nodes)
    ]


class TestRandomGraphModel:
    """Verify the random model generator produces approximately correct node counts."""

    @pytest.mark.slow
    @pytest.mark.rare
    def test_20k_nodes(self) -> None:
        """A 20k-node target lands near the requested scale."""

        model = RandomGraphModel(target_nodes=20000, seed=42)
        count = _count_nodes(model, torch.randn(2, 64))
        assert 18000 < count < 22000, f"Expected ~20000 nodes, got {count}"

    @pytest.mark.slow
    @pytest.mark.rare
    def test_50k_nodes(self) -> None:
        """A 50k-node target lands near the requested scale."""

        model = RandomGraphModel(target_nodes=50000, seed=42)
        count = _count_nodes(model, torch.randn(2, 64))
        assert 45000 < count < 55000, f"Expected ~50000 nodes, got {count}"

    @requires_large_graph_opt_in
    @pytest.mark.slow
    @pytest.mark.rare
    def test_100k_nodes(self) -> None:
        """A 100k-node target lands near the requested scale."""

        model = RandomGraphModel(target_nodes=100000, seed=42)
        count = _count_nodes(model, torch.randn(2, 64))
        assert 90000 < count < 110000, f"Expected ~100000 nodes, got {count}"

    @requires_large_graph_opt_in
    @pytest.mark.slow
    @pytest.mark.rare
    def test_250k_nodes(self) -> None:
        """A 250k-node target lands near the requested scale."""

        model = RandomGraphModel(target_nodes=250000, seed=42)
        count = _count_nodes(model, torch.randn(2, 64))
        assert 225000 < count < 275000, f"Expected ~250000 nodes, got {count}"

    @requires_large_graph_opt_in
    @pytest.mark.slow
    @pytest.mark.rare
    def test_1m_nodes(self) -> None:
        """A 1M-node target lands near the requested scale."""

        model = RandomGraphModel(target_nodes=1000000, seed=42)
        count = _count_nodes(model, torch.randn(2, 64))
        assert 900000 < count < 1100000, f"Expected ~1000000 nodes, got {count}"

    @pytest.mark.slow
    @pytest.mark.rare
    def test_validation_20k(self) -> None:
        """Validation succeeds for a 20k-node random model."""

        model = RandomGraphModel(target_nodes=20000, seed=42)
        assert validate_forward_pass(model, torch.randn(2, 64))

    @pytest.mark.slow
    @pytest.mark.rare
    # Peak RSS passed 15 GB within 13 minutes on a worker (still climbing), so the
    # 16 GB GitHub runner is shut down mid-test (Weekly 37192153842, 2026-10-04).
    @pytest.mark.big_memory
    def test_validation_50k(self) -> None:
        """Validation succeeds for a 50k-node random model."""

        model = RandomGraphModel(target_nodes=50000, seed=42)
        with _tolerate_op_count_disclosure():
            assert validate_forward_pass(model, torch.randn(2, 64))

    @requires_large_graph_opt_in
    @pytest.mark.slow
    @pytest.mark.rare
    def test_validation_100k(self) -> None:
        """Validation succeeds for a 100k-node random model."""

        model = RandomGraphModel(target_nodes=100000, seed=42)
        with _tolerate_op_count_disclosure():
            assert validate_forward_pass(model, torch.randn(2, 64))

    @pytest.mark.skipif(
        os.environ.get(LARGE_GRAPH_OPT_IN_ENV) != "1"
        and os.environ.get("TORCHLENS_RUN_250K") != "1",
        reason="250k-node validation OOMs / hangs for hours on most machines. "
        f"Opt in explicitly with: {LARGE_GRAPH_OPT_IN_ENV}=1 (or the legacy TORCHLENS_RUN_250K=1) "
        "pytest tests/test_large_graphs.py::TestRandomGraphModel::test_validation_250k -m rare",
    )
    @pytest.mark.slow
    @pytest.mark.rare
    def test_validation_250k(self) -> None:
        """Validation for 250k-node random models is manual-only."""

        model = RandomGraphModel(target_nodes=250000, seed=42)
        with _tolerate_op_count_disclosure():
            assert validate_forward_pass(model, torch.randn(2, 64))


class TestRankLayoutScale:
    """Exercise the kept rank layout at large synthetic scales."""

    @pytest.mark.slow
    @pytest.mark.rare
    @pytest.mark.parametrize(
        "num_nodes",
        [
            10000,
            20000,
            50000,
            pytest.param(100000, marks=requires_large_graph_opt_in),
            pytest.param(250000, marks=requires_large_graph_opt_in),
            pytest.param(1000000, marks=requires_large_graph_opt_in),
        ],
    )
    def test_rank_layout_scale_rare(self, num_nodes: int) -> None:
        """Rank layout scale ladder for expensive manual runs."""

        node_data = _synthetic_node_data(num_nodes)
        edges = _synthetic_rank_edges(num_nodes)
        sizes = {f"n{i}": (20.0, 10.0) for i in range(num_nodes)}
        positions, _, max_y = _compute_topological_layout(node_data, edges, sizes, {}, {})
        assert len(positions) == num_nodes
        assert max_y > 0


class TestRankEngineRendering:
    """Test renderer integration for dot, rank, auto, and legacy values."""

    def test_auto_hub_topology_selects_rank(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Any,
    ) -> None:
        """A high-cost auto render selects rank and emits the selection notice."""

        calls: list[str] = []

        def fake_render_rank_layout(*args: Any, **kwargs: Any) -> str:
            """Record rank renderer use without invoking neato."""

            calls.append("rank")
            return "digraph { rank_path }"

        monkeypatch.setattr(rank_layout, "RANK_LAYOUT_COST_THRESHOLD", 10)
        monkeypatch.setattr(
            rank_layout,
            "render_rank_layout",
            fake_render_rank_layout,
        )
        model = RandomGraphModel(target_nodes=80, seed=42)
        with pytest.warns(UserWarning, match="auto-selected rank layout"):
            source = _draw_model(
                model,
                torch.randn(2, 64),
                layout="auto",
                file_format="svg",
                path=str(tmp_path / "auto_rank"),
            )
        assert calls == ["rank"]
        assert source == "digraph { rank_path }"

    def test_manual_dot_forces_dot(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Any,
    ) -> None:
        """Explicit dot does not call the rank renderer even under a low threshold."""

        def fail_render_rank_layout(*args: Any, **kwargs: Any) -> str:
            """Fail if explicit dot accidentally reaches the rank renderer."""

            raise AssertionError("rank renderer should not be called")

        monkeypatch.setattr(rank_layout, "RANK_LAYOUT_COST_THRESHOLD", 0)
        monkeypatch.setattr(rank_layout, "render_rank_layout", fail_render_rank_layout)
        model = RandomGraphModel(target_nodes=60, seed=42)
        # A zero threshold also prices sibling-order verification out of budget,
        # so explicit dot skips that post-pass with its documented notice.
        with pytest.warns(UserWarning, match="skipped sibling-order verification"):
            source = _draw_model(
                model,
                torch.randn(2, 64),
                layout="dot",
                file_format="svg",
                path=str(tmp_path / "manual_dot"),
            )
        assert "digraph" in source

    def test_removed_engine_values_are_rejected(self, tmp_path: Any) -> None:
        """The retired 'elk' and 'sfdp' engine values raise like any bad value."""

        model = RandomGraphModel(target_nodes=60, seed=42)
        for removed in ("elk", "sfdp"):
            with pytest.raises(ValueError, match="'auto', 'dot', or 'rank'"):
                _draw_model(
                    model,
                    torch.randn(2, 64),
                    layout=removed,
                    file_format="svg",
                    path=str(tmp_path / f"removed_{removed}"),
                )

    def test_rank_svg_contains_labels_and_module_boxes(self, tmp_path: Any) -> None:
        """The rank renderer writes an SVG with node labels and module boxes."""

        model = RandomGraphModel(target_nodes=80, call_depth=2, seed=42)
        outpath = tmp_path / "rank_svg"
        source = _draw_model(
            model,
            torch.randn(2, 64),
            layout="rank",
            file_format="svg",
            path=str(outpath),
        )
        svg = outpath.with_suffix(".svg").read_text(encoding="utf-8")
        assert "input" in svg
        assert "output" in svg
        assert "cluster_" in source
        assert "@layers" in source


class TestSplineModeHeuristic:
    """Unit tests for the neato spline-mode degradation heuristic.

    Regression for the dense-graph spline-routing blowup (convnextv2_huge:
    785 nodes but high edge density timed out at 120s under -Gsplines=true).
    The heuristic must keep pretty splines for small/sparse graphs (the common
    case) and degrade to straight lines for large OR edge-dense graphs.
    """

    def test_repro_models_all_degrade_to_line(self) -> None:
        """The three real repro graphs (real node/edge counts) all use line.

        Calibration anchors (2026-06-20). These MUST NOT regress to "true":
          convnextv2_huge      785 nodes /  928 edges (node-count gate)
          smp_Unet_densenet201 772 nodes / 2578 edges (node-count + density gate)
          maxvit_xlarge_tf_512 2359 nodes / 2598 edges (node-count gate)
        """

        assert _choose_spline_mode(num_nodes=785, num_edges=928) == "line"
        assert _choose_spline_mode(num_nodes=772, num_edges=2578) == "line"
        assert _choose_spline_mode(num_nodes=2359, num_edges=2598) == "line"

    def test_dense_midsize_graph_degrades_to_line(self) -> None:
        """An edge-dense mid-size graph (below the node ceiling) uses line.

        This guards the density gate independently of the node-count gate: a
        DenseNet-style fan-out at ~500 nodes is too crossing-heavy for splines.
        """

        # 500 nodes, ratio 1.8 (> density ratio) -> "line".
        assert _choose_spline_mode(num_nodes=500, num_edges=900) == "line"


class TestRtreeRescale:
    """Unit tests for the rtree-overflow coordinate down-scaling mitigation.

    Regression for maxvit_xlarge_tf_512: a 2747-node @512px layout overflowed
    neato's rtree coordinate range ("area too large for rtree"). We rescale the
    pinned coordinates so the canvas fits, preserving geometry.
    """

    def test_below_ceiling_returns_none(self) -> None:
        """A layout already within range needs no rescale."""

        dot = 'a [pos="100,200!" width=1.0 height=0.5]'
        assert _rescale_dot_for_rtree(dot, ceiling=28000.0) is None

    def test_over_ceiling_scales_uniformly(self) -> None:
        """An over-range layout is shrunk so max coord == ceiling, preserving ratio."""

        dot = 'a [pos="50000,10000!" width=2.0 height=1.0]\nsubgraph c { bb="0,0,50000,10000" }'
        out = _rescale_dot_for_rtree(dot, ceiling=28000.0)
        assert out is not None
        coords = [
            (float(m.group(1)), float(m.group(2)))
            for m in re.finditer(r'pos="(-?[\d.]+),(-?[\d.]+)!"', out)
        ]
        max_coord = max(max(abs(x), abs(y)) for x, y in coords)
        assert max_coord <= 28000.0 + 1e-3
        # Uniform scale: the x/y ratio (5:1) is preserved.
        x, y = coords[0]
        assert abs(x / y - 5.0) < 1e-6
        # bb box and node dims are scaled too.
        assert 'bb="0.0,0.0,28000.0,5600.0"' in out
        assert "width=1.1200" in out


class TestDotThresholdBenchmark:
    """Manual benchmark for recalibrating the rank-layout threshold."""

    @pytest.mark.slow
    def test_benchmark_dot_scaling(self, tmp_path: Any) -> None:
        """Time graphviz dot at representative local-topology node counts."""

        import time

        failures: list[str] = []
        for target in [500, 1000, 2000, 3000, 3500, 4000]:
            model = RandomGraphModel(target_nodes=target, seed=42)
            x = torch.randn(2, 64)
            trace = trace_fn(model, x)
            actual_nodes = len(trace.layer_labels)
            start = time.time()
            try:
                trace.draw(
                    rolled=False,
                    path=str(tmp_path / f"bench_{target}"),
                    layout="dot",
                )
                elapsed = time.time() - start
                print(f"  {target} target ({actual_nodes} actual): {elapsed:.1f}s")
            except Exception as exc:
                elapsed = time.time() - start
                failures.append(f"{target} target ({actual_nodes} actual): {exc}")
                print(
                    f"  {target} target ({actual_nodes} actual): FAILED "
                    f"after {elapsed:.1f}s - {exc}"
                )
            trace.cleanup()
        assert failures == [], f"dot benchmark render failures: {failures}"
