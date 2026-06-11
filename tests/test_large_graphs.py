"""Tests for large graph rendering, rank layout, and RandomGraphModel."""

from __future__ import annotations

import os
import warnings
from collections.abc import Iterator
from typing import Any

import pytest
import torch
from torch import nn

from torchlens import trace as trace_fn
from torchlens.user_funcs import validate_forward_pass
from torchlens.visualization._rank_layout_internal import layout as rank_layout
from torchlens.visualization._rank_layout_internal.layout import (
    RANK_LAYOUT_COST_THRESHOLD,
    SPAN_LOCAL,
    _compute_topological_layout,
    compute_rank_depths,
    estimate_rank_layout_cost,
    get_node_placement_engine,
)

from example_models import RandomGraphModel

VIS_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_outputs", "visualizations", "large")


@pytest.fixture(autouse=True)
def _ensure_output_dir() -> Iterator[None]:
    """Create the visualization output directory for tests that render files."""

    os.makedirs(VIS_OUTPUT_DIR, exist_ok=True)
    yield


def _count_nodes(model: nn.Module, x: torch.Tensor) -> int:
    """Log a forward pass and return the number of captured layers."""

    trace = trace_fn(model, x)
    count = len(trace.layer_list)
    trace.cleanup()
    return count


def _draw_model(model: nn.Module, x: torch.Tensor, **kwargs: Any) -> str:
    """Trace a model, draw it, and return the generated DOT source."""

    trace = trace_fn(model, x)
    try:
        return str(trace.draw(**kwargs))
    finally:
        trace.cleanup()


def _chain_edges(num_nodes: int) -> list[tuple[str, str]]:
    """Build local chain edges for estimator tests."""

    return [(f"n{i}", f"n{i + 1}") for i in range(num_nodes - 1)]


def _hub_edges(num_nodes: int, fanout: int) -> list[tuple[str, str]]:
    """Build a chain plus long-range hub edges for estimator tests."""

    edges = _chain_edges(num_nodes)
    step = max(1, num_nodes // fanout)
    edges.extend(("n0", f"n{i}") for i in range(step, num_nodes, step))
    return edges


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

    def test_small_model(self) -> None:
        """A 500-node target lands near the requested scale."""

        model = RandomGraphModel(target_nodes=500, seed=42)
        count = _count_nodes(model, torch.randn(2, 64))
        assert 400 < count < 600, f"Expected ~500 nodes, got {count}"

    @pytest.mark.slow
    def test_3k_nodes(self) -> None:
        """A 3k-node target lands near the requested scale."""

        model = RandomGraphModel(target_nodes=3000, seed=42)
        count = _count_nodes(model, torch.randn(2, 64))
        assert 2500 < count < 3500, f"Expected ~3000 nodes, got {count}"

    @pytest.mark.slow
    def test_5k_nodes(self) -> None:
        """A 5k-node target lands near the requested scale."""

        model = RandomGraphModel(target_nodes=5000, seed=42)
        count = _count_nodes(model, torch.randn(2, 64))
        assert 4500 < count < 5500, f"Expected ~5000 nodes, got {count}"

    @pytest.mark.slow
    def test_10k_nodes(self) -> None:
        """A 10k-node target lands near the requested scale."""

        model = RandomGraphModel(target_nodes=10000, seed=42)
        count = _count_nodes(model, torch.randn(2, 64))
        assert 9000 < count < 11000, f"Expected ~10000 nodes, got {count}"

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

    @pytest.mark.slow
    @pytest.mark.rare
    def test_100k_nodes(self) -> None:
        """A 100k-node target lands near the requested scale."""

        model = RandomGraphModel(target_nodes=100000, seed=42)
        count = _count_nodes(model, torch.randn(2, 64))
        assert 90000 < count < 110000, f"Expected ~100000 nodes, got {count}"

    @pytest.mark.slow
    @pytest.mark.rare
    def test_250k_nodes(self) -> None:
        """A 250k-node target lands near the requested scale."""

        model = RandomGraphModel(target_nodes=250000, seed=42)
        count = _count_nodes(model, torch.randn(2, 64))
        assert 225000 < count < 275000, f"Expected ~250000 nodes, got {count}"

    @pytest.mark.slow
    @pytest.mark.rare
    def test_1m_nodes(self) -> None:
        """A 1M-node target lands near the requested scale."""

        model = RandomGraphModel(target_nodes=1000000, seed=42)
        count = _count_nodes(model, torch.randn(2, 64))
        assert 900000 < count < 1100000, f"Expected ~1000000 nodes, got {count}"

    def test_deterministic(self) -> None:
        """Same seed produces the same structure."""

        m1 = RandomGraphModel(target_nodes=1000, seed=123)
        m2 = RandomGraphModel(target_nodes=1000, seed=123)
        assert sum(p.numel() for p in m1.parameters()) == sum(p.numel() for p in m2.parameters())

    def test_different_seeds(self) -> None:
        """Different seeds produce different structures."""

        m1 = RandomGraphModel(target_nodes=1000, seed=1)
        m2 = RandomGraphModel(target_nodes=1000, seed=2)
        assert sum(p.numel() for p in m1.parameters()) != sum(p.numel() for p in m2.parameters())

    def test_call_depth(self) -> None:
        """Module nesting creates expected hierarchy."""

        model = RandomGraphModel(target_nodes=500, call_depth=4, seed=42)
        trace = trace_fn(model, torch.randn(2, 64))
        max_depth = max(len(trace[label].modules) for label in trace.layer_labels)
        assert max_depth >= 3
        trace.cleanup()

    def test_validation_small(self) -> None:
        """Validation succeeds for a small random model."""

        model = RandomGraphModel(target_nodes=200, seed=42)
        assert validate_forward_pass(model, torch.randn(2, 64))

    @pytest.mark.slow
    def test_validation_1k(self) -> None:
        """Validation succeeds for a 1k-node random model."""

        model = RandomGraphModel(target_nodes=1000, seed=42)
        assert validate_forward_pass(model, torch.randn(2, 64))

    @pytest.mark.slow
    def test_validation_3k(self) -> None:
        """Validation succeeds for a 3k-node random model."""

        model = RandomGraphModel(target_nodes=3000, seed=42)
        assert validate_forward_pass(model, torch.randn(2, 64))

    @pytest.mark.slow
    def test_validation_5k(self) -> None:
        """Validation succeeds for a 5k-node random model."""

        model = RandomGraphModel(target_nodes=5000, seed=42)
        assert validate_forward_pass(model, torch.randn(2, 64))

    @pytest.mark.slow
    def test_validation_10k(self) -> None:
        """Validation succeeds for a 10k-node random model."""

        model = RandomGraphModel(target_nodes=10000, seed=42)
        assert validate_forward_pass(model, torch.randn(2, 64))

    @pytest.mark.slow
    @pytest.mark.rare
    def test_validation_20k(self) -> None:
        """Validation succeeds for a 20k-node random model."""

        model = RandomGraphModel(target_nodes=20000, seed=42)
        assert validate_forward_pass(model, torch.randn(2, 64))

    @pytest.mark.slow
    @pytest.mark.rare
    def test_validation_50k(self) -> None:
        """Validation succeeds for a 50k-node random model."""

        model = RandomGraphModel(target_nodes=50000, seed=42)
        assert validate_forward_pass(model, torch.randn(2, 64))

    @pytest.mark.slow
    @pytest.mark.rare
    def test_validation_100k(self) -> None:
        """Validation succeeds for a 100k-node random model."""

        model = RandomGraphModel(target_nodes=100000, seed=42)
        assert validate_forward_pass(model, torch.randn(2, 64))

    @pytest.mark.skip(
        reason="250k-node validation OOMs / hangs for hours on most machines. "
        "Run manually with: pytest tests/test_large_graphs.py::"
        "TestRandomGraphModel::test_validation_250k --no-skip"
    )
    @pytest.mark.slow
    @pytest.mark.rare
    def test_validation_250k(self) -> None:
        """Validation for 250k-node random models is manual-only."""

        model = RandomGraphModel(target_nodes=250000, seed=42)
        assert validate_forward_pass(model, torch.randn(2, 64))


class TestRankLayoutUtilities:
    """Test rank layout helpers without invoking Graphviz."""

    def test_depths_for_chain(self) -> None:
        """A local chain gets monotonically increasing depths."""

        labels = {f"n{i}" for i in range(5)}
        depths = compute_rank_depths(labels, _chain_edges(5))
        assert depths == {"n0": 0, "n1": 1, "n2": 2, "n3": 3, "n4": 4}

    def test_local_topology_stays_under_threshold(self) -> None:
        """The measured local 5k topology stays on dot with the calibrated threshold."""

        labels = {f"n{i}" for i in range(5000)}
        cost = estimate_rank_layout_cost(labels, _chain_edges(5000))
        assert cost == 5000
        assert SPAN_LOCAL == 12
        assert cost < RANK_LAYOUT_COST_THRESHOLD
        assert get_node_placement_engine("auto", cost) == "dot"

    def test_hub_topology_exceeds_threshold(self) -> None:
        """Long hub edges exceed the threshold even at moderate node counts."""

        labels = {f"n{i}" for i in range(3500)}
        cost = estimate_rank_layout_cost(labels, _hub_edges(3500, fanout=24))
        assert cost > RANK_LAYOUT_COST_THRESHOLD
        assert get_node_placement_engine("auto", cost) == "rank"

    def test_manual_engines_force_selection(self) -> None:
        """Explicit dot and rank choices bypass auto cost selection."""

        assert get_node_placement_engine("dot", RANK_LAYOUT_COST_THRESHOLD + 1) == "dot"
        assert get_node_placement_engine("rank", 1) == "rank"

    def test_topological_layout_returns_module_boxes(self) -> None:
        """The Kahn layout returns positions and compound boxes for modules."""

        node_data = {
            "a": {"attrs": {"label": "A"}, "node_label": "a"},
            "b": {"attrs": {"label": "B"}, "node_label": "b"},
        }
        edges = [{"tail_name": "a", "head_name": "b"}]
        sizes = {"a": (100.0, 40.0), "b": (100.0, 40.0)}
        positions, compound_bboxes, max_y = _compute_topological_layout(
            node_data,
            edges,
            sizes,
            {"module": ["a", "b"]},
            {},
        )
        assert set(positions) == {"a", "b"}
        assert "group_module" in compound_bboxes
        assert max_y > 0


class TestRankLayoutScale:
    """Exercise the kept rank layout at large synthetic scales."""

    @pytest.mark.slow
    @pytest.mark.parametrize("num_nodes", [3000, 5000])
    def test_rank_layout_scale_smoke(self, num_nodes: int) -> None:
        """Rank layout computes positions for common large-graph sizes."""

        node_data = _synthetic_node_data(num_nodes)
        edges = _synthetic_rank_edges(num_nodes)
        sizes = {f"n{i}": (20.0, 10.0) for i in range(num_nodes)}
        positions, compound_bboxes, max_y = _compute_topological_layout(
            node_data,
            edges,
            sizes,
            {},
            {},
        )
        assert len(positions) == num_nodes
        assert compound_bboxes == {}
        assert max_y > 0

    @pytest.mark.slow
    @pytest.mark.rare
    @pytest.mark.parametrize("num_nodes", [10000, 20000, 50000, 100000, 250000, 1000000])
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

    def test_auto_local_topology_renders_with_dot(self, tmp_path: Any) -> None:
        """A local small graph renders through dot without the rank notice."""

        model = RandomGraphModel(target_nodes=80, seed=42)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            source = _draw_model(
                model,
                torch.randn(2, 64),
                vis_node_placement="auto",
                vis_save_only=True,
                vis_fileformat="svg",
                vis_outpath=str(tmp_path / "auto_dot"),
            )
        assert "digraph" in source
        assert not any("auto-selected rank layout" in str(w.message) for w in caught)

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
                vis_node_placement="auto",
                vis_save_only=True,
                vis_fileformat="svg",
                vis_outpath=str(tmp_path / "auto_rank"),
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
        source = _draw_model(
            model,
            torch.randn(2, 64),
            vis_node_placement="dot",
            vis_save_only=True,
            vis_fileformat="svg",
            vis_outpath=str(tmp_path / "manual_dot"),
        )
        assert "digraph" in source

    def test_manual_rank_forces_rank(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Any,
    ) -> None:
        """Explicit rank calls the rank renderer without an auto notice."""

        calls: list[str] = []

        def fake_render_rank_layout(*args: Any, **kwargs: Any) -> str:
            """Record rank renderer use without invoking neato."""

            calls.append("rank")
            return "digraph { manual_rank }"

        monkeypatch.setattr(rank_layout, "render_rank_layout", fake_render_rank_layout)
        model = RandomGraphModel(target_nodes=60, seed=42)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            source = _draw_model(
                model,
                torch.randn(2, 64),
                vis_node_placement="rank",
                vis_save_only=True,
                vis_fileformat="svg",
                vis_outpath=str(tmp_path / "manual_rank"),
            )
        assert calls == ["rank"]
        assert source == "digraph { manual_rank }"
        assert not any("auto-selected rank layout" in str(w.message) for w in caught)

    def test_removed_engine_values_are_rejected(self, tmp_path: Any) -> None:
        """The retired 'elk' and 'sfdp' engine values raise like any bad value."""

        model = RandomGraphModel(target_nodes=60, seed=42)
        for removed in ("elk", "sfdp"):
            with pytest.raises(ValueError, match="'auto', 'dot', or 'rank'"):
                _draw_model(
                    model,
                    torch.randn(2, 64),
                    vis_node_placement=removed,
                    vis_save_only=True,
                    vis_fileformat="svg",
                    vis_outpath=str(tmp_path / f"removed_{removed}"),
                )

    def test_rank_svg_contains_labels_and_module_boxes(self, tmp_path: Any) -> None:
        """The rank renderer writes an SVG with node labels and module boxes."""

        model = RandomGraphModel(target_nodes=80, call_depth=2, seed=42)
        outpath = tmp_path / "rank_svg"
        source = _draw_model(
            model,
            torch.randn(2, 64),
            vis_node_placement="rank",
            vis_save_only=True,
            vis_fileformat="svg",
            vis_outpath=str(outpath),
        )
        svg = outpath.with_suffix(".svg").read_text(encoding="utf-8")
        assert "input" in svg
        assert "output" in svg
        assert "cluster_" in source
        assert "@layers" in source

    def test_vis_node_placement_forwarded(self, tmp_path: Any) -> None:
        """vis_node_placement parameter reaches Trace.draw."""

        model = RandomGraphModel(target_nodes=80, seed=42)
        trace = trace_fn(model, torch.randn(2, 64))
        source = trace.draw(
            vis_mode="unrolled",
            vis_save_only=True,
            vis_fileformat="svg",
            vis_outpath=str(tmp_path / "placement_test"),
            vis_node_placement="dot",
        )
        assert "digraph" in source
        trace.cleanup()


class TestDotThresholdBenchmark:
    """Manual benchmark for recalibrating the rank-layout threshold."""

    @pytest.mark.slow
    def test_benchmark_dot_scaling(self, tmp_path: Any) -> None:
        """Time graphviz dot at representative local-topology node counts."""

        import time

        for target in [500, 1000, 2000, 3000, 3500, 4000]:
            model = RandomGraphModel(target_nodes=target, seed=42)
            x = torch.randn(2, 64)
            trace = trace_fn(model, x)
            actual_nodes = len(trace.layer_labels)
            start = time.time()
            try:
                trace.draw(
                    vis_mode="unrolled",
                    vis_save_only=True,
                    vis_outpath=str(tmp_path / f"bench_{target}"),
                    vis_node_placement="dot",
                )
                elapsed = time.time() - start
                print(f"  {target} target ({actual_nodes} actual): {elapsed:.1f}s")
            except Exception as exc:
                elapsed = time.time() - start
                print(
                    f"  {target} target ({actual_nodes} actual): FAILED "
                    f"after {elapsed:.1f}s - {exc}"
                )
            trace.cleanup()
