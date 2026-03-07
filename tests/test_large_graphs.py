"""Tests for large graph rendering and the RandomGraphModel generator."""

import os

import pytest
import torch

from torchlens import log_forward_pass, show_model_graph
from torchlens.user_funcs import validate_forward_pass
from torchlens.visualization.elk_layout import (
    elk_available,
    get_node_placement_engine,
    build_elk_graph,
    build_elk_graph_hierarchical,
    inject_elk_positions,
    _ELK_NODE_THRESHOLD,
)

from tests.example_models import RandomGraphModel

VIS_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_outputs", "visualizations", "large")


@pytest.fixture(autouse=True)
def _ensure_output_dir():
    os.makedirs(VIS_OUTPUT_DIR, exist_ok=True)


def _count_nodes(model, x):
    """Log forward pass and return node count."""
    ml = log_forward_pass(model, x)
    count = len(ml.layer_labels)
    ml.cleanup()
    return count


# -----------------------------------------------
# RandomGraphModel tests
# -----------------------------------------------


class TestRandomGraphModel:
    """Verify the random model generator produces approximately correct node counts."""

    def test_small_model(self):
        model = RandomGraphModel(target_nodes=500, seed=42)
        x = torch.randn(2, 64)
        count = _count_nodes(model, x)
        assert 400 < count < 600, f"Expected ~500 nodes, got {count}"

    @pytest.mark.slow
    def test_3k_nodes(self):
        model = RandomGraphModel(target_nodes=3000, seed=42)
        x = torch.randn(2, 64)
        count = _count_nodes(model, x)
        assert 2500 < count < 3500, f"Expected ~3000 nodes, got {count}"

    @pytest.mark.slow
    def test_5k_nodes(self):
        model = RandomGraphModel(target_nodes=5000, seed=42)
        x = torch.randn(2, 64)
        count = _count_nodes(model, x)
        assert 4500 < count < 5500, f"Expected ~5000 nodes, got {count}"

    @pytest.mark.slow
    def test_10k_nodes(self):
        model = RandomGraphModel(target_nodes=10000, seed=42)
        x = torch.randn(2, 64)
        count = _count_nodes(model, x)
        assert 9000 < count < 11000, f"Expected ~10000 nodes, got {count}"

    @pytest.mark.slow
    def test_20k_nodes(self):
        model = RandomGraphModel(target_nodes=20000, seed=42)
        x = torch.randn(2, 64)
        count = _count_nodes(model, x)
        assert 18000 < count < 22000, f"Expected ~20000 nodes, got {count}"

    @pytest.mark.slow
    def test_50k_nodes(self):
        model = RandomGraphModel(target_nodes=50000, seed=42)
        x = torch.randn(2, 64)
        count = _count_nodes(model, x)
        assert 45000 < count < 55000, f"Expected ~50000 nodes, got {count}"

    @pytest.mark.slow
    def test_100k_nodes(self):
        model = RandomGraphModel(target_nodes=100000, seed=42)
        x = torch.randn(2, 64)
        count = _count_nodes(model, x)
        assert 90000 < count < 110000, f"Expected ~100000 nodes, got {count}"

    @pytest.mark.slow
    @pytest.mark.rare
    def test_250k_nodes(self):
        model = RandomGraphModel(target_nodes=250000, seed=42)
        x = torch.randn(2, 64)
        count = _count_nodes(model, x)
        assert 225000 < count < 275000, f"Expected ~250000 nodes, got {count}"

    @pytest.mark.slow
    @pytest.mark.rare
    def test_1M_nodes(self):
        model = RandomGraphModel(target_nodes=1000000, seed=42)
        x = torch.randn(2, 64)
        count = _count_nodes(model, x)
        assert 900000 < count < 1100000, f"Expected ~1000000 nodes, got {count}"

    def test_deterministic(self):
        """Same seed produces same structure."""
        m1 = RandomGraphModel(target_nodes=1000, seed=123)
        m2 = RandomGraphModel(target_nodes=1000, seed=123)
        p1 = sum(p.numel() for p in m1.parameters())
        p2 = sum(p.numel() for p in m2.parameters())
        assert p1 == p2

    def test_different_seeds(self):
        """Different seeds produce different structures."""
        m1 = RandomGraphModel(target_nodes=1000, seed=1)
        m2 = RandomGraphModel(target_nodes=1000, seed=2)
        p1 = sum(p.numel() for p in m1.parameters())
        p2 = sum(p.numel() for p in m2.parameters())
        assert p1 != p2

    def test_nesting_depth(self):
        """Module nesting creates expected hierarchy."""
        model = RandomGraphModel(target_nodes=500, nesting_depth=4, seed=42)
        ml = log_forward_pass(model, torch.randn(2, 64))
        max_depth = max(
            len(ml[label].containing_modules_origin_nested) for label in ml.layer_labels
        )
        assert max_depth >= 3
        ml.cleanup()

    def test_validation_small(self):
        """Validation passes for small random model."""
        model = RandomGraphModel(target_nodes=200, seed=42)
        assert validate_forward_pass(model, torch.randn(2, 64))

    @pytest.mark.slow
    def test_validation_1k(self):
        """Validation passes for 1k-node random model."""
        model = RandomGraphModel(target_nodes=1000, seed=42)
        assert validate_forward_pass(model, torch.randn(2, 64))

    @pytest.mark.slow
    def test_validation_3k(self):
        """Validation passes for 3k-node random model."""
        model = RandomGraphModel(target_nodes=3000, seed=42)
        assert validate_forward_pass(model, torch.randn(2, 64))

    @pytest.mark.slow
    def test_validation_5k(self):
        """Validation passes for 5k-node random model."""
        model = RandomGraphModel(target_nodes=5000, seed=42)
        assert validate_forward_pass(model, torch.randn(2, 64))

    @pytest.mark.slow
    def test_validation_10k(self):
        """Validation passes for 10k-node random model."""
        model = RandomGraphModel(target_nodes=10000, seed=42)
        assert validate_forward_pass(model, torch.randn(2, 64))

    @pytest.mark.slow
    def test_validation_20k(self):
        """Validation passes for 20k-node random model."""
        model = RandomGraphModel(target_nodes=20000, seed=42)
        assert validate_forward_pass(model, torch.randn(2, 64))

    @pytest.mark.slow
    def test_validation_50k(self):
        """Validation passes for 50k-node random model."""
        model = RandomGraphModel(target_nodes=50000, seed=42)
        assert validate_forward_pass(model, torch.randn(2, 64))

    @pytest.mark.slow
    def test_validation_100k(self):
        """Validation passes for 100k-node random model."""
        model = RandomGraphModel(target_nodes=100000, seed=42)
        assert validate_forward_pass(model, torch.randn(2, 64))

    @pytest.mark.rare
    def test_validation_250k(self):
        """Validation passes for 250k-node random model."""
        model = RandomGraphModel(target_nodes=250000, seed=42)
        assert validate_forward_pass(model, torch.randn(2, 64))


# -----------------------------------------------
# Engine selection tests
# -----------------------------------------------


class TestEngineSelection:
    """Test get_node_placement_engine logic."""

    def test_auto_selects_dot_for_small(self):
        assert get_node_placement_engine("auto", 500) == "dot"

    def test_auto_selects_non_dot_for_large(self):
        engine = get_node_placement_engine("auto", 5000)
        assert engine in ("elk", "sfdp")

    def test_dot_always_returns_dot(self):
        assert get_node_placement_engine("dot", 10000) == "dot"

    def test_sfdp_always_returns_sfdp(self):
        assert get_node_placement_engine("sfdp", 100) == "sfdp"

    def test_elk_falls_back_when_unavailable(self):
        if not elk_available():
            assert get_node_placement_engine("elk", 100) == "sfdp"

    def test_threshold_boundary(self):
        assert get_node_placement_engine("auto", _ELK_NODE_THRESHOLD - 1) == "dot"
        engine = get_node_placement_engine("auto", _ELK_NODE_THRESHOLD)
        assert engine in ("elk", "sfdp")


# -----------------------------------------------
# ELK utilities tests (don't require elkjs)
# -----------------------------------------------


class TestElkUtilities:
    """Test ELK helper functions that don't need Node.js."""

    def test_build_elk_graph_from_dot_quoted(self):
        dot_source = """
        digraph {
            "node_a" [label="A"]
            "node_b" [label="B"]
            "node_a" -> "node_b"
        }
        """
        elk = build_elk_graph(dot_source)
        assert len(elk["children"]) == 2
        assert len(elk["edges"]) == 1

    def test_build_elk_graph_from_dot_unquoted(self):
        dot_source = """
        digraph {
            node [ordering=out]
            node_a [label="A"]
            node_b [label="B"]
            node_c [label="C"]
            node_a -> node_b
            node_b -> node_c
        }
        """
        elk = build_elk_graph(dot_source)
        assert len(elk["children"]) == 3
        assert len(elk["edges"]) == 2
        node_ids = {c["id"] for c in elk["children"]}
        assert node_ids == {"node_a", "node_b", "node_c"}

    def test_inject_positions_quoted(self):
        dot_source = '"my_node" [label="test"]'
        positioned = {"children": [{"id": "my_node", "x": 0, "y": 0, "width": 150, "height": 40}]}
        result = inject_elk_positions(dot_source, positioned)
        assert 'pos="' in result

    def test_inject_positions_unquoted(self):
        dot_source = 'my_node [label="test"]'
        positioned = {"children": [{"id": "my_node", "x": 0, "y": 0, "width": 150, "height": 40}]}
        result = inject_elk_positions(dot_source, positioned)
        assert 'pos="' in result

    def test_inject_positions_empty(self):
        dot_source = 'my_node [label="test"]'
        result = inject_elk_positions(dot_source, {"children": []})
        assert result == dot_source

    def test_build_hierarchical_has_groups(self):
        """Hierarchical builder creates compound nodes for module nesting."""
        model = RandomGraphModel(target_nodes=500, nesting_depth=3, seed=42)
        ml = log_forward_pass(model, torch.randn(2, 64))
        entries = ml.layer_dict_main_keys
        elk = build_elk_graph_hierarchical(entries)
        # Should have at least one group_ compound node at top level.
        group_ids = [c["id"] for c in elk["children"] if c["id"].startswith("group_")]
        assert len(group_ids) > 0, "Expected module groups in hierarchical ELK graph"
        # Groups should have children.
        for child in elk["children"]:
            if child["id"].startswith("group_"):
                assert "children" in child and len(child["children"]) > 0
        ml.cleanup()

    def test_inject_positions_hierarchical(self):
        """Position injection works with nested compound nodes."""
        positioned = {
            "children": [
                {
                    "id": "group_mod1",
                    "x": 0,
                    "y": 0,
                    "children": [
                        {"id": "node_a", "x": 10, "y": 20, "width": 150, "height": 40},
                        {"id": "node_b", "x": 10, "y": 80, "width": 150, "height": 40},
                    ],
                },
                {"id": "node_c", "x": 200, "y": 0, "width": 150, "height": 40},
            ]
        }
        dot_source = 'node_a [label="A"]\nnode_b [label="B"]\nnode_c [label="C"]'
        result = inject_elk_positions(dot_source, positioned)
        # All three leaf nodes should have positions.
        assert result.count('pos="') == 3


# -----------------------------------------------
# Rendering tests
# -----------------------------------------------


class TestLargeGraphRendering:
    """Test visualization at scale."""

    def test_dot_renders_small_graph(self):
        """dot engine works for small graphs."""
        model = RandomGraphModel(target_nodes=200, seed=42)
        show_model_graph(
            model,
            torch.randn(2, 64),
            vis_node_placement="dot",
            save_only=True,
            vis_outpath=os.path.join(VIS_OUTPUT_DIR, "dot_200"),
        )

    @pytest.mark.slow
    def test_sfdp_renders_large_graph(self):
        """sfdp engine works for large graphs."""
        model = RandomGraphModel(target_nodes=3000, seed=42)
        show_model_graph(
            model,
            torch.randn(2, 64),
            vis_node_placement="sfdp",
            save_only=True,
            vis_outpath=os.path.join(VIS_OUTPUT_DIR, "sfdp_3k"),
        )

    @pytest.mark.skipif(not elk_available(), reason="elkjs not installed")
    @pytest.mark.slow
    def test_elk_renders_3k(self):
        """ELK engine works for 3k-node graphs."""
        model = RandomGraphModel(target_nodes=3000, seed=42)
        show_model_graph(
            model,
            torch.randn(2, 64),
            vis_node_placement="elk",
            save_only=True,
            vis_outpath=os.path.join(VIS_OUTPUT_DIR, "elk_3k"),
        )

    @pytest.mark.skipif(not elk_available(), reason="elkjs not installed")
    @pytest.mark.slow
    def test_elk_renders_5k(self):
        """ELK engine works for 5k-node graphs."""
        model = RandomGraphModel(target_nodes=5000, seed=42)
        show_model_graph(
            model,
            torch.randn(2, 64),
            vis_node_placement="elk",
            save_only=True,
            vis_outpath=os.path.join(VIS_OUTPUT_DIR, "elk_5k"),
        )

    @pytest.mark.skipif(not elk_available(), reason="elkjs not installed")
    @pytest.mark.slow
    def test_elk_renders_10k(self):
        """ELK engine works for 10k-node graphs."""
        model = RandomGraphModel(target_nodes=10000, seed=42)
        show_model_graph(
            model,
            torch.randn(2, 64),
            vis_node_placement="elk",
            save_only=True,
            vis_outpath=os.path.join(VIS_OUTPUT_DIR, "elk_10k"),
        )

    @pytest.mark.skipif(not elk_available(), reason="elkjs not installed")
    @pytest.mark.slow
    def test_elk_renders_20k(self):
        """ELK engine works for 20k-node graphs."""
        model = RandomGraphModel(target_nodes=20000, seed=42)
        show_model_graph(
            model,
            torch.randn(2, 64),
            vis_node_placement="elk",
            save_only=True,
            vis_outpath=os.path.join(VIS_OUTPUT_DIR, "elk_20k"),
        )

    @pytest.mark.skipif(not elk_available(), reason="elkjs not installed")
    @pytest.mark.slow
    def test_elk_renders_50k(self):
        """ELK engine works for 50k-node graphs."""
        model = RandomGraphModel(target_nodes=50000, seed=42)
        show_model_graph(
            model,
            torch.randn(2, 64),
            vis_node_placement="elk",
            save_only=True,
            vis_outpath=os.path.join(VIS_OUTPUT_DIR, "elk_50k"),
        )

    @pytest.mark.skipif(not elk_available(), reason="elkjs not installed")
    @pytest.mark.slow
    def test_elk_renders_100k(self):
        """ELK engine works for 100k-node graphs."""
        model = RandomGraphModel(target_nodes=100000, seed=42)
        show_model_graph(
            model,
            torch.randn(2, 64),
            vis_node_placement="elk",
            save_only=True,
            vis_outpath=os.path.join(VIS_OUTPUT_DIR, "elk_100k"),
        )

    @pytest.mark.skipif(not elk_available(), reason="elkjs not installed")
    @pytest.mark.rare
    def test_elk_renders_250k(self):
        """ELK engine works for 250k-node graphs."""
        model = RandomGraphModel(target_nodes=250000, seed=42)
        show_model_graph(
            model,
            torch.randn(2, 64),
            vis_node_placement="elk",
            vis_fileformat="svg",
            save_only=True,
            vis_outpath=os.path.join(VIS_OUTPUT_DIR, "elk_250k"),
        )

    @pytest.mark.skipif(not elk_available(), reason="elkjs not installed")
    @pytest.mark.rare
    def test_elk_renders_1M(self):
        """ELK engine works for 1M-node graphs. The trophy file."""
        model = RandomGraphModel(target_nodes=1000000, seed=42)
        show_model_graph(
            model,
            torch.randn(2, 64),
            vis_node_placement="elk",
            vis_fileformat="svg",
            save_only=True,
            vis_outpath=os.path.join(VIS_OUTPUT_DIR, "elk_1M"),
        )

    def test_vis_node_placement_forwarded(self):
        """vis_node_placement parameter reaches render_graph."""
        model = RandomGraphModel(target_nodes=200, seed=42)
        ml = log_forward_pass(model, torch.randn(2, 64))
        ml.render_graph(
            vis_opt="unrolled",
            save_only=True,
            vis_outpath=os.path.join(VIS_OUTPUT_DIR, "placement_test"),
            vis_node_placement="dot",
        )
        ml.cleanup()


# -----------------------------------------------
# dot vs ELK aesthetic comparison (manual inspection)
# -----------------------------------------------


class TestDotVsElkComparison:
    """Generate side-by-side dot and ELK renders for visual comparison.

    Run with: pytest tests/test_large_graphs.py -k "TestDotVsElk" -v
    Then inspect tests/test_outputs/visualizations/large/compare_*/
    """

    COMPARE_DIR = os.path.join(VIS_OUTPUT_DIR, "dot_vs_elk")

    @pytest.fixture(autouse=True)
    def _ensure_compare_dir(self):
        os.makedirs(self.COMPARE_DIR, exist_ok=True)

    def _render_both(self, model, x, name):
        """Render with both dot and ELK, save side by side."""
        # dot
        show_model_graph(
            model,
            x,
            save_only=True,
            vis_opt="unrolled",
            vis_node_placement="dot",
            vis_outpath=os.path.join(self.COMPARE_DIR, f"{name}_dot"),
        )
        # ELK
        show_model_graph(
            model,
            x,
            save_only=True,
            vis_opt="unrolled",
            vis_node_placement="elk",
            vis_outpath=os.path.join(self.COMPARE_DIR, f"{name}_elk"),
        )

    @pytest.mark.skipif(not elk_available(), reason="elkjs not installed")
    def test_compare_15_nodes(self):
        """Tiny model — easiest to spot aesthetic differences."""
        model = RandomGraphModel(target_nodes=15, seed=42, nesting_depth=1)
        self._render_both(model, torch.randn(2, 64), "15n")

    @pytest.mark.skipif(not elk_available(), reason="elkjs not installed")
    def test_compare_100_nodes(self):
        """Small model with moderate nesting."""
        model = RandomGraphModel(target_nodes=100, seed=42, nesting_depth=2)
        self._render_both(model, torch.randn(2, 64), "100n")

    @pytest.mark.skipif(not elk_available(), reason="elkjs not installed")
    def test_compare_500_nodes(self):
        """Medium model — complexity where differences become visible."""
        model = RandomGraphModel(target_nodes=500, seed=42, nesting_depth=2)
        self._render_both(model, torch.randn(2, 64), "500n")

    @pytest.mark.skipif(not elk_available(), reason="elkjs not installed")
    def test_compare_1k_nodes(self):
        """Larger model — still renderable by both engines."""
        model = RandomGraphModel(target_nodes=1000, seed=42, nesting_depth=3)
        self._render_both(model, torch.randn(2, 64), "1k")

    @pytest.mark.skipif(not elk_available(), reason="elkjs not installed")
    @pytest.mark.slow
    def test_compare_3k_nodes(self):
        """Near the ELK threshold — last size where dot is comfortable."""
        model = RandomGraphModel(target_nodes=3000, seed=42, nesting_depth=3)
        self._render_both(model, torch.randn(2, 64), "3k")


# -----------------------------------------------
# Benchmark (manual inspection)
# -----------------------------------------------


class TestDotThresholdBenchmark:
    """Empirical benchmark to calibrate _ELK_NODE_THRESHOLD."""

    @pytest.mark.slow
    def test_benchmark_dot_scaling(self):
        """Time graphviz dot at various node counts."""
        import time

        for target in [500, 1000, 2000, 3000, 3500, 4000]:
            model = RandomGraphModel(target_nodes=target, seed=42)
            x = torch.randn(2, 64)
            ml = log_forward_pass(model, x)
            actual_nodes = len(ml.layer_labels)
            start = time.time()
            try:
                ml.render_graph(
                    vis_opt="unrolled",
                    save_only=True,
                    vis_outpath=os.path.join(VIS_OUTPUT_DIR, f"bench_{target}"),
                    vis_node_placement="dot",
                )
                elapsed = time.time() - start
                print(f"  {target} target ({actual_nodes} actual): {elapsed:.1f}s")
            except Exception as e:
                elapsed = time.time() - start
                print(
                    f"  {target} target ({actual_nodes} actual): FAILED after {elapsed:.1f}s — {e}"
                )
            ml.cleanup()
