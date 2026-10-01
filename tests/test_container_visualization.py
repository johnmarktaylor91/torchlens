"""Container visualization and UI tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.experimental import dagua


class DictOutputModel(nn.Module):
    """Return a two-leaf dictionary output."""

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the model."""

        return {"a": x + 1, "b": x + 2}


class TupleOutputModel(nn.Module):
    """Return a homogeneous tuple output."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Run the model."""

        return tuple(x + index for index in range(4))


class WideTupleOutputModel(nn.Module):
    """Return a 16-leaf homogeneous tuple (over the inline collapse cap)."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Run the model."""

        return tuple(x + float(index) for index in range(16))


class MixedShapeTupleModel(nn.Module):
    """Return a tuple whose leaf shapes differ."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the model."""

        return x + 1, torch.stack((x, x))


class DictInputModel(nn.Module):
    """Consume a two-leaf dictionary input."""

    def forward(self, payload: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run the model."""

        return payload["a"] + payload["b"]


class PairModule(nn.Module):
    """Return a two-leaf dictionary for a parent module to consume."""

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the model."""

        return {"left": x + 1, "right": x + 2}


class MidGraphContainerModel(nn.Module):
    """Consume leaves pulled from a submodule output container."""

    def __init__(self) -> None:
        """Initialize the model."""

        super().__init__()
        self.pair = PairModule()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model."""

        pair = self.pair(x)
        return pair["left"] * pair["right"]


def _trace(model: nn.Module) -> Any:
    """Trace a model with final-output structure enabled."""

    return tl.trace(
        model, torch.ones(2), capture=tl.options.CaptureOptions(intervention_ready=True)
    )


def test_show_containers_false_is_default_dot_identity(tmp_path: Path) -> None:
    """Default rendering stays byte-identical to explicit ``show_containers=False``."""

    trace = _trace(DictOutputModel())
    default_source = trace.draw(
        vis_save_only=True,
        vis_fileformat="dot",
        vis_outpath=str(tmp_path / "default"),
    )
    explicit_source = trace.draw(
        show_containers=False,
        vis_save_only=True,
        vis_fileformat="dot",
        vis_outpath=str(tmp_path / "explicit"),
    )

    assert default_source == explicit_source
    assert "container_group" not in default_source
    assert "container_kind" not in default_source
    assert "container_role" not in default_source


def test_show_containers_labels_adds_key_labeled_midpoint_edges(tmp_path: Path) -> None:
    """Label mode adds key labels on real edges into container leaves."""

    trace = _trace(DictOutputModel())
    source = trace.draw(
        show_containers="labels",
        vis_save_only=True,
        vis_fileformat="dot",
        vis_outpath=str(tmp_path / "labels"),
    )

    assert 'label=<<TABLE BORDER="0"' in source
    assert '<FONT POINT-SIZE="8">a</FONT>' in source
    assert '<FONT POINT-SIZE="8">b</FONT>' in source
    assert "labeldistance=" not in source
    assert "labelangle=" not in source


def test_show_containers_cluster_falls_back_without_single_owner(tmp_path: Path) -> None:
    """Cluster mode falls back to labels when leaves have no one module owner."""

    trace = _trace(DictOutputModel())
    source = trace.draw(
        show_containers="cluster",
        vis_save_only=True,
        vis_fileformat="dot",
        vis_outpath=str(tmp_path / "cluster"),
    )

    assert '<FONT POINT-SIZE="8">a</FONT>' in source
    assert "cluster_container_" not in source


def test_show_containers_collapsed_only_for_identical_shapes(tmp_path: Path) -> None:
    """Collapsed mode hides oversized homogeneous containers only."""

    homogeneous = _trace(TupleOutputModel())
    collapsed_source = homogeneous.draw(
        show_containers="collapsed",
        container_max_inline=2,
        vis_save_only=True,
        vis_fileformat="dot",
        vis_outpath=str(tmp_path / "collapsed"),
    )
    assert "container_final_output_0_tuple" in collapsed_source
    assert "output_1pass1" not in collapsed_source

    mixed = _trace(MixedShapeTupleModel())
    inline_source = mixed.draw(
        show_containers="collapsed",
        container_max_inline=1,
        vis_save_only=True,
        vis_fileformat="dot",
        vis_outpath=str(tmp_path / "inline"),
    )
    assert "container_final_output_0_tuple" not in inline_source
    assert "output_1pass1" in inline_source


def test_show_containers_nodes_adds_source_and_sink_nodes(tmp_path: Path) -> None:
    """Node mode adds boundary container nodes through overlay edges."""

    trace = tl.trace(
        DictInputModel(),
        {"a": torch.ones(2), "b": torch.ones(2)},
        capture=tl.options.CaptureOptions(capture_container_structure=True),
    )
    source = trace.draw(
        show_containers="nodes",
        vis_save_only=True,
        vis_fileformat="dot",
        vis_outpath=str(tmp_path / "nodes_input"),
    )

    assert "dict[2] (model input)" in source
    assert "container_node_" in source
    assert '<FONT POINT-SIZE="8">a</FONT>' in source
    assert '<FONT POINT-SIZE="8">b</FONT>' in source
    assert "constraint=false" in source


def test_show_containers_nodes_adds_midgraph_member_ties(tmp_path: Path) -> None:
    """Mid-graph containers use dashed non-constraining member-of ties."""

    trace = tl.trace(
        MidGraphContainerModel(),
        torch.ones(2),
        capture=tl.options.CaptureOptions(capture_container_structure=True),
    )
    source = trace.draw(
        show_containers="nodes",
        vis_save_only=True,
        vis_fileformat="dot",
        vis_outpath=str(tmp_path / "nodes_midgraph"),
    )

    assert "dict[2] (call output)" in source
    assert "arrowhead=none" in source
    assert "constraint=false" in source
    assert "container_node_" in source


def test_show_containers_nodes_homogeneous_collapse_is_coherent(tmp_path: Path) -> None:
    """Node mode suppresses collapsed leaves instead of orphaning them.

    Homogeneous-container collapse applies in ``"nodes"`` mode too
    (``docs/containers.md``), and the edge pass already reroutes every leaf
    edge to the summary box. The leaf-suppression gates must agree: drawing
    the leaves anyway produces N orphaned nodes with zero inbound edges
    beside a summary box that stole their edges.
    """

    import re

    trace = tl.trace(
        WideTupleOutputModel(),
        torch.ones(2, 4),
        capture=tl.options.CaptureOptions(capture_container_structure=True),
    )
    source = trace.draw(
        show_containers="nodes",
        vis_save_only=True,
        vis_fileformat="dot",
        vis_outpath=str(tmp_path / "nodes_wide"),
    )

    declared = set(re.findall(r'^\s*"?([\w.:\-+@]+)"? \[', source, re.MULTILINE))
    endpoints: set[str] = set()
    for tail, head in re.findall(r'"?([\w.:+\-]+)"? -> "?([\w.:+\-]+)"?', source):
        endpoints.add(tail)
        endpoints.add(head)

    # The collapse summary box is present and owns the leaf edges.
    assert "container_final_output_0_tuple" in declared
    assert "container_final_output_0_tuple" in endpoints
    # The 16 collapsed output leaves are suppressed, exactly as in
    # "collapsed"/"auto" mode -- never drawn as edgeless orphans.
    leaf_declarations = {name for name in declared if name.startswith("output_")}
    assert leaf_declarations == set()
    # No declared node is an orphan. The three DOT attr_stmt keywords
    # (graph/node/edge) are attribute defaults, not nodes: the typography
    # record pins ``fontname`` on the edge scope of every preset, so the
    # source now carries a global ``edge [...]`` line the node regex matches.
    orphan_declarations = declared - endpoints - {"graph", "node", "edge"}
    assert orphan_declarations == set()


def test_show_containers_nodes_draw_then_save_round_trips(tmp_path: Path) -> None:
    """Node-mode rendering leaves no runtime-only state in saved traces."""

    trace = tl.trace(
        DictInputModel(),
        {"a": torch.ones(2), "b": torch.ones(2)},
        capture=tl.options.CaptureOptions(capture_container_structure=True),
    )
    trace.draw(
        show_containers="nodes",
        vis_save_only=True,
        vis_fileformat="dot",
        vis_outpath=str(tmp_path / "nodes_before_save"),
    )
    save_path = tmp_path / "draw_then_save.tlspec"
    trace.save(save_path)
    loaded = tl.load(save_path)

    assert loaded.input_structure.kind == "dict"
    assert not hasattr(loaded, "_pending_container_collapse_nodes")


def test_dagua_graph_carries_container_semantic_attrs() -> None:
    """Dagua bridge carries portable semantic container metadata."""
    pytest.importorskip("dagua")

    trace = _trace(DictOutputModel())
    graph = dagua.trace_to_dagua_graph(trace)

    assert "final_output:0:dict" in graph.container_group
    assert "dict" in graph.container_kind
    assert "a" in graph.container_role
    assert "b" in graph.edge_container_role


class RecurrentDictOutputModel(nn.Module):
    """Run a layer three times, returning a dict output (multi-pass layers)."""

    def __init__(self) -> None:
        """Initialize the recurrent layer."""

        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Apply the layer three times and return a container output."""

        for _ in range(3):
            x = torch.relu(self.lin(x))
        return {"out": x, "double": x * 2}


@pytest.mark.parametrize("vis_mode", ["rolled", "unrolled"])
@pytest.mark.parametrize("show_containers", ["labels", "cluster", "collapsed", "auto", "nodes"])
def test_show_containers_survives_multipass_layers(
    tmp_path: Path, vis_mode: str, show_containers: str
) -> None:
    """Container modes never leak the multi-pass tripwire on recurrent traces.

    Regression: ``_container_edge_label`` (and its group/role siblings) read
    ``container_path`` with a plain ``getattr`` default, which swallows only
    ``AttributeError``; on a rolled multi-pass Layer the deliberate
    ``layer_pass_ambiguous`` ValueError escaped and crashed ``draw()``.
    Container metadata is per-pass, so the rolled aggregate explicitly
    degrades to no container decoration instead.
    """

    trace = tl.trace(RecurrentDictOutputModel(), torch.randn(2, 4))
    trace.draw(
        vis_mode=vis_mode,
        show_containers=show_containers,
        vis_save_only=True,
        vis_fileformat="dot",
        vis_outpath=str(tmp_path / f"{vis_mode}_{show_containers}"),
    )


def test_dagua_container_semantics_survive_multipass_layers() -> None:
    """The dagua bridge degrades per-pass container reads on rolled Layers.

    Exercises the bridge helper directly so the regression is covered even
    without the optional dagua runtime installed.
    """

    from torchlens.experimental.dagua._bridge import _container_semantic_attrs

    trace = tl.trace(RecurrentDictOutputModel(), torch.randn(2, 4))
    multipass_layer = trace.layers["relu_1_2"]
    assert multipass_layer.num_passes > 1

    attrs = _container_semantic_attrs(multipass_layer)

    assert attrs == {
        "container_group": None,
        "container_kind": None,
        "container_role": None,
    }
