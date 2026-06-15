"""Container visualization and UI tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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


class MixedShapeTupleModel(nn.Module):
    """Return a tuple whose leaf shapes differ."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the model."""

        return x + 1, torch.stack((x, x))


def _trace(model: nn.Module) -> Any:
    """Trace a model with final-output structure enabled."""

    return tl.trace(model, torch.ones(2), intervention_ready=True)


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


def test_dagua_graph_carries_container_semantic_attrs() -> None:
    """Dagua bridge carries portable semantic container metadata."""

    trace = _trace(DictOutputModel())
    graph = dagua.trace_to_dagua_graph(trace)

    assert "final_output:0:dict" in graph.container_group
    assert "dict" in graph.container_kind
    assert "a" in graph.container_role
    assert "b" in graph.edge_container_role
