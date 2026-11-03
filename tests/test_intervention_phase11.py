"""Phase 11 intervention visualization wiring tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from example_models import TinyReluAdd as _ReluAdd

import torchlens as tl
from torchlens.experimental import dagua


def _interventions() -> tl.Trace:
    """Build an intervention-ready log with a relu replacement recipe.

    Returns
    -------
    tl.Trace
        Model log with one ``set`` intervention at the relu site.
    """

    x = torch.randn(2, 3)
    log = tl.trace(_ReluAdd(), x, capture=tl.options.CaptureOptions(replayable=True))
    log.set(tl.func("relu"), torch.zeros(2, 3))
    return log


@pytest.mark.smoke
def test_trace_show_accepts_intervention_options() -> None:
    """``Trace.show`` accepts Phase 11 intervention visualization kwargs."""

    log = _interventions()
    try:
        assert "digraph" in str(log.show(intervention_mode="node_mark"))
        assert "digraph" in str(log.show(intervention_mode="node_mark", show_cone=True))
        assert "digraph" in str(log.show(intervention_mode="node_mark", show_cone=False))
        assert "digraph" in str(log.show(intervention_mode="as_node"))
    finally:
        log.cleanup()


def test_node_mark_styles_site_and_cone(tmp_path: Path) -> None:
    """``node_mark`` marks intervention sites and cone members in DOT output."""

    log = _interventions()
    try:
        dot = log.draw(
            path=str(tmp_path / "node_mark"),
            file_format="dot",
            intervention_mode="node_mark",
            show_cone=True,
        ).dot
        assert "#7B1FA2" in dot
        assert 'class="tl_edited"' in dot or "class=tl_edited" in dot
        assert "tl_downstream" in dot
        assert "#FFB3FF" not in dot and "#FF00FF" not in dot

        without_cone = log.draw(
            path=str(tmp_path / "node_mark_no_cone"),
            file_format="dot",
            intervention_mode="node_mark",
            show_cone=False,
        ).dot
        assert "#7B1FA2" in without_cone
        assert "tl_downstream" not in without_cone
    finally:
        log.cleanup()


def test_as_node_inserts_hook_node_between_site_and_consumers(tmp_path: Path) -> None:
    """``as_node`` inserts a diamond hook node after an intervention site."""

    log = _interventions()
    try:
        dot = log.draw(
            path=str(tmp_path / "as_node"),
            file_format="dot",
            intervention_mode="as_node",
        ).dot
        assert "intervention_hook_" in dot
        assert "shape=diamond" in dot
        assert "intervention_hook_relu" in dot
    finally:
        log.cleanup()


def test_bundle_show_draws_every_member(tmp_path: Path) -> None:
    """``Bundle.show`` takes ``Trace.draw`` keywords and returns one Drawing per member."""

    log = _interventions()
    clean = tl.trace(_ReluAdd(), torch.randn(2, 3))
    try:
        bundle = tl.Bundle({"intervened": log, "clean": clean})
        pictures = bundle.show(path=str(tmp_path / "strip"), file_format="svg")
        assert set(pictures) == {"intervened", "clean"}
        for name, picture in pictures.items():
            assert isinstance(picture, tl.viz.Drawing)
            assert any(written.name.startswith(f"strip_{name}") for written in tmp_path.iterdir())
    finally:
        log.cleanup()
        clean.cleanup()


def test_dagua_bridge_exposes_intervention_metadata() -> None:
    """Dagua graph objects expose Phase 11 intervention metadata arrays."""
    pytest.importorskip("dagua")

    log = _interventions()
    try:
        graph = dagua.trace_to_dagua_graph(log, rolled=False)
        assert hasattr(graph, "is_intervention_site")
        assert hasattr(graph, "is_in_cone")
        assert hasattr(graph, "interventions_summary")
        assert any(graph.is_intervention_site)
        assert any(graph.is_in_cone)
        assert len(graph.interventions_summary) == graph.num_nodes
    finally:
        log.cleanup()
