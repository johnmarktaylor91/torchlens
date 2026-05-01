"""Phase 11 intervention visualization wiring tests."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

import torchlens as tl
from torchlens.experimental import dagua


class _ReluAdd(nn.Module):
    """Small graph with an intervention site and downstream consumer."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a relu followed by an add."""

        return torch.relu(x) + 1


def _intervention_log() -> tl.ModelLog:
    """Build an intervention-ready log with a relu replacement recipe.

    Returns
    -------
    tl.ModelLog
        Model log with one ``set`` intervention at the relu site.
    """

    x = torch.randn(2, 3)
    log = tl.log_forward_pass(_ReluAdd(), x, vis_opt="none", intervention_ready=True)
    log.set(tl.func("relu"), torch.zeros(2, 3))
    return log


def test_model_log_show_accepts_intervention_options() -> None:
    """``ModelLog.show`` accepts Phase 11 intervention visualization kwargs."""

    log = _intervention_log()
    try:
        assert log.show(vis_opt="none", vis_intervention_mode="node_mark") is None
        assert (
            log.show(vis_opt="none", vis_intervention_mode="node_mark", vis_show_cone=True) is None
        )
        assert (
            log.show(vis_opt="none", vis_intervention_mode="node_mark", vis_show_cone=False) is None
        )
        assert log.show(vis_opt="none", vis_intervention_mode="as_node") is None
    finally:
        log.cleanup()


def test_node_mark_styles_site_and_cone(tmp_path: Path) -> None:
    """``node_mark`` marks intervention sites and cone members in DOT output."""

    log = _intervention_log()
    try:
        dot = log.render_graph(
            vis_save_only=True,
            vis_outpath=str(tmp_path / "node_mark"),
            vis_fileformat="dot",
            vis_intervention_mode="node_mark",
            vis_show_cone=True,
        )
        assert "#FF00FF" in dot
        assert "#FFB3FF" in dot

        without_cone = log.render_graph(
            vis_save_only=True,
            vis_outpath=str(tmp_path / "node_mark_no_cone"),
            vis_fileformat="dot",
            vis_intervention_mode="node_mark",
            vis_show_cone=False,
        )
        assert "#FF00FF" in without_cone
        assert "#FFB3FF" not in without_cone
    finally:
        log.cleanup()


def test_as_node_inserts_hook_node_between_site_and_consumers(tmp_path: Path) -> None:
    """``as_node`` inserts a diamond hook node after an intervention site."""

    log = _intervention_log()
    try:
        dot = log.render_graph(
            vis_save_only=True,
            vis_outpath=str(tmp_path / "as_node"),
            vis_fileformat="dot",
            vis_intervention_mode="as_node",
        )
        assert "intervention_hook_" in dot
        assert "shape=diamond" in dot
        assert "intervention_hook_relu" in dot
    finally:
        log.cleanup()


def test_bundle_show_accepts_vis_opt_none() -> None:
    """``Bundle.show`` accepts render kwargs and skips on ``vis_opt='none'``."""

    log = _intervention_log()
    clean = tl.log_forward_pass(_ReluAdd(), torch.randn(2, 3), vis_opt="none")
    try:
        bundle = tl.bundle({"intervened": log, "clean": clean})
        assert bundle.show(vis_opt="none") == {"intervened": None, "clean": None}
    finally:
        log.cleanup()
        clean.cleanup()


def test_dagua_bridge_exposes_intervention_metadata() -> None:
    """Dagua graph objects expose Phase 11 intervention metadata arrays."""

    log = _intervention_log()
    try:
        graph = dagua.model_log_to_dagua_graph(log, vis_mode="unrolled")
        assert hasattr(graph, "is_intervention_site")
        assert hasattr(graph, "is_in_cone")
        assert hasattr(graph, "intervention_log_summary")
        assert any(graph.is_intervention_site)
        assert any(graph.is_in_cone)
        assert len(graph.intervention_log_summary) == graph.num_nodes
    finally:
        log.cleanup()
