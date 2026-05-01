from pathlib import Path
import subprocess
import sys
import textwrap

import torch

from torchlens import log_forward_pass
from torchlens.experimental import dagua

import example_models


def test_dagua_renderer_requires_experimental_opt_in() -> None:
    """Core renderer rejects Dagua before the experimental module is imported."""

    script = textwrap.dedent(
        """
import torch
from torch import nn
import torchlens as tl

class M(nn.Module):
    def forward(self, x):
        return x + 1

log = tl.log_forward_pass(M(), torch.ones(1), layers_to_save=None)
try:
    try:
        log.render_graph(vis_renderer="dagua", vis_save_only=True, vis_fileformat="svg")
    except RuntimeError as exc:
        message = "opt in via `from torchlens.experimental import dagua` first"
        raise SystemExit(0 if message in str(exc) else 2)
    raise SystemExit(1)
finally:
    log.cleanup()
"""
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_model_log_to_dagua_graph_builds_semantic_nodes_and_clusters() -> None:
    model = example_models.AestheticFrozenMix().eval()
    log = log_forward_pass(model, torch.rand(1, 8), layers_to_save=None)
    try:
        graph = dagua.model_log_to_dagua_graph(log, vis_mode="unrolled", direction="leftright")
        labels = [graph.node_labels[i] for i in range(graph.num_nodes)]
        assert any("trainable" in label.lower() for label in labels)
        assert any(node_type == "frozen_params" for node_type in graph.node_types)
        assert any(name.startswith("frozen_fc") for name in graph.clusters)
        assert any(name.startswith("trainable_fc") for name in graph.clusters)
    finally:
        log.cleanup()


def test_dagua_renderer_exports_svg(tmp_path: Path) -> None:
    model = example_models.ResidualBlockModel().eval()
    log = log_forward_pass(model, torch.rand(1, 16, 16, 16), layers_to_save=None)
    out = tmp_path / "residual.svg"
    try:
        log.render_graph(
            vis_renderer="dagua",
            vis_theme="torchlens",
            vis_mode="unrolled",
            direction="leftright",
            vis_save_only=True,
            vis_fileformat="svg",
            vis_outpath=str(out),
        )
        assert out.exists()
        assert "svg" in out.read_text().lower()
    finally:
        log.cleanup()


def test_render_audit_exposes_unused_fields() -> None:
    model = example_models.SimpleFF().eval()
    log = log_forward_pass(model, torch.rand(5, 5), layers_to_save=None)
    try:
        audit = dagua.build_render_audit(log).to_dict()
        assert "model_log_unused" in audit
        assert "entry_unused" in audit
        assert "module_unused" in audit
        assert "time_total" in audit["model_log_unused"]
    finally:
        log.cleanup()
