"""Tests for minimal static export helpers."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

import torchlens as tl


def test_export_html_minimal_self_contained_without_viewer(tmp_path: Path) -> None:
    """tl.export.html should write pan/zoom/hover HTML without viewer extras."""

    log = tl.log_forward_pass(nn.Sequential(nn.Linear(3, 3), nn.ReLU()), torch.randn(1, 3))
    output = tmp_path / "graph.html"

    written = tl.export.html(log, output)
    text = output.read_text(encoding="utf-8")

    assert written == output
    assert "addEventListener('wheel'" in text
    assert "onmousemove" in text
    assert "<script src=" not in text
    assert "<link " not in text


def test_export_svg_editable_has_stable_ids_and_classes(tmp_path: Path) -> None:
    """tl.export.svg(editable=True) should include semantic SVG metadata."""

    log = tl.log_forward_pass(nn.Linear(3, 2), torch.randn(1, 3))
    output = tmp_path / "graph.svg"

    tl.export.svg(log, output, editable=True)
    text = output.read_text(encoding="utf-8")

    assert "tl-node-" in text
    assert 'class="tl-node' in text
    assert 'class="tl-edge"' in text
