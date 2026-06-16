"""Regression tests for Graphviz render failure handling."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.data_classes.trace import Trace
from torchlens.visualization.rendering import GraphvizRenderError


class _TinyRenderModel(nn.Module):
    """Small model that produces forward and backward graph nodes."""

    def __init__(self) -> None:
        """Initialize submodules."""
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass."""
        return torch.relu(self.linear(x)).sum()


class _LargeChainRenderModel(nn.Module):
    """Model with enough repeated ops to exercise large PDF page geometry."""

    def __init__(self, width: int = 4, depth: int = 48) -> None:
        """Initialize a deterministic linear chain."""

        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(width, width) for _ in range(depth))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the chain with one activation per layer."""

        for layer in self.layers:
            x = torch.relu(layer(x))
        return x


@pytest.fixture
def forward_trace() -> Trace:
    """Return a tiny forward Trace."""

    return tl.trace(_TinyRenderModel(), torch.randn(2, 3, requires_grad=True))


@pytest.fixture
def backward_trace() -> Trace:
    """Return a tiny Trace with backward metadata."""

    trace = tl.trace(_TinyRenderModel(), torch.randn(2, 3, requires_grad=True), save_grads="all")
    trace.log_backward(trace[trace.output_layers[0]].out)
    return trace


def _raise_timeout(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
    """Simulate a Graphviz timeout from ``subprocess.run``."""

    raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs.get("timeout"))


def _raise_called_process_error(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
    """Simulate a Graphviz process failure from ``subprocess.run``."""

    raise subprocess.CalledProcessError(returncode=1, cmd=args[0], stderr=b"graphviz failed")


def _write_zero_byte_output(args: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    """Simulate a successful Graphviz run that leaves an empty output file."""

    del kwargs
    output_flag_index = args.index("-o")
    Path(args[output_flag_index + 1]).write_bytes(b"")
    return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")


def test_forward_render_timeout_raises_typed_error(
    forward_trace: Trace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward rendering raises a typed error when Graphviz times out."""

    monkeypatch.setattr(subprocess, "run", _raise_timeout)

    with pytest.raises(GraphvizRenderError, match="timed out.*lowering dpi.*direct SVG.*node cap"):
        forward_trace.draw(
            vis_outpath=str(tmp_path / "forward"),
            vis_save_only=True,
            vis_fileformat="svg",
            order_siblings=False,
        )


def test_backward_render_timeout_raises_typed_error(
    backward_trace: Trace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backward rendering raises a typed error when Graphviz times out."""

    monkeypatch.setattr(subprocess, "run", _raise_timeout)

    with pytest.raises(GraphvizRenderError, match="timed out.*lowering dpi.*direct SVG.*node cap"):
        backward_trace.draw_backward(
            vis_outpath=str(tmp_path / "backward"),
            vis_save_only=True,
            vis_fileformat="svg",
        )


def test_combined_render_timeout_raises_typed_error(
    backward_trace: Trace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Combined rendering raises a typed error when Graphviz times out."""

    monkeypatch.setattr(subprocess, "run", _raise_timeout)

    with pytest.raises(GraphvizRenderError, match="timed out.*lowering dpi.*direct SVG.*node cap"):
        backward_trace.draw_combined(
            vis_outpath=str(tmp_path / "combined"),
            vis_save_only=True,
            vis_fileformat="svg",
        )


def test_forward_zero_byte_render_raises_typed_error(
    forward_trace: Trace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward rendering raises when Graphviz reports success with an empty file."""

    monkeypatch.setattr(subprocess, "run", _write_zero_byte_output)

    with pytest.raises(GraphvizRenderError, match="zero-byte.*lowering dpi.*direct SVG.*node cap"):
        forward_trace.draw(
            vis_outpath=str(tmp_path / "forward_empty"),
            vis_save_only=True,
            vis_fileformat="svg",
            order_siblings=False,
        )


def test_backward_zero_byte_render_raises_typed_error(
    backward_trace: Trace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backward rendering raises when Graphviz reports success with an empty file."""

    monkeypatch.setattr(subprocess, "run", _write_zero_byte_output)

    with pytest.raises(GraphvizRenderError, match="zero-byte.*lowering dpi.*direct SVG.*node cap"):
        backward_trace.draw_backward(
            vis_outpath=str(tmp_path / "backward_empty"),
            vis_save_only=True,
            vis_fileformat="svg",
        )


def test_combined_zero_byte_render_raises_typed_error(
    backward_trace: Trace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Combined rendering raises when Graphviz reports success with an empty file."""

    monkeypatch.setattr(subprocess, "run", _write_zero_byte_output)

    with pytest.raises(GraphvizRenderError, match="zero-byte.*lowering dpi.*direct SVG.*node cap"):
        backward_trace.draw_combined(
            vis_outpath=str(tmp_path / "combined_empty"),
            vis_save_only=True,
            vis_fileformat="svg",
        )


def test_forward_called_process_error_raises_typed_error(
    forward_trace: Trace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward rendering raises a typed error when Graphviz exits unsuccessfully."""

    monkeypatch.setattr(subprocess, "run", _raise_called_process_error)

    with pytest.raises(GraphvizRenderError, match="Graphviz failed.*graphviz failed"):
        forward_trace.draw(
            vis_outpath=str(tmp_path / "forward_failed"),
            vis_save_only=True,
            vis_fileformat="svg",
            order_siblings=False,
        )


def test_backward_called_process_error_raises_typed_error(
    backward_trace: Trace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backward rendering raises a typed error when Graphviz exits unsuccessfully."""

    monkeypatch.setattr(subprocess, "run", _raise_called_process_error)

    with pytest.raises(GraphvizRenderError, match="Graphviz failed.*graphviz failed"):
        backward_trace.draw_backward(
            vis_outpath=str(tmp_path / "backward_failed"),
            vis_save_only=True,
            vis_fileformat="svg",
        )


def test_combined_called_process_error_raises_typed_error(
    backward_trace: Trace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Combined rendering raises a typed error when Graphviz exits unsuccessfully."""

    monkeypatch.setattr(subprocess, "run", _raise_called_process_error)

    with pytest.raises(GraphvizRenderError, match="Graphviz failed.*graphviz failed"):
        backward_trace.draw_combined(
            vis_outpath=str(tmp_path / "combined_failed"),
            vis_save_only=True,
            vis_fileformat="svg",
        )


def test_large_composed_pdf_contains_visible_graph_region(tmp_path: Path) -> None:
    """Large composed PDF renders graph contents inside the page bounds."""

    fitz = pytest.importorskip("fitz")
    model = _LargeChainRenderModel().eval()
    trace = tl.trace(model, torch.randn(1, 4))
    pdf_path = tmp_path / "large_composed.pdf"
    try:
        trace.draw(
            vis_outpath=str(pdf_path.with_suffix("")),
            vis_save_only=True,
            vis_fileformat="pdf",
            code_panel=lambda _model: "def forward(self, x):\n    return x",
            order_siblings=False,
        )
    finally:
        trace.cleanup()

    document = fitz.open(pdf_path)
    try:
        page = document[0]
        page_rect = page.rect
        content_rect = fitz.Rect()
        for block in page.get_text("blocks"):
            content_rect |= fitz.Rect(block[:4])
        for drawing in page.get_drawings():
            rect = drawing.get("rect")
            if rect is not None:
                content_rect |= rect
        assert not content_rect.is_empty
        assert page_rect.intersects(content_rect)
        assert content_rect.get_area() > 0.01 * page_rect.get_area()
    finally:
        document.close()
