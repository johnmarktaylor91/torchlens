"""Regression tests for Graphviz buffer visibility modes."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.data_classes.model_log import ModelLog


class _BatchNormOnly(nn.Module):
    """Small model with BatchNorm running-stat buffers."""

    def __init__(self) -> None:
        """Initialize the BatchNorm module."""

        super().__init__()
        self.bn = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run BatchNorm in eval mode.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Batch-normalized tensor.
        """

        return self.bn(x)


class _BatchNormWithArchitecturalBuffer(nn.Module):
    """Small model with noise buffers and a meaningful architectural buffer."""

    def __init__(self) -> None:
        """Initialize the BatchNorm module and causal mask buffer."""

        super().__init__()
        self.bn = nn.BatchNorm1d(4)
        self.register_buffer("causal_mask", torch.ones(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run BatchNorm and add an architectural buffer.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Batch-normalized tensor shifted by the causal mask.
        """

        return self.bn(x) + self.causal_mask


def _log_model(model: nn.Module) -> ModelLog:
    """Return a metadata-only ModelLog for a small deterministic input.

    Parameters
    ----------
    model:
        Model to execute in eval mode.

    Returns
    -------
    ModelLog
        Logged forward-pass metadata.
    """

    model.eval()
    return tl.log_forward_pass(model, torch.randn(2, 4), layers_to_save="none")


def _render_dot(log: ModelLog, tmp_path: Path, show_buffer_layers: str | bool) -> str:
    """Render a Graphviz DOT source string for a buffer visibility mode.

    Parameters
    ----------
    log:
        ModelLog to render.
    tmp_path:
        Temporary directory for Graphviz output files.
    show_buffer_layers:
        Buffer visibility mode or legacy bool.

    Returns
    -------
    str
        Rendered DOT source.
    """

    return log.render_graph(
        vis_save_only=True,
        vis_fileformat="svg",
        vis_outpath=str(tmp_path / f"buffer_visibility_{show_buffer_layers}"),
        show_buffer_layers=show_buffer_layers,
        vis_node_placement="dot",
    )


def _node_line(dot_source: str, node_name: str) -> str:
    """Return the DOT line for a node.

    Parameters
    ----------
    dot_source:
        DOT source to inspect.
    node_name:
        Graphviz node name.

    Returns
    -------
    str
        Node definition line.
    """

    prefix = f"\t{node_name} ["
    for line in dot_source.splitlines():
        if line.startswith(prefix):
            return line
    raise AssertionError(f"Node {node_name!r} was not found in DOT source.")


def _buffer_node_lines(dot_source: str) -> list[str]:
    """Return DOT lines that define rendered buffer nodes.

    Parameters
    ----------
    dot_source:
        DOT source to inspect.

    Returns
    -------
    list[str]
        Lines defining Graphviz buffer nodes.
    """

    return [
        line
        for line in dot_source.splitlines()
        if line.startswith("\tbuffer_") and " [label=" in line
    ]


def _child_ops_for_buffer(log: ModelLog, buffer_address: str) -> list[str]:
    """Return child op labels for a buffer address.

    Parameters
    ----------
    log:
        ModelLog to inspect.
    buffer_address:
        Buffer address whose child ops should be returned.

    Returns
    -------
    list[str]
        Child layer labels for the matching buffer.
    """

    for node in log.layer_dict_main_keys.values():
        if node.is_buffer_layer and node.buffer_address == buffer_address:
            return [child.replace(":", "pass") for child in node.child_layers]
    raise AssertionError(f"Buffer {buffer_address!r} was not found in the log.")


@pytest.mark.smoke
def test_never_hides_all_and_marks_parents(tmp_path: Path) -> None:
    """Mode ``never`` hides all buffers and marks ops with hidden buffer parents."""

    log = _log_model(_BatchNormOnly())
    try:
        dot_source = _render_dot(log, tmp_path, "never")
    finally:
        log.cleanup()

    assert _buffer_node_lines(dot_source) == []
    assert "peripheries=2" in _node_line(dot_source, "batchnorm_1_1")
    assert 'tooltip="Hidden buffers:' in _node_line(dot_source, "batchnorm_1_1")


@pytest.mark.smoke
def test_meaningful_hides_noise_only(tmp_path: Path) -> None:
    """Mode ``meaningful`` hides noisy buffers but keeps architectural buffers."""

    log = _log_model(_BatchNormWithArchitecturalBuffer())
    try:
        dot_source = _render_dot(log, tmp_path, "meaningful")
        causal_mask_children = _child_ops_for_buffer(log, "causal_mask")
    finally:
        log.cleanup()

    buffer_node_lines = _buffer_node_lines(dot_source)
    assert not any("running_mean" in line for line in buffer_node_lines)
    assert not any("running_var" in line for line in buffer_node_lines)
    assert not any("num_batches_tracked" in line for line in buffer_node_lines)
    assert "causal_mask" in dot_source
    assert "peripheries=2" in _node_line(dot_source, "batchnorm_1_1")
    for child_name in causal_mask_children:
        assert "peripheries=2" not in _node_line(dot_source, child_name)


@pytest.mark.smoke
def test_always_shows_all_no_marker(tmp_path: Path) -> None:
    """Mode ``always`` shows all buffers and omits hidden-buffer markers."""

    log = _log_model(_BatchNormWithArchitecturalBuffer())
    try:
        dot_source = _render_dot(log, tmp_path, "always")
    finally:
        log.cleanup()

    assert "running_mean" in dot_source
    assert "running_var" in dot_source
    assert "causal_mask" in dot_source
    assert "peripheries=2" not in dot_source
    assert "Hidden buffers:" not in dot_source


@pytest.mark.smoke
def test_visible_buffer_uses_cylinder_shape(tmp_path: Path) -> None:
    """Visible buffers render as white cylinders instead of gray boxes."""

    log = _log_model(_BatchNormWithArchitecturalBuffer())
    try:
        dot_source = _render_dot(log, tmp_path, "always")
    finally:
        log.cleanup()

    buffer_node_lines = _buffer_node_lines(dot_source)
    causal_mask_lines = [line for line in buffer_node_lines if "causal_mask" in line]

    assert len(causal_mask_lines) == 1
    assert "shape=cylinder" in causal_mask_lines[0]
    assert "#888888" not in causal_mask_lines[0]


@pytest.mark.smoke
def test_legacy_true_matches_always(tmp_path: Path) -> None:
    """Legacy ``True`` buffer visibility maps to ``always``."""

    log = _log_model(_BatchNormWithArchitecturalBuffer())
    try:
        legacy_dot = _render_dot(log, tmp_path, True)
        tri_state_dot = _render_dot(log, tmp_path, "always")
    finally:
        log.cleanup()

    assert legacy_dot == tri_state_dot


@pytest.mark.smoke
def test_legacy_false_matches_never(tmp_path: Path) -> None:
    """Legacy ``False`` buffer visibility maps to ``never``."""

    log = _log_model(_BatchNormWithArchitecturalBuffer())
    try:
        legacy_dot = _render_dot(log, tmp_path, False)
        tri_state_dot = _render_dot(log, tmp_path, "never")
    finally:
        log.cleanup()

    assert legacy_dot == tri_state_dot
