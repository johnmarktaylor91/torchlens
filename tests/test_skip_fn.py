"""Tests for predicate-based node skipping in visualization."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

import torchlens as tl
from torchlens.data_classes.layer_log import LayerLog
from torchlens.visualization.rendering import GRADIENT_ARROW_COLOR


def _render_dot(log: tl.ModelLog, tmp_path: Any, **kwargs: Any) -> str:
    """Render a ModelLog to DOT using a temporary SVG output path."""

    tmp_path.mkdir(parents=True, exist_ok=True)
    return log.render_graph(
        vis_save_only=True,
        vis_fileformat="svg",
        vis_outpath=str(tmp_path / "graph"),
        **kwargs,
    )


def _first_label(log: tl.ModelLog, layer_type: str) -> str:
    """Return the first layer label with a matching layer type."""

    for layer_log in log.layer_logs.values():
        if layer_log.layer_type == layer_type:
            return layer_log.layer_label
    raise AssertionError(f"No layer with type {layer_type}")


def _labels(log: tl.ModelLog, layer_type: str) -> list[str]:
    """Return all labels with a matching layer type."""

    return [
        layer_log.layer_label
        for layer_log in log.layer_logs.values()
        if layer_log.layer_type == layer_type
    ]


class _ConvReluLinear(nn.Module):
    """Tiny Conv -> ReLU -> Linear model."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(6, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass."""

        return self.linear(self.relu(self.conv(x)))


class _ConsecutiveSkip(nn.Module):
    """Tiny Conv -> ReLU -> Dropout -> Linear model."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(6, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass."""

        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.linear(x)


class _BranchingSkip(nn.Module):
    """Model where one skipped node feeds two downstream branches."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.source = nn.Linear(4, 4)
        self.relu = nn.ReLU()
        self.left = nn.Linear(4, 2)
        self.right = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass."""

        y = self.relu(self.source(x))
        return self.left(y) + self.right(y)


def test_skip_single_layer_chains_edge(tmp_path: Any) -> None:
    """Skipping one layer should connect its parent to its child."""

    log = tl.log_forward_pass(_ConvReluLinear(), torch.randn(1, 3, 8, 8))
    conv_label = _first_label(log, "conv2d")
    linear_label = _first_label(log, "linear")

    def skip_fn(layer_log: LayerLog) -> bool:
        """Skip ReLU layers."""

        return layer_log.layer_type == "relu"

    dot = _render_dot(log, tmp_path, skip_fn=skip_fn)

    assert "relu_1_2 [" not in dot
    assert f"{conv_label} -> {linear_label}" in dot


def test_skip_consecutive_layers_chains_through(tmp_path: Any) -> None:
    """Skipping consecutive layers should chain to the next visible node."""

    log = tl.log_forward_pass(_ConsecutiveSkip(), torch.randn(1, 3, 8, 8))
    conv_label = _first_label(log, "conv2d")
    linear_label = _first_label(log, "linear")

    def skip_fn(layer_log: LayerLog) -> bool:
        """Skip ReLU and Dropout layers."""

        return layer_log.layer_type in {"relu", "dropout"}

    dot = _render_dot(log, tmp_path, skip_fn=skip_fn)

    assert "relu_1_2 [" not in dot
    assert "dropout_1_3 [" not in dot
    assert f"{conv_label} -> {linear_label}" in dot


def test_skip_branching(tmp_path: Any) -> None:
    """Skipping a branching node should create direct edges to both branches."""

    log = tl.log_forward_pass(_BranchingSkip(), torch.randn(1, 4))
    source_label = _first_label(log, "linear")
    branch_labels = _labels(log, "linear")[1:]

    def skip_fn(layer_log: LayerLog) -> bool:
        """Skip the shared ReLU branch input."""

        return layer_log.layer_type == "relu"

    dot = _render_dot(log, tmp_path, skip_fn=skip_fn)

    assert "relu_1_2 [" not in dot
    assert f"{source_label} -> {branch_labels[0]}" in dot
    assert f"{source_label} -> {branch_labels[1]}" in dot


def test_skip_input_or_output_raises(tmp_path: Any) -> None:
    """skip_fn may not elide graph input or output layers."""

    log = tl.log_forward_pass(nn.Linear(4, 4), torch.randn(1, 4))

    def skip_fn(layer_log: LayerLog) -> bool:
        """Try to skip the input layer."""

        return layer_log.is_input_layer

    try:
        _render_dot(log, tmp_path, skip_fn=skip_fn)
    except ValueError as exc:
        assert "skip_fn cannot skip input or output layer" in str(exc)
        assert "input" in str(exc)
    else:
        raise AssertionError("Expected ValueError when skipping an input layer")


def test_skip_preserves_gradient_edge_kind(tmp_path: Any) -> None:
    """Skipping a forward node should preserve gradient edge styling."""

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 1))
    log = tl.log_forward_pass(model, torch.randn(1, 4), save_gradients=True)
    output = log[log.output_layers[0]].activation
    output.sum().backward()

    def skip_fn(layer_log: LayerLog) -> bool:
        """Skip ReLU layers."""

        return layer_log.layer_type == "relu"

    dot = _render_dot(log, tmp_path, skip_fn=skip_fn)

    assert "relu_1_2 [" not in dot
    assert GRADIENT_ARROW_COLOR in dot
