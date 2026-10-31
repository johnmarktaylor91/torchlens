"""L5 M6 pins: rendering L1's ``Layer.shape_summary`` (across-pass shape row).

The data is L1's (memo 1.2: ONE field, slate-7.3 semantics, rendered
VERBATIM); L5 renders it as one plain-text row directly after the title row
on rolled multi-pass nodes whose passes vary in shape, escaped by the S5
choke point like every other row. The ``"shape_summary"`` selector token
joins ``node_label_fields`` with skip-when-absent semantics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.visualization._render_nodes import compute_default_node_lines


class VaryingRecurrent(nn.Module):
    """Variable-length recurrence: shape varies across passes."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for n in (4, 3, 2):
            x = torch.relu(self.fc(x[:n]))
        return x.sum()


class UniformRecurrent(nn.Module):
    """Fixed-shape recurrence: summary is None (no row)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(3):
            x = torch.relu(self.fc(x))
        return x.sum()


@pytest.fixture(scope="module")
def varying_log() -> Any:
    log = tl.trace(VaryingRecurrent(), torch.randn(4, 4))
    try:
        yield log
    finally:
        log.cleanup()


@pytest.fixture(scope="module")
def uniform_log() -> Any:
    log = tl.trace(UniformRecurrent(), torch.randn(4, 4))
    try:
        yield log
    finally:
        log.cleanup()


def _draw(log: tl.Trace, tmp_path: Path, **kwargs: Any) -> str:
    tmp_path.mkdir(parents=True, exist_ok=True)
    return log.draw(
        vis_save_only=True,
        file_format="svg",
        path=str(tmp_path / "graph"),
        **kwargs,
    )


def _varying_layer(log: tl.Trace) -> Any:
    return next(layer for layer in log._layer_logs.values() if len(layer.ops) > 1)


def test_rolled_varying_layer_gets_summary_row(varying_log: tl.Trace) -> None:
    layer = _varying_layer(varying_log)
    summary = layer.shape_summary
    assert isinstance(summary, str) and summary
    lines = compute_default_node_lines(layer, vis_mode="rolled")
    # Rendered VERBATIM, positioned directly after the title row.
    assert lines[1] == summary


def test_summary_row_is_escaped_at_the_choke_point(varying_log: tl.Trace, tmp_path: Path) -> None:
    """The summary contains '->' or '-'; '>' must leave through html.escape."""

    layer = _varying_layer(varying_log)
    summary = layer.shape_summary
    dot = _draw(varying_log, tmp_path, vis_mode="rolled")
    if ">" in summary:
        assert summary.replace(">", "&gt;") in dot
        assert f">{summary}<" not in dot
    else:
        assert summary in dot


def test_uniform_recurrence_has_no_summary_row(uniform_log: tl.Trace) -> None:
    layer = _varying_layer(uniform_log)
    assert layer.shape_summary is None
    lines = compute_default_node_lines(layer, vis_mode="rolled")
    assert all("->" not in line or "params" in line for line in lines[:2])


def test_single_pass_and_per_pass_nodes_have_no_summary_row(
    varying_log: tl.Trace,
) -> None:
    layer = _varying_layer(varying_log)
    summary = layer.shape_summary
    # Per-pass Op nodes (unrolled) carry no aggregate summary row.
    for op in layer.ops.values():
        lines = compute_default_node_lines(op, vis_mode="unrolled")
        assert summary not in lines


def test_shape_summary_selector_token(varying_log: tl.Trace, tmp_path: Path) -> None:
    layer = _varying_layer(varying_log)
    summary = layer.shape_summary
    lines = compute_default_node_lines(
        layer,
        vis_mode="rolled",
        node_label_fields=["name", "shape_summary"],
    )
    assert lines == [layer.layer_label, summary]
    # Skip-when-absent: a single-pass layer contributes no row for the token.
    single = next(lay for lay in varying_log._layer_logs.values() if len(lay.ops) == 1)
    single_lines = compute_default_node_lines(
        single, vis_mode="rolled", node_label_fields=["name", "shape_summary"]
    )
    assert single_lines == [single.layer_label]


def test_unknown_selector_still_refuses(varying_log: tl.Trace) -> None:
    layer = _varying_layer(varying_log)
    with pytest.raises(Exception) as excinfo:
        compute_default_node_lines(layer, vis_mode="rolled", node_label_fields=["no_such_field"])
    assert excinfo.value.fields["code"] == "node_label_field_invalid"
