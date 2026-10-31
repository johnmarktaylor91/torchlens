"""The ``"dtype"`` node label field (walk U06.3).

``draw(node_label_fields=[..., "dtype", ...])`` adds the node's output dtype
as one row, framework namespace dropped (``"float16"``, the Keras
``show_dtype`` convention), so mixed-precision graphs show where the dtype
changes. Skip-when-absent like ``"params"`` and ``"shape_summary"``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.visualization._label_format import format_dtype
from torchlens.visualization._render_nodes import compute_default_node_lines


class HalfTail(nn.Module):
    """Float32 linear followed by a cast to float16."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).to(torch.float16) * 2


@pytest.fixture(scope="module")
def half_log() -> Any:
    log = tl.trace(HalfTail(), torch.randn(2, 4))
    try:
        yield log
    finally:
        log.cleanup()


def test_dtype_token_renders_each_nodes_output_dtype(half_log: tl.Trace) -> None:
    by_dtype: dict[str, list[str]] = {}
    for layer in half_log.layer_logs.values():
        lines = compute_default_node_lines(
            layer, vis_mode="rolled", node_label_fields=["name", "dtype"]
        )
        assert lines[0] == layer.layer_label
        assert lines[1:] == [format_dtype(layer.dtype)]
        by_dtype.setdefault(lines[1], []).append(layer.layer_type)
    assert "linear" in by_dtype["float32"]
    assert "mul" in by_dtype["float16"]


def test_dtype_token_reaches_the_drawn_graph(half_log: tl.Trace, tmp_path: Path) -> None:
    pytest.importorskip("graphviz")
    dot = half_log.draw(
        vis_save_only=True,
        file_format="svg",
        path=str(tmp_path / "graph"),
        node_label_fields=["name", "dtype"],
    )
    assert "float16" in dot
    assert "float32" in dot


def test_format_dtype_drops_namespace_and_skips_absent() -> None:
    assert format_dtype(torch.bfloat16) == "bfloat16"
    assert format_dtype("mlx.core.float32") == "float32"
    assert format_dtype("int64") == "int64"
    assert format_dtype(None) is None
    assert format_dtype("") is None
