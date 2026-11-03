"""Preview and dry-run RecordContext fidelity tests."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._visualization.fastlog_preview import _build_preview_nodes, _make_node_spec_fn
from torchlens._visualization.node_spec import NodeSpec
from torchlens.errors import RecordContextFieldError
from torchlens.recording import RecordContext


class StaticGraph(nn.Module):
    """Static graph model for preview fidelity."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.layers = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the static graph."""

        return self.layers(x)


def test_missing_record_context_field_errors_in_preview_and_dry_run() -> None:
    """A nonexistent predicate field fails with RecordContextFieldError in both paths."""

    model = StaticGraph()
    x = torch.randn(1, 4)
    trace = tl.trace(model, x)

    def bad_predicate(ctx: RecordContext) -> bool:
        """Access a field outside the RecordContext schema."""

        return bool(ctx.recurrent_ops)

    dot = trace.preview_fastlog(predicate=bad_predicate)

    assert "exception" in dot.dot
    with pytest.raises(RecordContextFieldError):
        tl.recording.dry_run(model, x, save=bad_predicate)


def test_preview_nodes_include_short_label_keys() -> None:
    """Preview node lookup includes pass-free short labels."""

    model = StaticGraph()
    x = torch.randn(1, 4)
    trace = tl.trace(model, x)

    preview_nodes = _build_preview_nodes(trace, None)

    assert "relu_1" in preview_nodes


def test_preview_node_spec_consumes_short_label_keys() -> None:
    """Preview rendering reads the same short-label keys it writes."""

    model = StaticGraph()
    x = torch.randn(1, 4)
    trace = tl.trace(model, x)
    preview_nodes = _build_preview_nodes(trace, lambda ctx: True)
    node_spec_fn = _make_node_spec_fn(
        preview_nodes,
        color_kept="#00FF00",
        color_rejected="#FF0000",
        color_unreachable="#0000FF",
        color_predicate_error="#FFFF00",
        show_predicate_inputs=False,
        show_module_events=False,
    )

    styled = node_spec_fn(
        type(
            "ShortLabelOnlyLayer",
            (),
            {"layer_label": "missing_label", "short_layer_label": "relu_1"},
        )(),
        NodeSpec(lines=["relu"]),
    )

    assert styled.fillcolor == "#00FF00"
    assert "fastlog: kept" in styled.lines
