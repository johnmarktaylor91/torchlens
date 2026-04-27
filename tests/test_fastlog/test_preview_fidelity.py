"""Preview and dry-run RecordContext fidelity tests."""

from __future__ import annotations

from dataclasses import asdict

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.fastlog import RecordContext, RecordContextFieldError
from torchlens.visualization.fastlog_preview import _build_preview_nodes


class StaticGraph(nn.Module):
    """Static graph model for preview fidelity."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.layers = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the static graph."""

        return self.layers(x)


@pytest.mark.xfail(
    reason=(
        "Preview currently synthesizes ModelLog layer contexts only; dry_run includes "
        "module events and uses raw labels, so exact field-by-field parity is pending."
    )
)
def test_preview_and_dry_run_contexts_match_field_by_field() -> None:
    """Preview-synthesized and real dry-run contexts match field-by-field."""

    model = StaticGraph()
    x = torch.randn(1, 4)
    model_log = tl.log_forward_pass(model, x)
    trace = tl.fastlog.dry_run(model, x, keep_op=lambda ctx: True, include_source_events=True)
    preview_nodes = _build_preview_nodes(model_log, lambda ctx: True)
    preview_contexts = [node.ctx for node in dict.fromkeys(preview_nodes.values())]
    real_contexts = [
        ctx for ctx in trace.contexts if ctx.kind in {"input", "op"} and ctx.layer_type != "output"
    ]

    assert [asdict(ctx) for ctx in preview_contexts] == [asdict(ctx) for ctx in real_contexts]


def test_missing_record_context_field_errors_in_preview_and_dry_run() -> None:
    """A nonexistent predicate field fails with RecordContextFieldError in both paths."""

    model = StaticGraph()
    x = torch.randn(1, 4)
    model_log = tl.log_forward_pass(model, x)

    def bad_predicate(ctx: RecordContext) -> bool:
        """Access a field outside the RecordContext schema."""

        return bool(ctx.recurrent_group)

    dot = model_log.preview_fastlog(predicate=bad_predicate)

    assert "exception" in dot
    with pytest.raises(RecordContextFieldError):
        tl.fastlog.dry_run(model, x, keep_op=bad_predicate)
