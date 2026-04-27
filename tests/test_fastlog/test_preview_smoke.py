"""Smoke tests for fastlog ModelLog preview rendering."""

from __future__ import annotations

import torch
from pathlib import Path
from torch import nn

import torchlens as tl
from torchlens.fastlog.exceptions import RecordContextFieldError
from torchlens.fastlog.types import RecordContext


def _mlp() -> nn.Module:
    """Return a small two-layer MLP."""

    return nn.Sequential(nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 2))


def test_preview_fastlog_renders_on_mlp(tmp_path: Path) -> None:
    """Preview rendering returns Graphviz source for a small MLP."""

    model_log = tl.log_forward_pass(_mlp(), torch.randn(1, 4))
    dot = model_log.preview_fastlog(
        keep_op=lambda ctx: ctx.layer_type == "linear",
        vis_outpath=str(tmp_path / "preview"),
        vis_save_only=True,
    )
    assert "digraph" in dot
    assert "fastlog" in dot


def test_preview_fastlog_catches_record_context_field_error(tmp_path: Path) -> None:
    """Missing RecordContext fields are caught and rendered as hot nodes."""

    model_log = tl.log_forward_pass(_mlp(), torch.randn(1, 4))

    def bad_predicate(ctx: RecordContext) -> bool:
        """Access a field outside the RecordContext schema."""

        try:
            getattr(ctx, "not_a_record_context_field")
        except RecordContextFieldError:
            raise
        return True

    dot = tl.preview_fastlog(
        model_log,
        predicate=bad_predicate,
        color_predicate_error="#FF7AB6",
        vis_outpath=str(tmp_path / "preview_error"),
        vis_save_only=True,
    )
    assert "#FF7AB6" in dot
    assert "RecordContext has no field" in dot
