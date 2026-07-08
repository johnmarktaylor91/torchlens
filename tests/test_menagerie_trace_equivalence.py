"""Regression gate for the bucket-A metadata-unify trace reduction and equivalence audit.

Two locked guarantees:

1. ``summarize_trace`` derives the full summary row -- including both identity hashes --
   from ONE already-captured trace, byte-identical to the old three-trace
   ``summarize_model(..., compute_identity_hashes=True)`` path. This pins the 3->1
   re-trace reduction (the safe, universal A1 win) so a future change cannot silently
   reintroduce the extra hash re-traces or drift a metadata field.

2. The validation-config vs metadata-config field-equivalence audit
   (``menagerie.trace_equivalence_audit``) reproduces the sprint finding: a plain CNN is
   inference_only-INSENSITIVE (only the ``mark_layer_depths`` distance fields diverge),
   while a fused-attention model is inference_only-SENSITIVE (the no-grad fused SDPA
   kernel collapses the attention into far fewer ops, changing op counts/hashes/FLOPs).
   This is the gate that proves which fields may be sourced from the validation trace
   and which must NOT be (sourcing them would silently change published metadata).
"""

from __future__ import annotations

import torch
from torch import nn

from menagerie.structural_digest import (
    _trace_for_digest,
    architecture_distinctness_hash,
    structural_fingerprint,
)
from menagerie.trace_equivalence_audit import (
    KNOWN_DIVERGENT_FIELDS,
    MEASUREMENT_COLUMNS,
    PROVENANCE_COLUMNS,
)
from menagerie.trace_summary import (
    TRACE_SUMMARY_COLUMNS,
    summarize_model,
    summarize_trace,
)


class _CNN(nn.Module):
    """Plain conv/norm/pool/linear stack -- inference_only insensitive."""

    def __init__(self) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(3, 8, 3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.r = nn.ReLU()
        self.c2 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.r(self.bn(self.c1(x)))
        x = self.c2(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class _Attn(nn.Module):
    """MultiheadAttention model -- fused SDPA collapses under no_grad."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.MultiheadAttention(16, 4, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _weights = self.a(x, x, x)
        return out


def _summarize_via_observer(model: nn.Module, example: torch.Tensor) -> dict:
    """Summarize from a VALIDATION-config trace, exactly as the worker observer does."""

    from torchlens.user_funcs import _validate_forward_pass_torch

    captured: dict[str, dict] = {}

    def observe(trace: object) -> None:
        captured["row"] = summarize_trace("m", trace, "")

    _validate_forward_pass_torch(model, example, validate_metadata=True, _trace_observer=observe)
    return captured["row"]


def test_summarize_trace_single_trace_matches_old_three_trace_path() -> None:
    """``summarize_trace`` off one trace == old ``summarize_model`` three-trace output."""

    model = _CNN()
    model.eval()
    example = torch.randn(2, 3, 16, 16)

    # Old public path re-traced 3x (1 summary + 2 hash entry points).
    old_row = summarize_model("m", model, example, "recipe", compute_identity_hashes=True)

    # New path: one metadata-config trace, summary + both hashes read off it.
    trace = _trace_for_digest(model, example)
    new_row = summarize_trace("m", trace, "recipe", compute_identity_hashes=True)

    assert tuple(new_row) == TRACE_SUMMARY_COLUMNS
    # forward_peak_memory_* are per-capture runtime measurements (RSS delta /
    # tracemalloc peak), nondeterministic across separate traces by construction
    # since torchlens 45537c6e made them live; every structural/metadata field
    # must still match byte-identically.
    structural = {k: v for k, v in new_row.items() if k not in MEASUREMENT_COLUMNS}
    assert structural == {k: v for k, v in old_row.items() if k not in MEASUREMENT_COLUMNS}
    for column in MEASUREMENT_COLUMNS:
        assert new_row[column] >= 0
        assert old_row[column] >= 0


def test_unified_hashes_match_public_entry_point_retrace() -> None:
    """The single-trace hashes equal the old re-traced public-entry-point hashes."""

    model = _CNN()
    model.eval()
    example = torch.randn(2, 3, 16, 16)

    trace = _trace_for_digest(model, example)
    row = summarize_trace("m", trace, "recipe", compute_identity_hashes=True)

    assert row["graph_shape_hash"] == architecture_distinctness_hash(model, example)
    assert row["structural_fingerprint_hash"] == structural_fingerprint(model, example)


def test_summarize_model_traces_once(monkeypatch) -> None:
    """``summarize_model`` builds exactly ONE trace even when hashes are computed."""

    import menagerie.structural_digest as structural_digest
    import menagerie.trace_summary as trace_summary

    calls = {"n": 0}
    original = structural_digest._trace_for_digest

    def counting(model: object, example: object) -> object:
        calls["n"] += 1
        return original(model, example)

    monkeypatch.setattr(trace_summary, "_trace_for_digest", counting)

    summarize_model(
        "m", _CNN().eval(), torch.randn(2, 3, 16, 16), "r", compute_identity_hashes=True
    )
    assert calls["n"] == 1


def test_cnn_is_inference_only_insensitive_except_distance_fields() -> None:
    """CNN: validation-config and metadata-config summaries agree except distance fields.

    A plain CNN does not hit the fused-SDPA path, so its only divergence is the
    ``mark_layer_depths`` distance set -- proving those fields (and ONLY those) are the
    universal divergence axis for inference_only-insensitive models.
    """

    example = torch.randn(2, 3, 16, 16)
    val_row = _summarize_via_observer(_CNN().eval(), example)
    meta_row = summarize_trace("m", _trace_for_digest(_CNN().eval(), example), "")

    diffs = {
        column
        for column in TRACE_SUMMARY_COLUMNS
        if column not in PROVENANCE_COLUMNS
        and column not in MEASUREMENT_COLUMNS
        and val_row.get(column) != meta_row.get(column)
    }
    assert diffs <= KNOWN_DIVERGENT_FIELDS, (
        f"unexpected CNN divergence: {diffs - KNOWN_DIVERGENT_FIELDS}"
    )


def test_single_trace_row_is_self_consistent_for_dynamic_graphs() -> None:
    """For an instance-varying graph the unified row is internally self-consistent.

    The old three-trace path stitched summary fields from trace #1 with hashes from two
    SEPARATE re-traces; for a dynamic graph (e.g. DGCNN's input-dependent kNN) each
    re-trace produced a different graph, so the persisted row mixed three different
    graphs. The single-trace path draws every field -- including both hashes -- from ONE
    trace, so the row describes exactly the graph it summarizes. This pins that the
    unified hashes are the SAME trace's attributes (no cross-trace stitching).
    """

    class _DynamicWidth(nn.Module):
        """Graph whose op set depends on a per-call random branch."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if torch.rand(()) < 0.5:
                x = x + x
            return x * 2.0

    model = _DynamicWidth()
    model.eval()
    example = torch.randn(2, 4)
    trace = _trace_for_digest(model, example)
    row = summarize_trace("m", trace, "")

    assert row["graph_shape_hash"] == str(getattr(trace, "graph_shape_hash", "") or "")
    assert row["structural_fingerprint_hash"] == str(
        getattr(trace, "_raw_event_shape_hash", "") or ""
    )


def test_attention_is_inference_only_sensitive() -> None:
    """Attention: fused SDPA under no_grad collapses ops, so op counts/hashes DIVERGE.

    This is the gate that forbids silently sourcing attention-model metadata from the
    validation trace: the validation-config (inference_only=False) graph is materially
    different from the published metadata-config (inference_only=True) graph.
    """

    example = torch.randn(2, 5, 16)
    val_row = _summarize_via_observer(_Attn().eval(), example)
    meta_row = summarize_trace("m", _trace_for_digest(_Attn().eval(), example), "")

    # The no_grad metadata config uses the fused SDPA kernel -> FEWER captured ops.
    assert meta_row["n_compute_ops"] != val_row["n_compute_ops"]
    assert meta_row["graph_shape_hash"] != val_row["graph_shape_hash"]
