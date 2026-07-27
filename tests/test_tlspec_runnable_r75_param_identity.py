"""Capture-time parameter/state identity for gc-independent honest saves (r75 F2).

Round-74 (free-roam) root-caused a PRE-EXISTING, gc-timing-nondeterministic honest-save
over-refusal: the runnable producer resolved parameter-argument identity at SAVE time
through live references (``Param._param_ref``, released by postprocess ->
``_source_model_ref`` -> ``named_parameters()``), so a caller that never held the model
plus one ``gc.collect()`` between trace and save killed the whole chain. BN-like models
with two same-shape+dtype ``(C,)`` params then had NO identity rung left and refused
``unsupported_tensor_constant`` -- and, one stage further, the persistent-buffer slot
enumeration silently returned on the dead model, dropping never-forward-used buffers
(``num_batches_tracked``) from the declared universe so the embedded snapshot failed
strict binding with ``state_unexpected_key``. User-facing spelling:
``tl.trace(Model(), x).save(level="runnable")`` refused nondeterministically for any
BN-carrying model. Fail-closed (over-refusal), never a wrong verdict -- but a red gate.

The r75 closure resolves identity AT CAPTURE TIME, when the model is provably alive:

* ``_classify_arg_component`` snapshots the model-prep barcode onto the arg template
  (``LiteralTensor.param_barcode``); ``_match_parameter`` matches it against the cooked
  ``Param.barcode`` mirror as its first, gc-immune rung (live-reference, argument-name,
  and single-candidate rungs unchanged for model-alive captures);
* ``_add_persistent_buffer_slot_drafts`` rebuilds the persistent-buffer universe from
  the capture-boundary state snapshot + cooked ``Param`` addresses + the capture-time
  alias-topology snapshot when the live model is gone.

A genuinely-unmatched tensor constant (a non-parameter foreign literal) still refuses
typed -- the fix only ADDS a positive identity rung, never widens admission.
"""

from __future__ import annotations

import gc
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io.runnable import build_sparse_run_descriptor
from torchlens.options import CaptureOptions
from torchlens.runnable import PathFaithfulness, RunnableErrorCode, StateSource

_CAPTURE = CaptureOptions(intervention_ready=True, capture_container_structure=True, cache=False)


class _BNModel(nn.Module):
    """BatchNorm carrier: two same-shape+dtype ``(4,)`` params + an unused buffer."""

    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply eval-mode batch norm."""

        return self.bn(x)


_HOSTILE_CONSTANT = torch.randn(4, 4)


class _ForeignConstantModel(nn.Module):
    """Consumes a non-module, non-parameter tensor constant (no provenance)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add a free global tensor the sparse recipe cannot rebuild."""

        return x + _HOSTILE_CONSTANT


def _x() -> torch.Tensor:
    """Return the fixed probe input."""

    torch.manual_seed(0)
    return torch.randn(4, 4)


@pytest.mark.smoke
def test_r75_inline_model_gc_before_descriptor_preflights() -> None:
    """The r74 F2 repro direction: no model ref + gc.collect -> preflight passes."""

    trace = tl.trace(_BNModel().eval(), _x(), capture=_CAPTURE)  # model NOT held
    gc.collect()
    descriptor = build_sparse_run_descriptor(trace)
    assert descriptor.preflight.passed, [
        (diag.code, diag.message) for diag in descriptor.preflight.diagnostics
    ]


@pytest.mark.smoke
def test_r75_inline_model_gc_full_save_load_run(tmp_path: Path) -> None:
    """End-to-end: dead model + gc -> save embeds the FULL state universe and runs VERIFIED.

    Pins the second leg too: ``num_batches_tracked`` (never used by the eval forward, so
    absent from the graph) must still be declared, or the embedded snapshot refuses with
    ``state_unexpected_key``.
    """

    x = _x()
    trace = tl.trace(_BNModel().eval(), x, capture=_CAPTURE)  # model NOT held
    gc.collect()
    path = tmp_path / "bn.tlspec"
    trace.save(path, level="runnable", include_weights=True)

    loaded = tl.load(path)
    slot_names = {
        slot.state_binding.state_dict_name
        for slot in loaded.__dict__["_runnable_descriptor"].tensor_slots
        if slot.state_binding is not None
    }
    assert "bn.num_batches_tracked" in slot_names

    result = loaded.run(inputs=x.clone())
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert result.report.state_source is StateSource.EMBEDDED_CAPTURE_STATE


@pytest.mark.smoke
def test_r75_model_alive_matching_unchanged(tmp_path: Path) -> None:
    """Zero collateral: holding the model across the same gc still saves and verifies."""

    x = _x()
    model = _BNModel().eval()
    trace = tl.trace(model, x, capture=_CAPTURE)
    gc.collect()
    path = tmp_path / "alive.tlspec"
    trace.save(path, level="runnable", include_weights=True)
    result = tl.load(path).run(inputs=x.clone())
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert model is not None  # the strong hold is the point of the fixture


@pytest.mark.smoke
def test_r75_retrace_does_not_confuse_first_capture(tmp_path: Path) -> None:
    """Barcode identity is PER-CAPTURE: re-tracing re-stamps, the snapshot still matches.

    The template snapshot was taken during capture 1, so capture 2 re-stamping the same
    parameter objects must not break capture 1's deferred save (the snapshot, not the
    live registry, is the identity authority).
    """

    x = _x()
    model = _BNModel().eval()
    first = tl.trace(model, x, capture=_CAPTURE)
    second = tl.trace(model, x, capture=_CAPTURE)
    gc.collect()
    path = tmp_path / "first.tlspec"
    first.save(path, level="runnable", include_weights=True)
    result = tl.load(path).run(inputs=x.clone())
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert second is not None


@pytest.mark.smoke
def test_r75_unmatched_tensor_constant_still_refuses_typed() -> None:
    """Tripwire intact: a genuinely-unmatched foreign tensor literal still refuses typed."""

    with pytest.warns(UserWarning, match="no graph/source provenance"):
        trace = tl.trace(_ForeignConstantModel().eval(), _x(), capture=_CAPTURE)
    gc.collect()
    descriptor = build_sparse_run_descriptor(trace)
    assert not descriptor.preflight.passed
    codes = {diag.code for diag in descriptor.preflight.diagnostics}
    assert RunnableErrorCode.UNSUPPORTED_TENSOR_CONSTANT in codes
