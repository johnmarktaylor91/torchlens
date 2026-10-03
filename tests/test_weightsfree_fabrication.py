"""Fabrication pins (W1-FAB, memo L2/L5, sec 8.1 items 5-6), RED-first.

The generalized rule (D6): weights-free, a value comparison that cannot
answer resolves to ``unknown``, never ``changed``; no record and no
completeness-witness flag may rest on an ``unknown``. Fail-closed in the
HONEST direction: absence of a write claim, never a fabricated one.

The CI pins assert the fabrication DELTA and its kinds with the fix
disabled, never absolute record counts (memo correction C1).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from test_weightsfree_fixtures import (
    ConvBnPool,
    build_twins,
    meta_like,
    weightsfree_trace,
)

import torchlens as tl


def _write_rows(trace) -> list[tuple[str, str, object]]:
    """(kind, source_func_name, value_changed) per recorded buffer write."""

    rows = []
    for label in trace.buffer_write_ops:
        layer = trace[label]
        rows.append(
            (
                layer.buffer_write_kind,
                layer.buffer_source_func_name,
                layer.buffer_value_changed,
            )
        )
    return rows


def test_tristate_comparison_is_unknown_on_meta() -> None:
    """The comparison primitive answers None (unknown) weights-free."""

    from torchlens.backends.torch.buffer_writes import _tensor_equal_tristate

    a = torch.empty(3, device="meta")
    b = torch.empty(3, device="meta")
    assert _tensor_equal_tristate(a, b) is None
    real = torch.ones(3)
    assert _tensor_equal_tristate(real, real.clone()) is True
    assert _tensor_equal_tristate(real, real + 1) is False


def test_eval_bn_meta_twin_no_phantom_writes() -> None:
    """Eval-mode BatchNorm meta twin: zero fabricated inplace/reassign rows.

    L2's measured shape: 3 phantom rows per BatchNorm (running_mean,
    running_var, num_batches_tracked) — a real eval run never touches
    num_batches_tracked at all.
    """

    real, meta = build_twins(ConvBnPool)
    x = torch.randn(1, 3, 8, 8)
    tr_real = tl.trace(real, x)
    tr_meta = weightsfree_trace(meta, meta_like(x))

    real_kinds = sorted((k, f) for k, f, _ in _write_rows(tr_real))
    meta_kinds = sorted((k, f) for k, f, _ in _write_rows(tr_meta))
    assert meta_kinds == real_kinds, (
        f"buffer-write parity broken: real={real_kinds} meta={meta_kinds}"
    )
    assert not any(
        kind in ("inplace", "data_reassign", "reassign") for kind, _, _ in _write_rows(tr_meta)
    ), "eval-mode meta twin fabricated a non-declared write"


def test_declared_fused_writes_survive_weightsfree() -> None:
    """W1-BUF-2: declared (fused) mutations are recorded as HYPOTHESIS rows.

    The mechanism is the SAME shipped fused-mutator classification the real
    path uses — record existence never keys on a value comparison — so
    weights-free parity holds by construction; only ``value_changed`` is
    unknowable and rides the tri-state.
    """

    real, meta = build_twins(ConvBnPool)
    x = torch.randn(1, 3, 8, 8)
    tr_real = tl.trace(real, x)
    tr_meta = weightsfree_trace(meta, meta_like(x))
    real_fused = [r for r in _write_rows(tr_real) if r[0] == "fused"]
    meta_fused = [r for r in _write_rows(tr_meta) if r[0] == "fused"]
    assert len(meta_fused) == len(real_fused)
    assert len(meta_fused) == 2  # running_mean + running_var
    for _kind, source, value_changed in meta_fused:
        assert value_changed is None, (
            "weights-free fused write claims a value verdict it cannot make"
        )
        assert source, "fused write lost its source op"


def test_train_mode_bn_meta_twin_parity() -> None:
    """Train-mode BN: version-witnessed inplace writes keep parity."""

    torch.manual_seed(0)
    real = ConvBnPool()
    with torch.device("meta"):
        meta = ConvBnPool()
    real.train()
    meta.train()
    x = torch.randn(2, 3, 8, 8)
    tr_real = tl.trace(real, x)
    tr_meta = weightsfree_trace(meta, meta_like(x))
    assert sorted((k, f) for k, f, _ in _write_rows(tr_meta)) == sorted(
        (k, f) for k, f, _ in _write_rows(tr_real)
    )
    assert tl.hash.trace(tr_real) == tl.hash.trace(tr_meta)


def test_no_witness_flag_rests_on_unknown() -> None:
    """The opaque-host-write flag is set by the MODEL, never the substrate.

    L2's second half: on meta, ``reconcile()`` read "cannot know" as
    "changed" and set the UNVERIFIABLE flag on every buffer-holding model.
    """

    from torchlens.backends.torch.completeness_witness import (
        _HOST_ESCAPE_MUTABLE_WRITEBACK,
    )

    _, meta = build_twins(ConvBnPool)
    tr_meta = weightsfree_trace(meta, torch.empty(1, 3, 8, 8, device="meta"))
    assert tr_meta not in _HOST_ESCAPE_MUTABLE_WRITEBACK


def test_fabrication_delta_with_fix_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """RED-capability pin: disabling the tri-state fix fabricates rows again.

    Asserts the DELTA (extra journal rows with the fix reverted to the
    historical cannot-compare-means-changed reading), proving the pins above
    are load-bearing rather than vacuously green.
    """

    from torchlens.backends.torch import buffer_writes as bw

    _, meta = build_twins(ConvBnPool)
    x = torch.empty(1, 3, 8, 8, device="meta")
    clean = _write_rows(weightsfree_trace(meta, x))

    def legacy(left: torch.Tensor, right: torch.Tensor):
        if left.is_meta or right.is_meta:
            return False  # the historical fabrication: unknown read as changed
        return bw._tensor_equal(left, right)

    monkeypatch.setattr(bw, "_tensor_equal_tristate", legacy)
    from torchlens._errors import WeightsfreeIntegrityError

    try:
        fabricated = _write_rows(weightsfree_trace(meta, x))
    except WeightsfreeIntegrityError as exc:
        # The D22 settlement net catches the fabrication before the trace can
        # settle (the legacy reading raises the opaque host-write witness on
        # the substrate) — the STRONGER proof the fix stack is load-bearing.
        assert exc.fields["code"] == "structure_only_settlement_incoherent"
    else:
        assert len(fabricated) > len(clean), (
            "the tri-state fix is not load-bearing: disabling it fabricated nothing"
        )
        extra_kinds = sorted(k for k, _, _ in fabricated)
        assert "data_reassign" in extra_kinds or "inplace" in extra_kinds


def test_rescue_state_sweep_never_fabricates_on_meta() -> None:
    """L5: the rescue state-change sweep resolves unknown, never changed."""

    from torchlens.backends.torch.rescue import _restore_changed_state

    with torch.device("meta"):
        model = nn.Linear(4, 4)
    snapshot = {f"param:{name}": tensor for name, tensor in model.named_parameters()}
    snapshot.update({f"buffer:{name}": tensor for name, tensor in model.named_buffers()})
    changed = _restore_changed_state(model, snapshot)
    assert changed == (), f"rescue fabricated state changes on a storage-less substrate: {changed}"
