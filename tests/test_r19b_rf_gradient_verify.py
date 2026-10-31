"""Regression tests for RF gradient/verify honesty and rule-install rollback (r19b).

Covers hardening findings:

* W2A3-04 -- ``cross_batch_influence`` must not report a false ``True`` when a
  batch-moving transform repositions the batch axis, while still detecting a
  genuine cross-batch dependence (the ``undeclared_batch`` validation tripwire).
* W2A3-07 -- ``verify(direction="projective")`` must report the sampled far
  target element as ``target_unit``, not the source's validation unit.
* W2A3-11 -- installing the built-in RF rule pack must be transactional: a
  mid-loop reload failure must not leak a partial registry or advance the epoch.
* W2A3-15a -- the complete-index gradient contract wraps negative indices with
  Python semantics (documented, guarded here so the doc cannot silently drift).
"""

from __future__ import annotations

import importlib

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.receptive_field import _gradient, _rules
from torchlens.receptive_field._engine_geometry import _AxisState, _Full, _InputState
from torchlens.receptive_field._errors import ReceptiveFieldError
from torchlens.receptive_field._gradient import _batch_semantics, _normalize_unit


def _state(batch_axis: int, output_axes: tuple[int | None, ...]) -> _InputState:
    """Build a minimal engine state with an explicit batch/output-axis mapping."""

    axes = tuple(_AxisState(_Full(), out, "pointwise", "test") for out in output_axes)
    return _InputState(
        input_op_label="in",
        io_role="input.x",
        input_shape=(),
        axes=axes,
        taint=None,
        notes=(),
        rule="test",
        batch_axis=batch_axis,
    )


def _mask(shape: tuple[int, ...], live: tuple[tuple[int, ...], ...]) -> torch.Tensor:
    """Return a Boolean support mask with the given coordinates set live."""

    mask = torch.zeros(shape, dtype=torch.bool)
    for coord in live:
        mask[coord] = True
    return mask


def _backward_trace(model: nn.Module, inputs: torch.Tensor) -> object:
    """Capture a backward-ready, graph-connected trace for RF probing."""

    capture = tl.options.CaptureOptions(backward_ready=True)
    return tl.trace(model, inputs, capture=capture, save_mode="reference")


# ---------------------------------------------------------------------------
# W2A3-04: cross_batch_influence coordinate honesty (unit-level, both spaces).
# ---------------------------------------------------------------------------


def test_cross_batch_semantics_receptive_moved_axis_no_false_true() -> None:
    """Repositioned batch axis: seeded sample owns all support -> not cross-batch."""

    # input axis 0 (batch) -> output axis 1 after a permute-like move.
    state = _state(batch_axis=0, output_axes=(1, 0, None, 3))
    mask = _mask((2, 1, 2, 3), ((1, 0, 0, 2),))  # support only at input batch 1
    unit = (0, 1, 0, 2)  # target-space; batch coordinate lives at output axis 1 -> 1
    batch_support, cross_batch = _batch_semantics(mask, state, unit)
    assert batch_support == (1,)
    assert cross_batch is False


def test_cross_batch_semantics_receptive_moved_axis_detects_genuine() -> None:
    """Repositioned batch axis: support in another sample -> genuine cross-batch."""

    state = _state(batch_axis=0, output_axes=(1, 0, None, 3))
    mask = _mask((2, 1, 2, 3), ((0, 0, 0, 2), (1, 0, 0, 2)))  # batches 0 and 1
    unit = (0, 1, 0, 2)  # seeded batch coordinate is 1
    batch_support, cross_batch = _batch_semantics(mask, state, unit)
    assert batch_support == (0, 1)
    assert cross_batch is True


def test_cross_batch_semantics_projective_moved_axis_no_false_true() -> None:
    """Projective mirror: mask lives in target space, unit in source space."""

    # source axis 0 (batch) -> target axis 1.
    state = _state(batch_axis=0, output_axes=(1, 0, None, 3))
    mask = _mask((2, 2, 1, 3), ((0, 1, 0, 2),))  # target batch (axis 1) index 1
    unit = (1, 0, 0, 2)  # source-space; batch coordinate at source axis 0 -> 1
    batch_support, cross_batch = _batch_semantics(mask, state, unit, projective=True)
    assert batch_support == (1,)
    assert cross_batch is False


def test_cross_batch_semantics_projective_moved_axis_detects_genuine() -> None:
    """Projective genuine cross-batch survives the coordinate remap."""

    state = _state(batch_axis=0, output_axes=(1, 0, None, 3))
    mask = _mask((2, 2, 1, 3), ((0, 0, 0, 2), (0, 1, 0, 2)))  # target batches 0 and 1
    unit = (1, 0, 0, 2)  # seeded source batch coordinate is 1
    batch_support, cross_batch = _batch_semantics(mask, state, unit, projective=True)
    assert batch_support == (0, 1)
    assert cross_batch is True


def test_cross_batch_semantics_fully_coupled_axis_uses_support_span() -> None:
    """A fully coupled batch axis (batch_norm-like) keeps the tripwire armed."""

    # Batch axis is coupled with no 1:1 output counterpart (output_axis None).
    state = _state(batch_axis=0, output_axes=(None, None, None, None))
    unit = (0, 0, 0, 0)

    # Support spanning multiple samples IS cross-batch even though the seeded
    # sample cannot be pinned to a single coordinate (the batch_norm case).
    multi = _mask((3, 1, 2, 3), ((0, 0, 0, 0), (1, 0, 0, 0), (2, 0, 0, 0)))
    batch_support, cross_batch = _batch_semantics(multi, state, unit)
    assert batch_support == (0, 1, 2)
    assert cross_batch is True

    # A single supported sample under the same coupling is not cross-batch.
    single = _mask((3, 1, 2, 3), ((1, 0, 0, 0),))
    batch_support, cross_batch = _batch_semantics(single, state, unit)
    assert batch_support == (1,)
    assert cross_batch is False


# ---------------------------------------------------------------------------
# W2A3-04: end-to-end permute-after-conv (real capture).
# ---------------------------------------------------------------------------


class _MoveBatch(nn.Module):
    """1x1 conv followed by a permute that moves the batch axis 0 -> 1."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Return the permuted convolution output."""

        return self.conv(value).permute(2, 0, 1, 3)


def test_cross_batch_influence_false_after_permute_receptive() -> None:
    """A pure batch-axis reposition must not fabricate cross-batch influence."""

    x = torch.arange(12.0).reshape(2, 1, 2, 3).requires_grad_()
    trace = _backward_trace(_MoveBatch(), x)
    result = trace.output_ops[0].receptive_field.probe_grad(
        (0, 1, 0, 2), input_op=trace.input_ops[0]
    )
    assert result.cross_batch_influence is False


def test_cross_batch_influence_false_after_permute_projective() -> None:
    """The projective probe shares the fixed coordinate handling."""

    x = torch.arange(12.0).reshape(2, 1, 2, 3).requires_grad_()
    trace = _backward_trace(_MoveBatch(), x)
    result = trace.input_ops[0].projective_field.probe_grad(
        (1, 0, 0, 2), target=trace.output_ops[0]
    )
    assert result.cross_batch_influence is False


# ---------------------------------------------------------------------------
# W2A3-07: verify(projective) reports the sampled far target element.
# ---------------------------------------------------------------------------


def test_verify_projective_reports_sampled_far_target_unit() -> None:
    """The reported ``target_unit`` must be the sampled far element, not the seed."""

    model = nn.Sequential(nn.Conv2d(1, 1, 3, padding=1), nn.Conv2d(1, 1, 3, padding=1))
    with torch.no_grad():
        for layer in model:
            layer.weight.fill_(1.0)
            layer.bias.zero_()
    log = _backward_trace(model, torch.randn(1, 1, 5, 5))

    verification = tl.receptive_field.verify(log, direction="projective")
    check = next(c for c in verification.empirical_adjoint if c.source_label == "conv2d_1_1:1")
    assert check.source_unit == (0, 0, 2, 2)
    # The projective field of conv2d_1_1 at (2, 2) spreads to a 3x3 neighborhood
    # in output space; its first (lexicographically smallest) support element is
    # (0, 0, 1, 1). The reported target_unit must equal that sampled element, and
    # must NOT parrot the source's validation unit.
    assert check.target_unit == (0, 0, 1, 1)
    assert check.target_unit != check.source_unit

    # Independent cross-check: the sampled element really is the far support[0].
    projective = log["conv2d_1_1:1"].projective_field.probe_grad(
        (0, 0, 2, 2), target=log.output_ops[0]
    )
    support = torch.nonzero(projective.support_mask, as_tuple=False)
    assert tuple(int(v) for v in support[0].tolist()) == check.target_unit


# ---------------------------------------------------------------------------
# W2A3-11: transactional built-in rule install.
# ---------------------------------------------------------------------------


def test_builtin_rules_install_rolls_back_on_reload_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mid-loop reload failure must leave the registry and epoch untouched."""

    saved_rules = dict(_rules._RF_RULES)
    saved_epoch = _rules._RF_RULES_EPOCH
    try:
        _rules._RF_RULES.clear()  # force the "registry empty" install path
        pre_epoch = _rules._RF_RULES_EPOCH
        real_reload = importlib.reload
        calls = {"n": 0}

        def flaky_reload(module: object) -> object:
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("synthetic second-reload failure")
            return real_reload(module)  # type: ignore[arg-type]

        monkeypatch.setattr(importlib, "reload", flaky_reload)
        with pytest.raises(RuntimeError, match="synthetic second-reload failure"):
            with _gradient._builtin_rules_when_registry_empty():
                pass  # pragma: no cover - never reached, install raises first

        assert calls["n"] >= 2  # the failure really fired mid reload loop
        assert len(_rules._RF_RULES) == 0  # no partial registry leaked
        assert pre_epoch == _rules._RF_RULES_EPOCH  # epoch fully restored
    finally:
        _rules._RF_RULES.clear()
        _rules._RF_RULES.update(saved_rules)
        _rules._RF_RULES_EPOCH = saved_epoch


# ---------------------------------------------------------------------------
# W2A3-15a: complete-index negative-wrap contract (documented behavior guard).
# ---------------------------------------------------------------------------


def test_normalize_unit_wraps_negative_indices() -> None:
    """Negative indices wrap per axis with Python semantics, then bound-check."""

    shape = (2, 3, 4)
    assert _normalize_unit((-1, -1, -1), shape, "op") == (1, 2, 3)
    assert _normalize_unit((0, -3, 2), shape, "op") == (0, 0, 2)
    with pytest.raises(ReceptiveFieldError):
        _normalize_unit((-3, 0, 0), shape, "op")  # wraps below zero -> out of bounds
