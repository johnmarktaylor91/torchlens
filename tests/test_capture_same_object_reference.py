"""A same-object return saved by reference is the live object downstream ops consume.

Eval-mode dropout, ``contiguous()`` on a contiguous tensor and ``type()`` with an
unchanged dtype return their receiver. TorchLens logs such a call against a
fresh copy so the receiver keeps its own label; under ``save_mode="reference"``
that copy used to be the saved payload too. Nothing downstream consumes the
copy, so the op was a dead end as a gradient source: ``autograd.grad`` from any
later op to its payload came back empty, and a projective field seeded there
failed. The payload is now the live object; a mutating same-object return
(``add_``) and the default copy mode keep their copies.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

import torchlens as tl

_ARMED = {"backward_ready": True, "save_mode": "reference"}
_SEQ = 5
_SEED_POSITION = 3


class _PassThroughChain(nn.Module):
    """Linear, then three same-object returns, then a second linear."""

    def __init__(self) -> None:
        super().__init__()
        self.pre = nn.Linear(4, 4)
        self.drop = nn.Dropout(0.25)
        self.post = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(self.pre(x))
        h = h.contiguous()
        h = h.type(h.dtype)
        return self.post(h)


class _InPlaceAdd(nn.Module):
    """A linear whose output is bumped in place before a second linear."""

    def __init__(self) -> None:
        super().__init__()
        self.pre = nn.Linear(4, 4)
        self.post = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.pre(x))
        h.add_(1.0)
        return self.post(h)


class _CausalAttentionAfterDropout(nn.Module):
    """One causal attention head whose input passes an eval-mode dropout first."""

    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.pre = nn.Linear(4, width)
        self.drop = nn.Dropout(0.1)
        self.q = nn.Linear(width, width)
        self.k = nn.Linear(width, width)
        self.v = nn.Linear(width, width)
        self.proj = nn.Linear(width, width)
        self.scale = 1.0 / math.sqrt(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(self.pre(x))
        scores = (self.q(h) @ self.k(h).transpose(-1, -2)) * self.scale
        length = scores.shape[-1]
        future = torch.ones(length, length, dtype=torch.bool).triu(1)
        weights = scores.masked_fill(future, float("-inf")).softmax(dim=-1)
        return self.proj(weights @ self.v(h))


def _ops_named(trace, func_name: str):
    return [op for op in trace.ops if op.func_name == func_name]


def _capture(model: nn.Module, x: torch.Tensor, **options):
    torch.manual_seed(0)
    return tl.trace(model.eval(), x, capture=tl.options.CaptureOptions(**options))


@pytest.mark.parametrize("func_name", ["dropout", "contiguous", "type"])
def test_same_object_payload_is_a_gradient_source(func_name: str) -> None:
    """Gradients from the last linear reach each same-object op's saved payload."""

    trace = _capture(_PassThroughChain(), torch.randn(2, 4), **_ARMED)
    try:
        (op,) = _ops_named(trace, func_name)
        last = _ops_named(trace, "linear")[-1]
        assert op.is_inplace is False
        (grad,) = torch.autograd.grad(last.out.sum(), op.out, allow_unused=True)
        assert grad is not None, f"{op.label} payload is not in the autograd graph"
        assert grad.shape == op.out.shape
    finally:
        trace.cleanup()


def test_same_object_payload_is_the_parent_payload_object() -> None:
    """The pass-through chain shares one live object with the linear that produced it."""

    trace = _capture(_PassThroughChain(), torch.randn(2, 4), **_ARMED)
    try:
        first = _ops_named(trace, "linear")[0]
        chain = [_ops_named(trace, name)[0] for name in ("dropout", "contiguous", "type")]
        assert all(op.out is first.out for op in chain)
        assert [op.parents for op in chain] == [
            [first.layer_label],
            [chain[0].layer_label],
            [chain[1].layer_label],
        ]
    finally:
        trace.cleanup()


def test_copy_mode_keeps_distinct_same_object_payloads() -> None:
    """The default save mode still stores an independent copy per op."""

    trace = _capture(_PassThroughChain(), torch.randn(2, 4))
    first = _ops_named(trace, "linear")[0]
    (drop,) = _ops_named(trace, "dropout")
    assert drop.out is not first.out
    assert torch.equal(drop.out, first.out)


def test_mutating_same_object_return_keeps_its_copy_under_reference_mode() -> None:
    """An in-place ``add_`` payload is its post-write value, not the live receiver."""

    trace = _capture(_InPlaceAdd(), torch.randn(2, 4), **_ARMED)
    try:
        (add,) = _ops_named(trace, "add_")
        (tanh,) = _ops_named(trace, "tanh")
        assert add.is_inplace is True
        assert add.out is not tanh.out
        assert torch.allclose(add.out, torch.tanh(trace.ops[tanh.parents[0]].out) + 1.0)
    finally:
        trace.cleanup()


def test_reference_capture_with_same_object_returns_validates() -> None:
    """Forward replay passes on the reference-mode capture of the pass-through chain."""

    torch.manual_seed(0)
    model = _PassThroughChain().eval()
    x = torch.randn(2, 4)
    with torch.no_grad():
        truth = [model(x)]
    trace = tl.trace(model, x, capture=tl.options.CaptureOptions(**_ARMED))
    try:
        status = trace.validate_forward_pass(truth)
        assert status is True, status
    finally:
        trace.cleanup()
    assert tl.validate(_PassThroughChain().eval(), x, scope="forward")
    assert tl.validate(_PassThroughChain().eval(), x, scope="backward")


def test_gradient_projective_field_seeded_at_an_eval_dropout() -> None:
    """A unit at position 3 of a dropout output reaches the attention output at 3 and 4.

    The combined README example seeded a projective field at an attention block
    whose output is an eval-mode dropout; that seed used to fail with autograd
    returning no VJP.
    """

    torch.manual_seed(1)
    trace = tl.trace(
        _CausalAttentionAfterDropout().eval(),
        torch.randn(1, _SEQ, 4),
        capture=tl.options.CaptureOptions(**_ARMED),
    )
    try:
        (seed,) = _ops_named(trace, "dropout")
        module = trace.modules["proj"]
        (target_label,) = module.calls[0].output_ops
        units = [(0, _SEED_POSITION, j) for j in range(seed.out.shape[-1])]
        reached = seed.projective_field.of(units, within="proj", method="gradient")
        rows = [entry for entry in reached if f"{entry.key[0]}:{entry.key[1]}" == target_label]
        assert len(rows) == 1, [entry.key for entry in reached]
        mask = rows[0].mask
        positions = [pos for pos in range(mask.shape[1]) if bool(mask[:, pos].any())]
        assert positions == list(range(_SEED_POSITION, _SEQ))
    finally:
        trace.cleanup()
