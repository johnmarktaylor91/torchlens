"""``__setitem__`` validation with no saved arguments returns a verdict, never a TypeError.

The pre-execution perturbation exemptions run before the missing-saved-args guard of the
per-op replay check, so ``_check_setitem_exempt`` sees
``saved_args is None`` whenever a trace was captured with ``save_arg_values=False``.
It must answer "not exempt" (the later guard then reports the op unverified), and it must
never exempt a case that would fail today.
"""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl
from torchlens.validation import exemptions


class _MaskedWrite(nn.Module):
    """``y[x > 0] = 0``: the bool mask parent sits at ``__setitem__`` slot 1."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = x > 0
        y = x * 2
        y[mask] = 0.0
        return y + 1


class _SliceWrite(nn.Module):
    """``y[:, :2] = v``: the destination parent sits at slot 0, the value at slot 2."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * 2
        y[:, :2] = x[:, 2:4] + 1
        return y + 1


def _setitem_op(trace: tl.Trace):
    ops = [op for op in trace.layer_list if getattr(op, "func_name", None) == "__setitem__"]
    assert len(ops) == 1
    return ops[0]


def _parent_at(op, slot: int) -> str:
    return str(op.parent_arg_positions["args"][slot])


def test_setitem_exempt_with_no_saved_args_answers_not_exempt() -> None:
    """Both slot-1 (mask) and slot-0 (destination) parents: False, no TypeError."""

    torch.manual_seed(0)
    for model, slot in ((_MaskedWrite(), 1), (_SliceWrite(), 0)):
        x = torch.randn(3, 5)
        trace = tl.trace(model, x, capture=tl.options.CaptureOptions(save_args=False))
        op = _setitem_op(trace)
        assert op.saved_args is None
        assert exemptions._check_setitem_exempt(trace, op, [_parent_at(op, slot)]) is False


def test_setitem_validation_with_no_saved_args_returns_a_verdict() -> None:
    """Public capture plus validation answers; the argument-logging stage stops replay first."""

    torch.manual_seed(0)
    model = _MaskedWrite()
    x = torch.randn(3, 5)
    trace = tl.trace(model, x, capture=tl.options.CaptureOptions(save_args=False))
    result = trace.validate_forward_pass(model(x))
    assert result is not None


def test_setitem_exemption_unchanged_when_args_are_saved() -> None:
    """With saved args the mask parent stays exempt and a partial slice destination does not."""

    torch.manual_seed(0)
    x = torch.randn(3, 5)
    masked = tl.trace(_MaskedWrite(), x, capture=tl.options.CaptureOptions(save_args=True))
    op = _setitem_op(masked)
    assert exemptions._check_setitem_exempt(masked, op, [_parent_at(op, 1)]) is True
    sliced = tl.trace(_SliceWrite(), x, capture=tl.options.CaptureOptions(save_args=True))
    op = _setitem_op(sliced)
    assert exemptions._check_setitem_exempt(sliced, op, [_parent_at(op, 0)]) is False
