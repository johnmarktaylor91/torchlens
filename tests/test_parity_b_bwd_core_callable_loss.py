"""Trace.backward accepts a callable loss with loss_fn's meaning."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from tests.test_parity_b_bwd_core_dispatch import _native_module, _patched
from torchlens.backends._backward_native import NativeBackwardSurface


class _TwoHeads(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Linear(4, 3)
        self.left = nn.Linear(3, 1)
        self.right = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = torch.tanh(self.body(x))
        return self.left(hidden), self.right(hidden)


def _trace() -> tl.Trace:
    torch.manual_seed(3)
    return tl.trace(
        _TwoHeads(),
        torch.randn(2, 4, requires_grad=True),
        capture=tl.options.CaptureOptions(save_grads="all"),
        save="all",
    )


def _grads(trace: tl.Trace) -> dict[str, torch.Tensor]:
    return {op.layer_label: op.grad for op in trace._layer_list if op.has_grad}


def _loss_fn(output: Any) -> torch.Tensor:
    left, right = output
    return left.sum() + (right**2).sum()


def test_callable_loss_matches_the_tensor_loss_on_torch() -> None:
    by_tensor = _trace()
    by_tensor.backward(_loss_fn(by_tensor.output))
    by_callable = _trace()
    seen: list[Any] = []

    def loss(output: Any) -> torch.Tensor:
        seen.append(output)
        return _loss_fn(output)

    assert by_callable.backward(loss) is by_callable
    assert len(seen) == 1 and isinstance(seen[0], tuple) and len(seen[0]) == 2
    assert len(by_callable.autograd.backward_passes) == 1
    want, got = _grads(by_tensor), _grads(by_callable)
    assert want and want.keys() == got.keys()
    for label in want:
        assert torch.equal(want[label], got[label]), label


def test_callable_loss_forwards_backward_keywords() -> None:
    trace = _trace()
    trace.backward(_loss_fn, retain_graph=True)
    trace.backward(_loss_fn, retain_graph=True)
    assert len(trace.autograd.backward_passes) == 2


def test_native_backend_receives_the_callable_itself(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _native_module([], {})
    _patched(monkeypatch, "jax", NativeBackwardSurface("jax", module))
    trace = _trace()
    trace._provenance_backend = "jax"
    trace.backward(_loss_fn)
    assert module.calls == [(_loss_fn, {})]
