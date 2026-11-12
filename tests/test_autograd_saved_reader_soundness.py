"""The per-class saved-attribute reader against the generic walk over every ``_saved_*`` name.

``_ops_autograd`` caches, per C-defined gradient-function class, the ``_saved_*`` names
that can hold a tensor, and reads only those. These tests compare the statistics it
produces (``autograd_memory``, ``num_autograd_tensors`` and the saved band) with the
generic walk (``_iter_autograd_saved_candidates``) on direct ops and on whole captures,
including a class first seen with a ``None`` field and later with a tensor in it, and a
custom ``torch.autograd.Function``, whose Python-defined node is never cached.
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import torchlens as tl
from torchlens.backends.torch import _ops_autograd, _ops_exhaustive

_ENTRY = "_output_entries_with_autograd_stats"


class _SaveBoth(torch.autograd.Function):
    """Custom autograd function saving its input and output for backward."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        """Save the input and the output; return the doubled input."""
        out = x * 2
        ctx.save_for_backward(x, out)
        return out

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> torch.Tensor:
        """Return the gradient scaled by the saved input."""
        x, _out = ctx.saved_tensors
        return grad * x


class _Mixed(nn.Module):
    """Ops whose nodes save tensors beside sizes, strides, scalars, flags and strings."""

    def __init__(self) -> None:
        """Build the layers."""
        super().__init__()
        self.emb = nn.Embedding(11, 8)
        self.ln_plain = nn.LayerNorm(8, elementwise_affine=False)
        self.ln = nn.LayerNorm(8)
        self.lin = nn.Linear(8, 8)
        self.lin_nobias = nn.Linear(8, 8, bias=False)
        self.conv = nn.Conv1d(4, 4, 3, padding=1, bias=False)
        self.conv_b = nn.Conv1d(4, 4, 3, padding=1)
        self.bn = nn.BatchNorm1d(4)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Run the mixed ops."""
        h = self.ln_plain(self.emb(idx))
        h = self.ln(h)
        h = F.gelu(self.lin(h), approximate="tanh")
        h = _SaveBoth.apply(self.lin_nobias(h))
        q, k, v = h.chunk(3, dim=0) if h.shape[0] % 3 == 0 else (h, h, h)
        h = F.scaled_dot_product_attention(q, k, v) + q.pow(2).clamp(min=0.1)
        c = h.reshape(-1, 4, 2)
        c = self.bn(self.conv_b(self.conv(c)))
        c = torch.relu_(c * 0.5)
        parts = torch.split(c, 1, dim=1)
        return torch.cat(parts, dim=1).sum(dim=-1).softmax(dim=-1)


def _reference_entry(trace: Any, output: Any) -> Any:
    """The partition with the generic walk, as the statistics were computed before the cache."""
    return _ops_autograd._partition_output_entries_with_autograd_stats(trace, output)


def _capture(model: nn.Module, x: torch.Tensor) -> Any:
    torch.manual_seed(0)
    return tl.trace(model, x, capture=tl.options.CaptureOptions(seed=0), save="all")


def _stats(trace: Any) -> tuple[list[tuple[str, Any, Any]], dict[Any, Any]]:
    rows = [
        (layer.label, layer.cost.autograd_memory, layer.autograd.num_autograd_tensors)
        for layer in trace._layer_list
    ]
    return rows, dict(trace.__dict__.get("_autograd_saved_bands", {}))


def _reference_capture(
    monkeypatch: pytest.MonkeyPatch, model: nn.Module, x: torch.Tensor
) -> tuple[list[tuple[str, Any, Any]], dict[Any, Any]]:
    with monkeypatch.context() as patch:
        patch.setattr(
            _ops_autograd, "_saved_values", _ops_exhaustive._iter_autograd_saved_candidates
        )
        for holder in (_ops_autograd, _ops_exhaustive):
            patch.setattr(holder, _ENTRY, _reference_entry)
        return _stats(_capture(model, x))


def test_capture_statistics_match_the_generic_walk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every op's saved bytes, saved count and band equal the generic walk's, cold and warm."""
    torch.manual_seed(0)
    model = _Mixed()
    x = torch.randint(0, 11, (3, 4))
    reference = _reference_capture(monkeypatch, model, x)
    assert any(count for _label, _bytes, count in reference[0])
    monkeypatch.setattr(_ops_autograd, "_SAVED_READER_BY_CLASS", {})
    cold = _stats(_capture(model, x))
    warm = _stats(_capture(model, x))
    assert cold == reference
    assert warm == reference


def _direct_ops() -> list[Callable[[], Any]]:
    w = torch.randn(4, 3, requires_grad=True)
    x = torch.randn(2, 3, requires_grad=True)
    img = torch.randn(1, 2, 5, 5, requires_grad=True)
    kern = torch.randn(3, 2, 3, 3, requires_grad=True)
    bias = torch.randn(3, requires_grad=True)
    bias4 = torch.randn(4, requires_grad=True)
    gamma = torch.randn(3, requires_grad=True)
    return [
        lambda: F.layer_norm(x, (3,)),
        lambda: F.layer_norm(x, (3,), gamma, gamma),
        lambda: F.conv2d(img, kern),
        lambda: F.conv2d(img, kern, bias, stride=2, padding=1),
        lambda: F.linear(x, w),
        lambda: F.linear(x, w, bias4),
        lambda: F.gelu(x, approximate="tanh"),
        lambda: x.clamp(min=-0.5),
        lambda: x.pow(3),
        lambda: x[torch.tensor([1, 0])],
        lambda: torch.stack([x, x * 2]),
        lambda: F.max_pool2d(img, 2),
        lambda: F.interpolate(img, scale_factor=2.0, mode="bilinear"),
        lambda: torch.var_mean(x, dim=0),
        lambda: F.cross_entropy(x, torch.tensor([0, 2])),
        lambda: _SaveBoth.apply(x),
    ]


def test_direct_ops_match_the_generic_walk_in_learning_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A class learned on one call (a ``None`` bias or weight) reads a later call's tensor."""
    monkeypatch.setattr(_ops_autograd, "_SAVED_READER_BY_CLASS", {})
    trace = SimpleNamespace()
    for make in _direct_ops():
        out = make()
        first = out if isinstance(out, torch.Tensor) else out[0]
        grad_fn = first.grad_fn
        expected = _ops_autograd._get_autograd_saved_stats_by_output(first)[0]
        for _ in range(2):
            got, band = _ops_autograd._saved_tensor_stats(trace, grad_fn, set(), set())
            assert got == expected, type(grad_fn).__name__
            assert (
                sum(band[k] for k in ("saved_parameter", "saved_buffer", "saved_activation"))
                == (expected[0])
            )


def test_custom_function_node_takes_the_walk_and_is_never_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A custom Function's backward node is read through ``saved_tensors`` on every op."""
    cache: dict[type, Any] = {}
    monkeypatch.setattr(_ops_autograd, "_SAVED_READER_BY_CLASS", cache)
    x = torch.randn(2, 3, requires_grad=True)
    out = _SaveBoth.apply(x)
    stats, _band = _ops_autograd._saved_tensor_stats(SimpleNamespace(), out.grad_fn, set(), set())
    assert stats == (2 * x.numel() * x.element_size(), 2)
    assert type(out.grad_fn) not in cache
    assert all(not cls.__flags__ & _ops_autograd._HEAPTYPE_FLAG for cls in cache)


def test_cached_reader_keeps_only_names_that_can_hold_a_tensor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``AddmmBackward0`` keeps its two matrices and drops its sizes, strides and scalars."""
    cache: dict[type, Any] = {}
    monkeypatch.setattr(_ops_autograd, "_SAVED_READER_BY_CLASS", cache)
    x = torch.randn(2, 3, requires_grad=True)
    lin = nn.Linear(3, 4)
    out = lin(x)
    _ops_autograd._saved_tensor_stats(SimpleNamespace(), out.grad_fn, set(), set())
    names = cache[type(out.grad_fn)][2]
    assert set(names) == {"_saved_mat1", "_saved_mat2"}
    assert _ops_autograd._never_holds_tensor((1, 2))
    assert _ops_autograd._never_holds_tensor("tanh")
    assert not _ops_autograd._never_holds_tensor(())
    assert not _ops_autograd._never_holds_tensor(None)
    assert not _ops_autograd._never_holds_tensor((x, None))
