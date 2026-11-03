"""Fused attention modules: the captured output equals eager's on the same program.

Capture holds torch's public fused-attention switch off (``torch.backends.mha``)
and never forces ``requires_grad`` on frozen weights, so a captured
``nn.MultiheadAttention`` or ``nn.TransformerEncoderLayer`` always records the
unfused attention internals, and the validation ground truth runs under the same
hold. ``tests/test_fused_attention_fastpath_capture.py`` pins that contract on a
frozen ``nn.MultiheadAttention``. This file covers the neighbouring rows of the
stress cluster that motivated it: the fused encoder layer, and the cases where
eager itself takes the unfused path (trainable weights, a grad-requiring input,
training mode), where the captured output must equal plain eager bit for bit.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._utils._torch_compat import (
    get_mha_fastpath_enabled,
    get_mha_fastpath_switch_support,
    set_mha_fastpath_enabled,
)
from torchlens.validation import last_validation_failure

WIDTH = 8
FUSED_OPS = frozenset({"_native_multi_head_attention", "_transformer_encoder_layer_fwd"})


class FrozenAttention(nn.Module):
    """``nn.MultiheadAttention`` eligible for its inference fast path (eval, frozen, batch-first)."""

    def __init__(self) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(WIDTH, 2, batch_first=True)
        self.attention.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.attention(x, x, x, need_weights=False)
        return out


class FrozenEncoderLayer(nn.Module):
    """``nn.TransformerEncoderLayer`` eligible for its fused fast path."""

    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(WIDTH, 2, dim_feedforward=16, batch_first=True)
        self.layer.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


def _needs_switch() -> None:
    if not get_mha_fastpath_switch_support():
        pytest.skip("torch build without the public torch.backends.mha fast-path switch")


def _build(cls: type[nn.Module]) -> tuple[nn.Module, torch.Tensor]:
    torch.manual_seed(0)
    return cls().eval(), torch.randn(3, 5, WIDTH)


def _eager_unfused(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Return eager's output with the public fast-path switch held off."""

    prior = get_mha_fastpath_enabled()
    set_mha_fastpath_enabled(False)
    try:
        return model(x.clone()).detach().clone()
    finally:
        if prior is not None:
            set_mha_fastpath_enabled(prior)


def _captured(trace: tl.Trace) -> torch.Tensor:
    return trace[trace.output_layers[0]].out


def _op_names(trace: tl.Trace) -> list[str]:
    return [op.func_name for op in trace.ops]


def test_frozen_encoder_layer_records_the_unfused_program_eager_runs_unfused() -> None:
    """Grad on, every tensor frozen: capture equals eager's unfused program bit for bit."""

    _needs_switch()
    model, x = _build(FrozenEncoderLayer)
    assert torch.is_grad_enabled()
    expected = _eager_unfused(model, x)
    trace = tl.trace(model, [x.clone()])
    assert not FUSED_OPS.intersection(_op_names(trace)), _op_names(trace)
    assert torch.equal(_captured(trace), expected)
    assert not any(p.requires_grad for p in model.parameters())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_frozen_encoder_layer_forward_validation_passes_with_grad_on(dtype: torch.dtype) -> None:
    """The stress repro on the encoder layer: ground truth and capture run one program."""

    _needs_switch()
    model, x = _build(FrozenEncoderLayer)
    model, x = model.to(dtype), x.to(dtype)
    assert torch.is_grad_enabled()
    assert tl.validate(model, [x], scope="forward"), last_validation_failure()


def test_trainable_attention_matches_plain_eager() -> None:
    """Trainable weights fail eager's fast-path gate: both sides run unfused and agree."""

    model, x = _build(FrozenAttention)
    model.attention.requires_grad_(True)
    eager = model(x.clone()).detach().clone()
    trace = tl.trace(model, [x.clone()])
    assert not FUSED_OPS.intersection(_op_names(trace)), _op_names(trace)
    assert torch.equal(_captured(trace), eager)
    assert all(p.requires_grad for p in model.parameters())


def test_grad_requiring_input_matches_plain_eager() -> None:
    """A grad-requiring query fails eager's gate: both sides run unfused and agree."""

    model, x = _build(FrozenAttention)
    x = x.requires_grad_(True)
    eager = model(x).detach().clone()
    trace = tl.trace(model, [x])
    assert not FUSED_OPS.intersection(_op_names(trace)), _op_names(trace)
    assert torch.equal(_captured(trace), eager)
    assert not any(p.requires_grad for p in model.parameters())


def test_training_mode_attention_matches_plain_eager() -> None:
    """A training-mode module fails eager's gate before the flags matter."""

    model, x = _build(FrozenAttention)
    model.train()
    eager = model(x.clone()).detach().clone()
    trace = tl.trace(model, [x.clone()])
    assert not FUSED_OPS.intersection(_op_names(trace)), _op_names(trace)
    assert torch.equal(_captured(trace), eager)
