"""Capture takes the fused attention fast path exactly when eager does.

Capture forces ``requires_grad=True`` on prepared Parameters; the fused fast
paths of ``nn.MultiheadAttention`` and ``nn.TransformerEncoderLayer`` are gated
on that flag, so a frozen eval module used to run decomposed under capture and
its captured output differed from eager at rounding level (the forward
validation ground-truth check caught it). ``backends/torch/fastpath_flags.py``
restores the user's flags for that module call.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._utils._torch_compat import get_mha_fastpath_enabled, set_mha_fastpath_enabled
from torchlens.options import CaptureOptions

WIDTH = 8
FUSED_MHA_OP = "_native_multi_head_attention"
FUSED_LAYER_OP = "_transformer_encoder_layer_fwd"


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


def _eager_outputs(model: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return eager outputs with the public fast-path switch on and off."""

    prior = get_mha_fastpath_enabled()
    try:
        set_mha_fastpath_enabled(True)
        fast = model(x.clone()).detach().clone()
        set_mha_fastpath_enabled(False)
        slow = model(x.clone()).detach().clone()
    finally:
        if prior is not None:
            set_mha_fastpath_enabled(prior)
    return fast, slow


def _build(cls: type[nn.Module]) -> tuple[nn.Module, torch.Tensor]:
    torch.manual_seed(0)
    model = cls().eval()
    x = torch.randn(3, 5, WIDTH)
    return model, x


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("cls", "fused_op"), [(FrozenAttention, FUSED_MHA_OP), (FrozenEncoderLayer, FUSED_LAYER_OP)]
)
def test_frozen_eval_capture_takes_the_fused_path_eager_takes(
    cls: type[nn.Module], fused_op: str
) -> None:
    """Grad on, every tensor frozen: eager fuses, so capture fuses and matches bit for bit."""

    model, x = _build(cls)
    eager_fast, eager_slow = _eager_outputs(model, x)
    assert torch.is_grad_enabled()
    trace = tl.trace(model, x.clone(), capture=CaptureOptions(seed=0))
    func_names = [op.func_name for op in trace.ops]
    assert fused_op in func_names, func_names
    captured = trace[trace.output_layers[0]].out
    assert torch.equal(captured, eager_fast)
    if not torch.equal(eager_fast, eager_slow):
        assert not torch.equal(captured, eager_slow)
    # The user's own flags are back after the capture.
    assert not any(p.requires_grad for p in model.parameters())


@pytest.mark.smoke
def test_frozen_eval_forward_validation_passes_with_grad_on() -> None:
    """The S6 stress repro in small: validate's ground truth and capture take the same path."""

    model, x = _build(FrozenAttention)
    assert torch.is_grad_enabled()
    assert tl.validate(model, x, scope="forward", seed=0)


def test_trainable_attention_keeps_the_decomposed_path() -> None:
    """A trainable module decomposes in eager, so capture decomposes too and still matches."""

    model, x = _build(FrozenAttention)
    model.attention.requires_grad_(True)
    eager = model(x.clone()).detach().clone()
    trace = tl.trace(model, x.clone(), capture=CaptureOptions(seed=0))
    func_names = [op.func_name for op in trace.ops]
    assert FUSED_MHA_OP not in func_names, func_names
    assert torch.equal(trace[trace.output_layers[0]].out, eager)
    assert all(p.requires_grad for p in model.parameters())


def test_grad_requiring_input_keeps_the_decomposed_path() -> None:
    """A grad-requiring query fails eager's gate: both sides decompose."""

    model, x = _build(FrozenAttention)
    x = x.requires_grad_(True)
    eager = model(x).detach().clone()
    trace = tl.trace(model, x, capture=CaptureOptions(seed=0))
    func_names = [op.func_name for op in trace.ops]
    assert FUSED_MHA_OP not in func_names, func_names
    assert torch.equal(trace[trace.output_layers[0]].out, eager)


def test_no_grad_capture_is_unchanged() -> None:
    """Under no_grad the gate never consults the flags; the fused path was already taken."""

    model, x = _build(FrozenAttention)
    with torch.no_grad():
        eager = model(x.clone()).detach().clone()
        trace = tl.trace(model, x.clone(), capture=CaptureOptions(seed=0))
    assert [op.func_name for op in trace.ops] == [FUSED_MHA_OP]
    assert torch.equal(trace[trace.output_layers[0]].out, eager)


def test_training_mode_module_is_left_alone() -> None:
    """A training-mode module fails eager's gate before the flags matter."""

    model, x = _build(FrozenAttention)
    model.train()
    eager = model(x.clone()).detach().clone()
    trace = tl.trace(model, x.clone(), capture=CaptureOptions(seed=0))
    assert FUSED_MHA_OP not in [op.func_name for op in trace.ops]
    assert torch.equal(trace[trace.output_layers[0]].out, eager)
