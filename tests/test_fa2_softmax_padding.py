"""FlashAttention 2's undefined ``S_dmask`` padding is zeroed at the recorded call.

The real kernel (``flash_attn::_flash_attn_forward``) returns its dropout mask shaped
``(batch, heads, round128(seqlen_q), round128(seqlen_k))`` and defines only
``[..., :seqlen_q, :seqlen_k]``; the padding holds whatever memory it was given, so two
calls on the same inputs under the same generator state differ there. The recorder zeroes
that padding where the call returns, in the capture and in the replay callable, so forward
replay compares the whole tensor bit for bit. A CPU stand-in with the real schema plants a
different nonzero padding value on every call to model the undefined cells. The GPU proof
on the real kernel is ``tests/real_model/test_compat_matrix_rows.py::
test_flash_attention_2_dropout_replays_exactly``.
"""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Iterator
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.backends.torch import _torch_ops_calls
from torchlens.backends.torch._foreign_data import FLASH_ATTENTION_2_FUNC_NAME
from torchlens.options import CaptureOptions
from torchlens.validation import last_validation_failure, validate_forward_pass
from torchlens.validation.core import _check_whether_func_on_saved_parents_yields_saved_tensor

_SCHEMA = (
    "_flash_attn_forward(Tensor q, Tensor k, Tensor v, float dropout_p, float softmax_scale, "
    "bool causal, int window_size_left, int window_size_right, float softcap, "
    "Tensor? alibi_slopes, bool return_softmax) -> (Tensor, Tensor, Tensor, Tensor)"
)
_SEQ = 5
_ROUNDED = 128
_PAD_VALUES = itertools.count(7)
"""Every stand-in call plants the next value in its padding: the undefined cells differ."""
_SEEN: list[torch.Tensor] = []
"""The ``S_dmask`` each forward received, to compare with what the trace saved."""


def _padded_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    alibi_slopes: torch.Tensor | None,
    return_softmax: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The real output layout in unpatchable C calls, with a fresh padding value per call."""

    del dropout_p, softmax_scale, causal, window_size_left, window_size_right, softcap
    del alibi_slopes
    vf: Any = torch._C._VariableFunctions
    out = vf.add(q, vf.mul(k, v))
    lse = vf.logsumexp(vf.add(q, k), [3])  # reads q and k, as the declared row says
    if return_softmax:
        scores = vf.matmul(vf.transpose(q, 1, 2), vf.transpose(vf.transpose(k, 1, 2), 2, 3))
        pad_q, pad_k = _ROUNDED - q.shape[1], _ROUNDED - k.shape[1]
        mask = vf.constant_pad_nd(scores, [0, pad_k, 0, pad_q], float(next(_PAD_VALUES)))
    else:
        mask = vf.empty((0,), dtype=q.dtype)
    return out, lse, mask, vf.full((2,), 1234, dtype=torch.int64)


def _scoped_standin(namespace: str) -> Iterator[Any]:
    """Yield ``torch.ops.<namespace>._flash_attn_forward`` backed by the padded stand-in."""

    with torch.library._scoped_library(namespace, "FRAGMENT") as lib:
        if not hasattr(getattr(torch.ops, namespace), "_flash_attn_forward"):
            lib.define(_SCHEMA)
        torch.library.register_kernel(
            f"{namespace}::_flash_attn_forward", "cpu", _padded_kernel, lib=lib
        )
        yield getattr(torch.ops, namespace)._flash_attn_forward


@pytest.fixture
def fa2_op() -> Iterator[Any]:
    """The stand-in registered as the declared ``flash_attn::_flash_attn_forward``."""

    yield from _scoped_standin("flash_attn")


@pytest.fixture
def undeclared_op() -> Iterator[Any]:
    """The same operator, schema and output shapes under an undeclared namespace."""

    yield from _scoped_standin("tltest_fa2pad")


class _Attention(nn.Module):
    """One projection into q, k, v of shape (1, seq, 2 heads, 4) and the four-output op."""

    def __init__(self, op: Any) -> None:
        super().__init__()
        self.op = op
        self.proj = nn.Linear(8, 24)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.proj(x).view(1, _SEQ, 3, 2, 4).unbind(2)
        out, _lse, s_dmask, _rng = self.op(q, k, v, 0.25, 1.0, False, -1, -1, 0.0, None, True)
        _SEEN.append(s_dmask)
        return torch.relu(out)


def _inputs() -> torch.Tensor:
    return torch.randn(1, _SEQ, 8, generator=torch.Generator().manual_seed(3))


def _trace(model: nn.Module) -> Any:
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return tl.trace(model, _inputs(), capture=CaptureOptions(save_args=True), save="all")


def _validates(model: nn.Module) -> bool:
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return bool(validate_forward_pass(model, [_inputs()], input_kwargs={}))


def _s_dmask_op(trace: Any) -> Any:
    ops = [op for op in trace._layer_list if op.func_name == FLASH_ATTENTION_2_FUNC_NAME]
    (op,) = [op for op in ops if op.multi_output_index == 2]
    return op


def _replay_decision(trace: Any, op: Any) -> str:
    return _check_whether_func_on_saved_parents_yields_saved_tensor(trace, op.label).decision


def test_declared_fa2_padding_is_zeroed_and_replays_exactly(fa2_op: Any) -> None:
    """The model, the saved value and the replay hold the same bytes; validation passes."""

    model = _Attention(fa2_op)
    trace = _trace(model)
    try:
        saved = _s_dmask_op(trace).out
        assert saved.shape == (1, 2, _ROUNDED, _ROUNDED)
        assert torch.equal(saved, _SEEN[-1])
        assert not saved[..., _SEQ:, :].any() and not saved[..., :_SEQ, _SEQ:].any()
        assert saved[..., :_SEQ, :_SEQ].any()
        assert _replay_decision(trace, _s_dmask_op(trace)) == "validated"
    finally:
        trace.cleanup()
    assert _validates(model) is True, last_validation_failure()


def test_undeclared_custom_op_padding_is_left_alone(undeclared_op: Any) -> None:
    """The same op and output shape under another namespace keeps its planted padding."""

    trace = _trace(_Attention(undeclared_op))
    try:
        op = _s_dmask_op(trace)
        assert op.out.shape == (1, 2, _ROUNDED, _ROUNDED)
        assert (op.out[..., _SEQ:, :] >= 7).all() and (op.out[..., :_SEQ, _SEQ:] >= 7).all()
        assert _replay_decision(trace, op) == "failed"
    finally:
        trace.cleanup()


def test_removing_the_zeroing_brings_the_replay_mismatch_back(
    fa2_op: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutation proof: without the zeroing the declared op's dropout mask fails replay."""

    monkeypatch.setattr(_torch_ops_calls, "_zero_fa2_softmax_padding", lambda *_args: None)
    model = _Attention(fa2_op)
    assert _validates(model) is False
    failure = last_validation_failure()
    assert failure is not None and failure.check_name == "forward_replay", failure
    trace = _trace(model)
    try:
        assert failure.op_label == _s_dmask_op(trace).label, failure
    finally:
        trace.cleanup()


def test_a_corrupted_cell_still_fails_replay(fa2_op: Any) -> None:
    """A changed cell inside the defined region, or in the zeroed padding, fails replay."""

    for cell in ((0, 0, 1, 2), (0, 1, _SEQ + 3, 0)):
        trace = _trace(_Attention(fa2_op))
        try:
            op = _s_dmask_op(trace)
            assert _replay_decision(trace, op) == "validated"
            op.out[cell] += 1
            assert _replay_decision(trace, op) == "failed", cell
        finally:
            trace.cleanup()
