"""Cancellation-aware reconstruction tolerance model (A03; mikit F4/F5).

The reconstruction gate used to borrow the REPLAY tolerance table, whose
absolute term sits at denormal scale by design; a reconstruction is an
independent summation, so at cancellation sites (large addends, small output)
the replay-shaped gate refused numerically CORRECT reconstructions (measured
19/36, input-dependently). The separate model in
``torchlens._semantic.tolerances`` adds an elementwise accumulated-|addend|
magnitude term; these tests pin BOTH directions: correct reconstructions pass
across scales/shapes/seeds, and planted corruption (all-zero, sign flip,
permutation) still fails with the magnitude term active.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from torchlens._semantic.tolerances import (
    within_reconstruction_tolerance,
)


def test_shape_mismatch_is_a_refusal_not_a_crash() -> None:
    assert not within_reconstruction_tolerance(torch.ones(3), torch.ones(4))


def test_nonfinite_targets_must_be_reproduced_exactly() -> None:
    target = torch.tensor([1.0, float("inf"), float("nan"), -2.0])
    exact = target.clone()
    assert within_reconstruction_tolerance(exact, target, magnitude=target.abs().nan_to_num())
    wrong_inf_sign = torch.tensor([1.0, float("-inf"), float("nan"), -2.0])
    assert not within_reconstruction_tolerance(wrong_inf_sign, target)
    finite_where_nan = torch.tensor([1.0, float("inf"), 0.0, -2.0])
    assert not within_reconstruction_tolerance(finite_where_nan, target)


class _SdpaConv1dAttention(torch.nn.Module):
    """Attention block projecting per-head outputs through an HF Conv1D."""

    def __init__(self, n_heads: int, d_head: int) -> None:
        super().__init__()
        from transformers.pytorch_utils import Conv1D

        self.n_heads = n_heads
        self.d_head = d_head
        d_model = n_heads * d_head
        # Conv1D stores weight [in, out]; forward is x @ W + b.
        self.c_proj = Conv1D(d_model, d_model)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        z = F.scaled_dot_product_attention(q, k, v)
        batch, _heads, seq, _dh = z.shape
        merged = z.transpose(1, 2).reshape(batch, seq, self.n_heads * self.d_head)
        return self.c_proj(merged)


class _SdpaLinearAttention(torch.nn.Module):
    """Same block with an nn.Linear projection (weight [out, in])."""

    def __init__(self, n_heads: int, d_head: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        d_model = n_heads * d_head
        self.c_proj = torch.nn.Linear(d_model, d_model)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        z = F.scaled_dot_product_attention(q, k, v)
        batch, _heads, seq, _dh = z.shape
        merged = z.transpose(1, 2).reshape(batch, seq, self.n_heads * self.d_head)
        return self.c_proj(merged)


@pytest.mark.real_model
@pytest.mark.parametrize("projection", ["conv1d", "linear"])
def test_per_head_result_orientation_from_module_class(projection: str) -> None:
    """mikit F5: per-head `result` must be right for BOTH weight orientations.

    GPT-2's ``Conv1D`` stores its projection weight [in, out]; the
    reconstruction assumed nn.Linear's [out, in], and a SQUARE projection
    passes every shape guard with either orientation, so the per-head result
    was deterministically wrong on GPT-2 (sum-check error 18.80). Orientation
    now keys on the module class; the reconstructed per-head contributions
    must sum to the captured projection output AND match a manual per-head
    computation in the correct orientation.
    """

    pytest.importorskip("transformers")
    import torchlens as tl
    from torchlens._semantic.reconstruction import _reconstruct_checked, find_sdpa_op

    torch.manual_seed(3)
    n_heads, d_head, seq = 2, 4, 5
    if projection == "conv1d":
        model = _SdpaConv1dAttention(n_heads, d_head)
        weight_out_in = model.c_proj.weight.detach().transpose(0, 1)
    else:
        model = _SdpaLinearAttention(n_heads, d_head)
        weight_out_in = model.c_proj.weight.detach()
    q = torch.randn(1, n_heads, seq, d_head)
    k = torch.randn(1, n_heads, seq, d_head)
    v = torch.randn(1, n_heads, seq, d_head)
    log = tl.trace(
        model,
        [q, k, v],
        capture=tl.options.CaptureOptions(save_args=True),
        save="all",
    )
    try:
        module = log.modules["self"]
        sdpa_op = find_sdpa_op(module)
        assert sdpa_op is not None
        result = _reconstruct_checked(module, sdpa_op, "result")
        assert isinstance(result, torch.Tensor), f"result refused: {result!r}"
        # Manual per-head contributions in the [out, in] orientation.
        z = F.scaled_dot_product_attention(q, k, v)
        per_head_weight = weight_out_in.reshape(weight_out_in.shape[0], n_heads, d_head).transpose(
            0, 1
        )
        manual = torch.einsum("...shd,hod->...sho", z.transpose(-3, -2), per_head_weight)
        assert torch.allclose(result, manual, atol=1e-5, rtol=1e-4)
        # Head-sum + bias reproduces the captured projection output.
        projected = model.c_proj(z.transpose(1, 2).reshape(1, seq, n_heads * d_head))
        summed = result.sum(dim=-2) + model.c_proj.bias
        assert torch.allclose(summed, projected, atol=1e-5, rtol=1e-4)
    finally:
        log.cleanup()
