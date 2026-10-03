"""AUD-CODE 2.12 (lane W051-SEMANTIC): half-precision norm reconstruction.

The shared norm linearization (``torchlens.semantic._norm_reconstruction``)
recomputed a norm in the PAYLOAD dtype -- five bf16 roundings -- while the
reconstruction tolerance model (``torchlens.semantic.tolerances``,
``RECONSTRUCTION_ULP_HEADROOM[bf16] = 4``) budgets an fp32-accumulate,
round-once recompute. Every bf16/fp16 norm therefore refused
``norm_convention_unmatched`` with a wrong diagnosis (a precision artifact
reported as a convention mismatch). The fix computes in the accumulation
dtype and rounds once for the comparison; the ULP headroom table is the
contract and is NOT widened. These tests pin:

* bf16 and fp16 LayerNorm on ``3 * randn(2, 5, 768)`` reconstruct
  (``kind == "layernorm_affine"``) and the fp32-arithmetic recompute sits
  inside ``within_reconstruction_tolerance``;
* the payload-dtype recompute (the defect's arithmetic) does NOT -- the
  diagnosis, pinned so the model is never "fixed" by widening headroom;
* the frozen ``scale``/``mean`` are handed out in fp32 and the receipt
  discloses payload/accumulation dtypes;
* RMSNorm and affine-free forms reconstruct in half precision too;
* a GENUINE convention mismatch on a bf16 model (Gemma-style ``1 + weight``)
  still refuses, and the remedy names the accumulate/round-once comparison
  so the user is not sent hunting for a precision bug;
* fp32 behaviour is unchanged (same kind, fp32 scale, sub-ULP residual).

The gpt2-bf16 ``direct_logit_contributions`` end-to-end row is deliberately
NOT here: DLA's own identity gate became dtype-aware on lane W051-MECH
(``_identity_eps`` budgets at the coarsest floating dtype among logits/norm
input), which is not on this lane's base. That row belongs to the merged
tree, beside W051-MECH's tests.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from torchlens.semantic._norm_reconstruction import (
    NormReconstructionError,
    reconstruct_norm,
)
from torchlens.semantic.tolerances import within_reconstruction_tolerance
from torchlens.utils._torch_compat import HAS_RMSNORM_MODULE, get_cpu_half_kernels_support

pytestmark = pytest.mark.smoke

_requires_cpu_half_kernels = pytest.mark.skipif(
    not get_cpu_half_kernels_support(),
    reason="CPU addmm/layer_norm for float16 postdates the torch 2.1 floor",
)

D_MODEL = 768
HALF_DTYPES = (torch.bfloat16, torch.float16)
# fp16 needs the CPU addmm/layer_norm kernels that postdate the torch 2.1
# floor; bf16 does not, so skip only the fp16 case rather than the whole
# parametrized test.
HALF_DTYPE_PARAMS = [
    pytest.param(torch.bfloat16, id="bf16"),
    pytest.param(torch.float16, id="fp16", marks=_requires_cpu_half_kernels),
]


def _norm_input(dtype: torch.dtype, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return (3.0 * torch.randn(2, 5, D_MODEL, generator=generator)).to(dtype)


def _affine(dtype: torch.dtype, seed: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    gamma = 1.0 + 0.1 * torch.randn(D_MODEL, generator=generator)
    beta = 0.1 * torch.randn(D_MODEL, generator=generator)
    return gamma.to(dtype), beta.to(dtype)


def _layer_norm(dtype: torch.dtype) -> tuple[torch.nn.LayerNorm, torch.Tensor, torch.Tensor]:
    module = torch.nn.LayerNorm(D_MODEL).to(dtype)
    gamma, beta = _affine(dtype)
    with torch.no_grad():
        module.weight.copy_(gamma)
        module.bias.copy_(beta)
    x = _norm_input(dtype)
    with torch.no_grad():
        y = module(x)
    return module, x, y


def _payload_dtype_recompute(
    x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """The DEFECT's arithmetic: every step rounded to the payload dtype."""

    mean = x.mean(dim=-1, keepdim=True)
    var = (x - mean).pow(2).mean(dim=-1, keepdim=True)
    scale = torch.sqrt(var + eps)
    recon = (x - mean) / scale * gamma + beta
    magnitude = ((x.abs() + mean.abs()) / scale) * gamma.abs() + beta.abs()
    return recon, magnitude


def _accumulate_recompute(
    x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """The kernel's arithmetic: fp32 accumulate, one rounding to storage."""

    acc = x.float()
    mean = acc.mean(dim=-1, keepdim=True)
    var = (acc - mean).pow(2).mean(dim=-1, keepdim=True)
    scale = torch.sqrt(var + eps)
    recon = (acc - mean) / scale * gamma.float() + beta.float()
    magnitude = ((acc.abs() + mean.abs()) / scale) * gamma.float().abs() + beta.float().abs()
    return recon.to(x.dtype), magnitude


@pytest.mark.parametrize("dtype", HALF_DTYPE_PARAMS)
def test_half_precision_layer_norm_reconstructs(dtype: torch.dtype) -> None:
    """bf16/fp16 LayerNorm reconstructs as layernorm_affine (was: refused)."""

    module, x, y = _layer_norm(dtype)
    record = reconstruct_norm(
        input=x,
        output=y,
        gamma=module.weight.detach(),
        beta=module.bias.detach(),
        eps=module.eps,
        module_address="ln_f",
        class_name="LayerNorm",
    )
    assert record.kind == "layernorm_affine"
    assert record.centered is True
    # The frozen operating point is handed out in the accumulation dtype.
    assert record.scale.dtype == torch.float32
    assert record.mean is not None and record.mean.dtype == torch.float32
    assert record.scale.shape == (2, 5, 1)
    # The evidenced payloads stay as evidenced.
    assert record.gamma is not None and record.gamma.dtype == dtype
    assert record.beta is not None and record.beta.dtype == dtype
    receipt = record.validation_receipt
    assert receipt["result"] == "validated"
    assert receipt["payload_dtype"] == str(dtype)
    assert receipt["accumulation_dtype"] == str(torch.float32)
    # Residual is a storage rounding at most: one ULP of the largest output.
    one_ulp = float(torch.finfo(dtype).eps) * float(y.float().abs().max())
    assert receipt["max_abs_residual"] <= one_ulp


@pytest.mark.parametrize("dtype", HALF_DTYPE_PARAMS)
def test_accumulate_recompute_is_within_tolerance_payload_recompute_is_not(
    dtype: torch.dtype,
) -> None:
    """Pin the DIAGNOSIS: the model fits round-once arithmetic, not per-step.

    If this test ever needs the headroom table widened to pass, the fix is
    wrong: the table is the contract, the arithmetic is what must match it.
    """

    module, x, y = _layer_norm(dtype)
    gamma, beta = module.weight.detach(), module.bias.detach()
    good, good_magnitude = _accumulate_recompute(x, gamma, beta, module.eps)
    assert within_reconstruction_tolerance(
        good, y, magnitude=good_magnitude, reduction_length=D_MODEL
    )
    bad, bad_magnitude = _payload_dtype_recompute(x, gamma, beta, module.eps)
    assert not within_reconstruction_tolerance(
        bad, y, magnitude=bad_magnitude, reduction_length=D_MODEL
    )


@pytest.mark.skipif(
    not HAS_RMSNORM_MODULE,
    reason="torch.nn.functional.rms_norm postdates the torch 2.1 floor (added 2.4)",
)
@pytest.mark.parametrize("dtype", HALF_DTYPES, ids=["bf16", "fp16"])
def test_half_precision_rms_norm_reconstructs(dtype: torch.dtype) -> None:
    """The uncentered affine form reconstructs in half precision as well."""

    eps = 1e-6
    x = _norm_input(dtype, seed=3)
    gamma, _ = _affine(dtype, seed=4)
    with torch.no_grad():
        y = F.rms_norm(x, (D_MODEL,), weight=gamma, eps=eps)
    record = reconstruct_norm(
        input=x, output=y, gamma=gamma, beta=None, eps=eps, module_address="norm"
    )
    assert record.kind == "rmsnorm_affine"
    assert record.centered is False
    assert record.mean is None
    assert record.scale.dtype == torch.float32


@pytest.mark.parametrize("dtype", HALF_DTYPE_PARAMS)
def test_half_precision_affine_free_layer_norm_reconstructs(dtype: torch.dtype) -> None:
    """normalize_only stays EVIDENCED (a passing affine-free check), never defaulted."""

    module = torch.nn.LayerNorm(D_MODEL, elementwise_affine=False).to(dtype)
    x = _norm_input(dtype, seed=5)
    with torch.no_grad():
        y = module(x)
    record = reconstruct_norm(
        input=x, output=y, gamma=None, beta=None, eps=module.eps, module_address="ln"
    )
    assert record.kind == "normalize_only"
    assert record.centered is True
    assert record.gamma is None and record.beta is None


def test_bf16_convention_mismatch_still_refuses_with_dtype_aware_remedy() -> None:
    """A GENUINE Gemma-style (1 + weight) mismatch on a bf16 model refuses.

    The refusal must teach the right cause: the recompute already ran in the
    accumulation dtype, so the user is told this is a convention mismatch and
    not a precision artifact of the half payload.
    """

    dtype = torch.bfloat16
    eps = 1e-5
    x = _norm_input(dtype, seed=6)
    generator = torch.Generator().manual_seed(7)
    weight = (0.3 * torch.randn(D_MODEL, generator=generator)).to(dtype)
    with torch.no_grad():
        normalized = F.layer_norm(x.float(), (D_MODEL,), eps=eps)
        y = (normalized * (1.0 + weight.float())).to(dtype)
    with pytest.raises(NormReconstructionError) as excinfo:
        reconstruct_norm(
            input=x,
            output=y,
            gamma=weight,
            beta=torch.zeros(D_MODEL, dtype=dtype),
            eps=eps,
            module_address="model.norm",
            class_name="GemmaRMSNorm",
        )
    fields = excinfo.value.fields
    assert fields["code"] == "norm_convention_unmatched"
    assert fields["payload_dtype"] == str(torch.bfloat16)
    assert fields["accumulation_dtype"] == str(torch.float32)
    remedy = fields["remedy"]
    assert "torch.bfloat16" in remedy
    assert "torch.float32-accumulate" in remedy
    assert "RECONSTRUCTION_ULP_HEADROOM[torch.bfloat16]" in remedy
    assert "not a precision artifact" in remedy
    assert "Remedy:" in str(excinfo.value)


def test_fp32_convention_mismatch_remedy_has_no_half_precision_line() -> None:
    """The dtype-aware remedy line is reserved for half-precision payloads."""

    dtype = torch.float32
    eps = 1e-5
    x = _norm_input(dtype, seed=8)
    weight = 0.3 * torch.randn(D_MODEL, generator=torch.Generator().manual_seed(9))
    y = F.layer_norm(x, (D_MODEL,), eps=eps) * (1.0 + weight)
    with pytest.raises(NormReconstructionError) as excinfo:
        reconstruct_norm(input=x, output=y, gamma=weight, beta=torch.zeros(D_MODEL), eps=eps)
    fields = excinfo.value.fields
    assert fields["code"] == "norm_convention_unmatched"
    assert fields["payload_dtype"] == str(torch.float32)
    assert fields["accumulation_dtype"] == str(torch.float32)
    assert "accumulate" not in fields["remedy"]


def test_fp32_layer_norm_behaviour_unchanged() -> None:
    """fp32 payloads accumulate as themselves: same kind, fp32 scale, tiny residual."""

    module, x, y = _layer_norm(torch.float32)
    record = reconstruct_norm(
        input=x,
        output=y,
        gamma=module.weight.detach(),
        beta=module.bias.detach(),
        eps=module.eps,
    )
    assert record.kind == "layernorm_affine"
    assert record.scale.dtype == torch.float32
    assert record.validation_receipt["accumulation_dtype"] == str(torch.float32)
    assert record.validation_receipt["max_abs_residual"] < 1e-4


def test_apply_frozen_folds_half_component_through_fp32_scale() -> None:
    """A float32 frozen scale on a bf16 payload is the correct fold.

    ``apply_norm_scale`` casts rows to float32 before ``apply_frozen``; a raw
    bf16 component promotes to float32 through the fp32 scale, and the stack
    of components pushed through the frozen map sums to the norm output minus
    the beta constant (the additivity the D8 record exists to guarantee).
    """

    module, x, y = _layer_norm(torch.bfloat16)
    record = reconstruct_norm(
        input=x,
        output=y,
        gamma=module.weight.detach(),
        beta=module.bias.detach(),
        eps=module.eps,
    )
    generator = torch.Generator().manual_seed(10)
    part = (torch.randn(2, 5, D_MODEL, generator=generator)).to(torch.bfloat16)
    rest = (x.float() - part.float()).to(torch.float32)
    folded_sum = record.apply_frozen(part.float()) + record.apply_frozen(rest)
    assert folded_sum.dtype == torch.float32
    expected = y.float() - module.bias.detach().float()
    one_ulp = float(torch.finfo(torch.bfloat16).eps) * float(y.float().abs().max())
    assert float((folded_sum - expected).abs().max()) <= 2.0 * one_ulp
    assert record.apply_frozen(part).dtype == torch.float32
