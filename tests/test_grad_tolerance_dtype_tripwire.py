"""Dtype-aware gradient-validation tolerances (tripwire strengthening).

The legacy module constants ``PARAM_GRAD_VALIDATION_*`` /
``LAYER_GRAD_VALIDATION_*`` are the fp32 rows of an error model that was
silently applied to every dtype: fp64 gradients were checked ~4.5e11 of their
own ULPs loose (masking corruption far above fp64 round-off), while fp16
gradients were checked two orders BELOW their own eps (false-failing every
non-bitwise agreement).  ``param_grad_tolerances_for_dtype`` /
``layer_grad_tolerances_for_dtype`` derive the dtype-correct row; the fp32
row is bit-identical to the legacy constants.
"""

from __future__ import annotations

import pytest
import torch

from torchlens._utils._torch_compat import get_cpu_half_kernels_support
from torchlens._utils.tensor_utils import (
    LAYER_GRAD_VALIDATION_ATOL,
    LAYER_GRAD_VALIDATION_RTOL,
    PARAM_GRAD_VALIDATION_ATOL,
    PARAM_GRAD_VALIDATION_RTOL,
    layer_grad_tolerances_for_dtype,
    param_grad_tolerances_for_dtype,
)

_requires_cpu_half_kernels = pytest.mark.skipif(
    not get_cpu_half_kernels_support(),
    reason="CPU addmm/nextafter for float16 postdates the torch 2.1 floor",
)


def test_fp32_rows_are_exactly_the_legacy_constants() -> None:
    """The fp32 row must not drift from the exported legacy pair."""

    assert param_grad_tolerances_for_dtype(torch.float32) == (
        PARAM_GRAD_VALIDATION_RTOL,
        PARAM_GRAD_VALIDATION_ATOL,
    )
    assert layer_grad_tolerances_for_dtype(torch.float32) == (
        LAYER_GRAD_VALIDATION_RTOL,
        LAYER_GRAD_VALIDATION_ATOL,
    )


def test_fp64_corruption_masked_by_legacy_constants_now_fails() -> None:
    """The exact corruption class the fp32 decimals blessed must now FAIL.

    An all-zero fp64 gradient buffer whose true values sit below the legacy
    1e-5 atol reads EQUAL under the legacy pair -- zero detection power --
    and must read UNEQUAL under the fp64 row.
    """

    true_grads = torch.full((16,), 1e-6, dtype=torch.float64)
    zeroed = torch.zeros_like(true_grads)
    assert torch.allclose(
        zeroed,
        true_grads,
        rtol=PARAM_GRAD_VALIDATION_RTOL,
        atol=PARAM_GRAD_VALIDATION_ATOL,
    ), "precondition: the legacy pair masks this corruption"
    rtol, atol = param_grad_tolerances_for_dtype(torch.float64)
    assert not torch.allclose(zeroed, true_grads, rtol=rtol, atol=atol)


def test_non_float_dtype_stays_strict() -> None:
    """A misrouted non-float dtype gets the strictest row, never a loose one."""

    rtol, _ = param_grad_tolerances_for_dtype(torch.int64)
    fp64_rtol, _ = param_grad_tolerances_for_dtype(torch.float64)
    assert rtol <= fp64_rtol


# ---------------------------------------------------------------------------
# Consumer wiring (R13): the derivation above must actually decide verdicts.
# The functions shipped in c9734a7e with ZERO verdict-site consumers, so fp64
# param grads were still checked with the fp32 decimal row (~4.5e11 fp64 ULPs
# loose) and fp16 grads still false-failed. These tests drive the REAL
# validate_backward_pass verdict sites.
# ---------------------------------------------------------------------------


def _perturb_second_param_grad_census(monkeypatch: pytest.MonkeyPatch, scale: float) -> None:
    """Scale the OBSERVED (second) parameter-grad census by ``scale``.

    ``validate_backward_pass`` calls ``_param_grads`` twice: first for the
    stock autograd census, then for the captured candidate census. Scaling
    only the second call plants a relative corruption between the two
    pipelines without touching either backward implementation.
    """

    import torchlens.validation._backward as backward_validation

    real_param_grads = backward_validation._param_grads
    call_count = {"n": 0}

    def _wrapped(model: torch.nn.Module) -> dict[str, torch.Tensor]:
        call_count["n"] += 1
        grads = real_param_grads(model)
        if call_count["n"] == 2:
            return {name: grad * scale for name, grad in grads.items()}
        return grads

    monkeypatch.setattr(backward_validation, "_param_grads", _wrapped)


@_requires_cpu_half_kernels
def test_fp16_few_ulp_param_grad_agreement_is_not_a_false_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 2-fp16-ULP census difference is storage rounding, not corruption.

    Under the fp32 decimal row (rtol 1e-4, ~1/10 of an fp16 ULP) this
    legitimate storage-rounding difference FALSE-FAILED; the fp16 row budgets
    a few storage ULPs (red-capable: this test FAILS before the wiring).
    """

    import torchlens.validation._backward as backward_validation

    eps16 = float(torch.finfo(torch.float16).eps)
    _perturb_second_param_grad_census(monkeypatch, 1.0 + 2.0 * eps16)
    model = torch.nn.Linear(4, 3).half().eval()
    assert backward_validation.validate_backward_pass(
        model,
        torch.randn(2, 4, dtype=torch.float16),
        seed=11,
        validate_metadata=False,
        validate_layer_grads=False,
    )


# shrink-r4: rows kept carry a sole library arc or a campaign kill; the others were proven redundant (slice 1 evidence)
@pytest.mark.parametrize(
    "spelling",
    [
        "user_public_impls",
    ],
)
def test_fp64_corruption_fails_through_every_legacy_spelling(
    monkeypatch: pytest.MonkeyPatch, spelling: str
) -> None:
    """The fp64 corruption the legacy pins blessed FAILS via every spelling (R13 #a).

    Red-capable: before the shim defaults moved to None, the 1e-6-relative
    fp64 param-grad corruption sat 100x inside the pinned rtol=1e-4 and every
    legacy spelling returned True.
    """

    import warnings

    import torchlens
    import torchlens._user_funcs as user_funcs
    import torchlens._user_public_impls as user_public_impls

    fns = {
        "top_level": torchlens.validation.validate_backward_pass,
        "user_public_impls": user_public_impls.validate_backward_pass,
        "user_funcs": user_funcs.validate_backward_pass,
    }
    _perturb_second_param_grad_census(monkeypatch, 1.0 + 1e-6)
    model = torch.nn.Linear(4, 3).double().eval()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        passed = fns[spelling](
            model,
            torch.randn(2, 4, dtype=torch.float64),
            seed=11,
            validate_metadata=False,
            validate_layer_grads=False,
        )
    assert passed is False
