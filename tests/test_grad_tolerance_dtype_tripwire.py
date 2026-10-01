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

from torchlens.utils._torch_compat import get_cpu_half_kernels_support
from torchlens.utils.tensor_utils import (
    LAYER_GRAD_VALIDATION_ATOL,
    LAYER_GRAD_VALIDATION_RTOL,
    PARAM_GRAD_VALIDATION_ATOL,
    PARAM_GRAD_VALIDATION_RTOL,
    layer_grad_tolerances_for_dtype,
    param_grad_tolerances_for_dtype,
)

pytestmark = pytest.mark.smoke

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


def test_fp64_rows_are_tighter_than_fp32_by_the_eps_ratio() -> None:
    """fp64 gradients earn the fp32 ULP budget in fp64 ULPs, not fp32 decimals."""

    eps_ratio = float(torch.finfo(torch.float64).eps) / float(torch.finfo(torch.float32).eps)
    rtol, atol = param_grad_tolerances_for_dtype(torch.float64)
    assert rtol == pytest.approx(PARAM_GRAD_VALIDATION_RTOL * eps_ratio)
    assert atol == pytest.approx(rtol / 10.0)
    layer_rtol, layer_atol = layer_grad_tolerances_for_dtype(torch.float64)
    assert layer_rtol == pytest.approx(LAYER_GRAD_VALIDATION_RTOL * eps_ratio)
    assert layer_atol == pytest.approx(layer_rtol / 10.0)


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


@_requires_cpu_half_kernels
def test_fp16_one_ulp_agreement_passes_and_corruption_fails() -> None:
    """fp16 rows admit storage rounding while catching sign flips and zeroing."""

    rtol, atol = param_grad_tolerances_for_dtype(torch.float16)
    eps16 = float(torch.finfo(torch.float16).eps)
    assert rtol >= eps16, "a tolerance below one fp16 ULP false-fails every replay"
    assert rtol < 1.0, "the row must still catch sign flips and zeroed buffers"
    grads = torch.ones((8,), dtype=torch.float16)
    one_ulp = torch.nextafter(grads, torch.tensor(2.0, dtype=torch.float16))
    assert torch.allclose(one_ulp, grads, rtol=rtol, atol=atol)
    assert not torch.allclose(-grads, grads, rtol=rtol, atol=atol)
    assert not torch.allclose(torch.zeros_like(grads), grads, rtol=rtol, atol=atol)


def test_bf16_rows_scale_with_bf16_eps() -> None:
    """bf16 rows follow bf16's own eps, and still catch total corruption."""

    rtol, atol = layer_grad_tolerances_for_dtype(torch.bfloat16)
    eps = float(torch.finfo(torch.bfloat16).eps)
    assert rtol >= eps
    assert rtol < 1.0
    grads = torch.full((8,), 1e-2, dtype=torch.bfloat16)
    assert not torch.allclose(torch.zeros_like(grads), grads, rtol=rtol, atol=atol)


def test_layer_rows_are_tighter_than_param_rows_per_dtype() -> None:
    """Elementwise (layer) comparisons keep their 10x-tighter budget per dtype."""

    for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        param_rtol, _ = param_grad_tolerances_for_dtype(dtype)
        layer_rtol, _ = layer_grad_tolerances_for_dtype(dtype)
        assert layer_rtol < param_rtol


def test_complex_dtypes_use_component_precision() -> None:
    """Complex dtypes derive from their component real dtype's finfo."""

    assert param_grad_tolerances_for_dtype(torch.complex64) == param_grad_tolerances_for_dtype(
        torch.float32
    )
    assert param_grad_tolerances_for_dtype(torch.complex128) == param_grad_tolerances_for_dtype(
        torch.float64
    )


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

    import torchlens.validation.backward as backward_validation

    real_param_grads = backward_validation._param_grads
    call_count = {"n": 0}

    def _wrapped(model: torch.nn.Module) -> dict[str, torch.Tensor]:
        call_count["n"] += 1
        grads = real_param_grads(model)
        if call_count["n"] == 2:
            return {name: grad * scale for name, grad in grads.items()}
        return grads

    monkeypatch.setattr(backward_validation, "_param_grads", _wrapped)


def test_fp64_param_grad_corruption_fails_the_backward_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 1e-6-relative fp64 param-grad corruption must FAIL validation.

    1e-6 relative is ~4.5e9 fp64 ULPs -- far above fp64 accumulation
    round-off -- yet it sat 100x inside the fp32 decimal rtol (1e-4) that the
    verdict site applied to every dtype, so it was blessed (red-capable: this
    test FAILS before the dtype-aware wiring).
    """

    import torchlens.validation.backward as backward_validation

    _perturb_second_param_grad_census(monkeypatch, 1.0 + 1e-6)
    model = torch.nn.Linear(4, 3).double().eval()
    assert not backward_validation.validate_backward_pass(
        model,
        torch.randn(2, 4, dtype=torch.float64),
        random_seed=11,
        validate_metadata=False,
        validate_layer_grads=False,
    )


def test_fp64_param_grad_clean_run_still_passes() -> None:
    """The tightened fp64 row must not false-fail an honest fp64 capture."""

    import torchlens.validation.backward as backward_validation

    model = torch.nn.Linear(4, 3).double().eval()
    assert backward_validation.validate_backward_pass(
        model,
        torch.randn(2, 4, dtype=torch.float64),
        random_seed=11,
        validate_metadata=False,
        validate_layer_grads=False,
    )


@_requires_cpu_half_kernels
def test_fp16_few_ulp_param_grad_agreement_is_not_a_false_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 2-fp16-ULP census difference is storage rounding, not corruption.

    Under the fp32 decimal row (rtol 1e-4, ~1/10 of an fp16 ULP) this
    legitimate storage-rounding difference FALSE-FAILED; the fp16 row budgets
    a few storage ULPs (red-capable: this test FAILS before the wiring).
    """

    import torchlens.validation.backward as backward_validation

    eps16 = float(torch.finfo(torch.float16).eps)
    _perturb_second_param_grad_census(monkeypatch, 1.0 + 2.0 * eps16)
    model = torch.nn.Linear(4, 3).half().eval()
    assert backward_validation.validate_backward_pass(
        model,
        torch.randn(2, 4, dtype=torch.float16),
        random_seed=11,
        validate_metadata=False,
        validate_layer_grads=False,
    )


def test_explicit_tolerances_still_override_every_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit atol/rtol pair applies unchanged to every dtype."""

    import torchlens.validation.backward as backward_validation

    _perturb_second_param_grad_census(monkeypatch, 1.0 + 1e-6)
    model = torch.nn.Linear(4, 3).double().eval()
    # The same corruption the fp64 default now catches stays blessed under an
    # explicit legacy-loose override -- explicit user tolerances are honored.
    assert backward_validation.validate_backward_pass(
        model,
        torch.randn(2, 4, dtype=torch.float64),
        random_seed=11,
        validate_metadata=False,
        validate_layer_grads=False,
        atol=PARAM_GRAD_VALIDATION_ATOL,
        rtol=PARAM_GRAD_VALIDATION_RTOL,
    )


def test_fp64_layer_grad_corruption_fails_the_backward_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 1e-6-relative fp64 LAYER-grad corruption must FAIL validation.

    The layer-grad comparator applied the fp32 elementwise row (rtol 1e-5) to
    every dtype, so a 1e-6-relative fp64 module-output-grad corruption --
    ~4.5e9 fp64 ULPs -- was blessed (red-capable pre-wiring).
    """

    import torchlens.validation.backward as backward_validation
    from torchlens.validation import _stock_layer_grads as stock_module

    real_stock_layer_grads = stock_module._stock_layer_grads

    def _perturbed(*args: object, **kwargs: object) -> object:
        stock_grads, identity_addresses = real_stock_layer_grads(*args, **kwargs)
        return (
            {key: grad * (1.0 + 1e-6) for key, grad in stock_grads.items()},
            identity_addresses,
        )

    monkeypatch.setattr(stock_module, "_stock_layer_grads", _perturbed)
    model = torch.nn.Sequential(torch.nn.Linear(4, 4)).double().eval()
    assert not backward_validation.validate_backward_pass(
        model,
        torch.randn(2, 4, dtype=torch.float64),
        random_seed=11,
        validate_metadata=False,
    )


def test_legacy_public_spellings_default_to_dtype_derived_tolerances() -> None:
    """Every legacy public spelling defaults atol/rtol to None (R13 #a).

    The deprecated top-level shim and the ``_user_public_impls`` spelling
    pinned the fp32 decimal pair ``atol=1e-5, rtol=1e-4`` as non-None
    defaults passed unconditionally, keeping the dtype-aware derivation dead
    on 3 of 4 public entrypoints (fp64 false-PASS / fp16 false-FAIL shipped).
    """

    import inspect

    import torchlens
    import torchlens._user_public_impls as user_public_impls
    import torchlens.validation.backward as backward_validation

    for spelling in (
        torchlens.validation.validate_backward_pass,
        user_public_impls.validate_backward_pass,
        backward_validation.validate_backward_pass,
    ):
        params = inspect.signature(spelling).parameters
        assert params["atol"].default is None, spelling.__module__
        assert params["rtol"].default is None, spelling.__module__


@pytest.mark.parametrize(
    "spelling",
    ["top_level", "user_public_impls", "user_funcs"],
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
    import torchlens._user_public_impls as user_public_impls
    import torchlens.user_funcs as user_funcs

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
            random_seed=11,
            validate_metadata=False,
            validate_layer_grads=False,
        )
    assert passed is False
