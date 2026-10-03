"""r7 R74 F1 (opus b9 MED-HIGH, PROVEN): pin the forward-replay tolerance band.

The band behind every per-op replay verdict (`_tolerances_for_dtype`,
consumed by ``validation/core.py`` with ``allow_tolerance=True``) appeared in
ZERO test files, while the sibling GRAD band is pinned hard
(``tests/test_grad_tolerance_dtype_tripwire.py``). The comparator's own entry
self-test is a PRESENCE check, not a TIGHTNESS check: opus binary-searched the
fp32 row and measured that an rtol of 0.3333 -- 5,461x looser than the shipped
6.1e-5 -- still passes the self-test and blesses 30% corruption of every
replayed activation. This file is the union of the two independent hunt-6
pins: exact derived rows (table and out-of-table), rtol ceilings no future
headroom change can walk out of, corruption-fails assertions per dtype family
through both the raw band and the real ``tensor_nanequal`` comparator path,
and a poisoned-band self-test red check.

The shipped band is correct; this is the tripwire that keeps it so.
"""

from __future__ import annotations

import pytest
import torch

from torchlens.utils.tensor_utils import (
    _ACCUMULATING_REPLAY_ULP_HEADROOM,
    _DTYPE_FLOAT_TOLERANCES,
    _LOW_PRECISION_REPLAY_ULP_HEADROOM,
    _REPLAY_ULP_HEADROOM,
    _tolerances_for_dtype,
    derive_float_tolerances,
    tensor_nanequal,
)

#: No replay row may ever reach this: bf16 (the loosest shipped row) is
#: 4 * eps = 3.125e-2, and anything approaching O(1) blesses gross corruption
#: (the accidental 512-ULP derivation for a storage-rounding dtype produced
#: rtol 0.5 -- exactly the class this ceiling refuses).
_RTOL_CEILING = 0.05

#: LITERAL headroom pins spelled HERE, independently of the source constants
#: (R13 walk-out: asserting rows against ``derive_float_tolerances(d,
#: H_imported)`` was tautological -- bumping the source headroom 512 -> 8192
#: kept every pin green because test and source co-drifted). Any source-side
#: headroom change must consciously edit these numbers.
_PINNED_HEADROOMS: dict[torch.dtype, float] = {
    torch.float32: 512.0,
    torch.float64: 512.0,
    torch.float16: 4.0,
    torch.bfloat16: 4.0,
}


def _pinned_expected_row(dtype: torch.dtype) -> tuple[float, float]:
    """Return the (rtol, atol) row from torch.finfo and the LITERAL headroom.

    Spelled from first principles (headroom * eps, headroom * tiny * eps)
    rather than through ``derive_float_tolerances``, so the expected value
    cannot co-drift with either the source headroom constants or the
    derivation helper.
    """

    finfo = torch.finfo(dtype)
    headroom = _PINNED_HEADROOMS[dtype]
    return (
        headroom * float(finfo.eps),
        headroom * float(finfo.tiny) * float(finfo.eps),
    )


def test_headroom_constants_are_pinned_literals() -> None:
    """The source headroom constants and map rows equal the pinned literals.

    R13 fix (a): the source constants themselves are pinned, so a headroom
    walk-out (512 -> 8192 measured green pre-fix) goes red here directly.
    """

    assert _ACCUMULATING_REPLAY_ULP_HEADROOM == 512.0
    assert _LOW_PRECISION_REPLAY_ULP_HEADROOM == 4.0
    assert _REPLAY_ULP_HEADROOM == _PINNED_HEADROOMS


def test_rows_are_exactly_the_derived_ulp_model() -> None:
    """Each shipped row equals the finfo derivation at the PINNED headroom.

    R13 fix (b): the expected pair is computed from torch.finfo and the
    literal headrooms spelled in THIS file (never the imported constants),
    and the derivation helper itself is held to the same formula.
    """

    for dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
        expected = _pinned_expected_row(dtype)
        assert _tolerances_for_dtype(dtype) == expected, dtype
        assert derive_float_tolerances(dtype, _PINNED_HEADROOMS[dtype]) == expected, dtype


def test_fp32_row_is_the_exported_legacy_pair() -> None:
    """The facade constants and the fp32 replay row must not drift apart."""

    from torchlens.utils.tensor_utils import (
        MAX_FLOATING_POINT_TOLERANCE,
        REL_FLOATING_POINT_TOLERANCE,
    )

    assert _tolerances_for_dtype(torch.float32) == (
        REL_FLOATING_POINT_TOLERANCE,
        MAX_FLOATING_POINT_TOLERANCE,
    )


def test_fp64_row_is_tighter_than_fp32_by_the_eps_ratio() -> None:
    """fp64 replays earn the fp32 ULP budget in fp64 ULPs, not fp32 decimals."""

    eps_ratio = float(torch.finfo(torch.float64).eps) / float(torch.finfo(torch.float32).eps)
    rtol32, _ = _tolerances_for_dtype(torch.float32)
    rtol64, _ = _tolerances_for_dtype(torch.float64)
    assert rtol64 == pytest.approx(rtol32 * eps_ratio)


def test_every_row_respects_the_eps_floor_and_the_rtol_ceiling() -> None:
    """One-ULP agreement must pass; no row may drift toward O(1) looseness."""

    for dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
        rtol, atol = _tolerances_for_dtype(dtype)
        eps = float(torch.finfo(dtype).eps)
        assert rtol >= eps, f"{dtype}: below one ULP false-fails every replay"
        assert rtol < _RTOL_CEILING, f"{dtype}: rtol {rtol} walked toward blessing corruption"
        assert atol < eps, f"{dtype}: atol {atol} blesses corruption of small normal values"


def test_derivation_cache_holds_the_derived_rows() -> None:
    """The process-global cache rows equal the PINNED derivation (poison pin)."""

    for dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
        assert _DTYPE_FLOAT_TOLERANCES[dtype] == _pinned_expected_row(dtype), dtype


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
def test_ten_percent_corruption_fails_in_every_dtype_family(dtype: torch.dtype) -> None:
    """10% relative corruption must read UNEQUAL at every dtype's band.

    The loosest shipped row is bf16 at 4 ULP = 3.125e-2, so 10% clears every
    band with margin; a future loosening that blesses this class is exactly
    the corruption the comparator exists to catch.
    """

    values = torch.tensor([1.0, -2.0, 0.5, 8.0], dtype=dtype)
    corrupted = values * 1.10
    assert not tensor_nanequal(values, corrupted, allow_tolerance=True)


def test_zeroed_small_values_fail_fp32_and_fp64() -> None:
    """The atol term must never bless total corruption of small values.

    The former decimal atol floors (1e-5 fp32) read ``zeros == full(1e-6)``
    EQUAL -- zero detection power below the floor. The subnormal-quantum atol
    model must read it UNEQUAL.
    """

    for dtype in (torch.float32, torch.float64):
        true_values = torch.full((8,), 1e-6, dtype=dtype)
        zeroed = torch.zeros_like(true_values)
        assert not tensor_nanequal(zeroed, true_values, allow_tolerance=True), dtype


def test_comparator_self_test_bounds_the_effective_band(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 0.33-rtol poisoned band must FAIL the comparator self-test.

    Red-capable (the R74r6-F1 measurement): pre-fix the self-test's loosest
    sentinel gap was 1/3 relative, so ``rtol = 0.33`` -- ~5,400x looser than
    the shipped fp32 row, blessing 30% corruption -- still passed (measured
    at the hunt-6 pin). The two-sided band sentinels now reject it, and an
    absurdly TIGHT band fails the accept-side sentinel too.
    """

    from torchlens.utils import tensor_utils as tensor_utils_mod
    from torchlens.validation.core import _comparator_self_test

    _comparator_self_test()  # healthy band passes

    monkeypatch.setitem(tensor_utils_mod._DTYPE_FLOAT_TOLERANCES, torch.float32, (0.33, 1e-5))
    with pytest.raises(RuntimeError, match="comparator self-test failed"):
        _comparator_self_test()

    monkeypatch.setitem(tensor_utils_mod._DTYPE_FLOAT_TOLERANCES, torch.float32, (1e-9, 0.0))
    with pytest.raises(RuntimeError, match="comparator self-test failed"):
        _comparator_self_test()


#: The four precomputed rows and the corruption factor each must reject
#: (chosen far above the row's rtol so the assertion is about the BAND, not
#: about float noise: 1% >> 512*eps32, 10% >> 4*eps16, 30% >> 4*eps_bf16).
#: Headrooms are the PINNED literals, never the imported source constants.
_ROW_CASES = (
    (torch.float32, 512.0, 1.01),
    (torch.float64, 512.0, 1.01),
    (torch.float16, 4.0, 1.10),
    (torch.bfloat16, 4.0, 1.30),
)


def test_replay_rows_equal_their_derivation_exactly() -> None:
    """Each precomputed row is exactly its PINNED finfo derivation."""

    for dtype, _, _ in _ROW_CASES:
        expected = _pinned_expected_row(dtype)
        assert _tolerances_for_dtype(dtype) == expected, dtype
        # The mutable process-global cache row matches a fresh derivation:
        # a poisoned or leaked row (the cache is written on first use) is a
        # verdict threshold, so drift here is corruption, not staleness.
        assert _DTYPE_FLOAT_TOLERANCES[dtype] == expected, dtype
    assert _REPLAY_ULP_HEADROOM[torch.float64] == _REPLAY_ULP_HEADROOM[torch.float32]


def test_fp32_and_fp64_reject_corruption_at_the_walkout_window() -> None:
    """R13 fix (c): 1 + 1e-4 relative corruption must FAIL for fp32/fp64.

    This pins the measured walk-out window: bumping the accumulating
    headroom 512 -> 8192 (16x looser) kept all prior pins green and was
    caught only at 8388 by an unrelated comparator sentinel literal. 1e-4
    sits 1.64x above the shipped fp32 rtol (512 * eps32 = 6.1e-5), so ANY
    accumulating headroom >= 840 goes red here through the REAL
    ``tensor_nanequal`` comparator path.
    """

    for dtype in (torch.float32, torch.float64):
        base64 = torch.tensor([1.0, -2.0, 0.5, 8.0], dtype=torch.float64)
        base = base64.to(dtype)
        corrupted = (base64 * (1.0 + 1.0e-4)).to(dtype)
        assert not tensor_nanequal(base, corrupted, allow_tolerance=True), dtype
        # Non-vacuity: the uncorrupted pair still reads EQUAL.
        assert tensor_nanequal(base, base.clone(), allow_tolerance=True), dtype


def test_deep_numeric_replay_constants_are_pinned_literals() -> None:
    """R13 fix (e): pin the band-C ceiling/derivation constants literally.

    These are absolute verdict ceilings for the deep-numeric replay lane
    (``validation/core.py``); a silent bump is a tolerance walk-out of the
    same class as the headroom walk pinned above. Includes the R13
    dynamic-range guard constant introduced with the atol-amplification fix.
    """

    from torchlens.validation import core as validation_core

    assert validation_core.DEEP_NUMERIC_REPLAY_MIN_REDUCTION_DEPTH == 64
    assert validation_core.DEEP_NUMERIC_REPLAY_RTOL == 1e-3
    assert validation_core.DEEP_NUMERIC_REPLAY_ATOL == 1e-4
    assert validation_core.DEEP_NUMERIC_REPLAY_OUTLIER_RTOL == 5e-2
    assert validation_core.DEEP_NUMERIC_REPLAY_OUTLIER_ATOL == 1e-2
    assert validation_core.DEEP_NUMERIC_REPLAY_MAX_OUTLIER_FRACTION == 1e-4
    assert validation_core.DEEP_NUMERIC_REPLAY_MAX_SCALED_DIFF == 5e-2
    assert validation_core.DEEP_NUMERIC_REPLAY_MAX_MEAN_SCALED_DIFF == 1e-3
    assert validation_core.DEEP_NUMERIC_REPLAY_BASE_SQRT_DEPTH_FACTOR == 16.0
    assert validation_core.DEEP_NUMERIC_REPLAY_OUTLIER_SQRT_DEPTH_FACTOR == 128.0
    assert validation_core.DEEP_NUMERIC_REPLAY_STORAGE_ULP_HEADROOM == 4.0
    assert validation_core.DEEP_NUMERIC_REPLAY_MAX_ATOL_DYNAMIC_RANGE == 1e4


def test_replay_rtol_stays_far_below_the_self_test_blind_spot() -> None:
    """No row's rtol may approach 1/3 -- the comparator self-test's blind spot.

    The entry self-test's only relative sentinel is a 1/3 gap (0.5 vs 0.75),
    so any band under 1/3 passes it. This ceiling (1/8) keeps every row an
    order of magnitude inside that, so a future headroom bump must
    consciously edit this test rather than silently walk the band out.
    """

    for dtype, _, _ in _ROW_CASES:
        rtol, atol = _tolerances_for_dtype(dtype)
        eps = float(torch.finfo(dtype).eps)
        assert eps <= rtol < 1.0 / 8.0, (dtype, rtol)
        # atol forgives jitter only at the bottom of the representable range:
        # it must sit below the smallest NORMAL value, never blessing total
        # corruption of small normals (the pre-derivation decimal floors did).
        assert 0.0 < atol < float(torch.finfo(dtype).tiny), (dtype, atol)


def test_replay_band_rejects_corruption_per_dtype_family() -> None:
    """Uniform relative corruption well above each row's rtol must FAIL the band."""

    for dtype, _, corruption in _ROW_CASES:
        rtol, atol = _tolerances_for_dtype(dtype)
        base = torch.linspace(0.1, 4.0, 64).to(dtype)
        corrupted = (base.to(torch.float64) * corruption).to(dtype)
        assert not torch.allclose(base, corrupted, rtol=rtol, atol=atol), (
            f"{dtype}: {corruption:.2f}x corruption passed the replay band "
            f"(rtol={rtol}, atol={atol})"
        )
        # And the band is not vacuously tight: the value itself passes.
        assert torch.allclose(base, base.clone(), rtol=rtol, atol=atol)


def test_derived_out_of_table_rows_follow_the_eps_class() -> None:
    """Out-of-table float/complex dtypes derive their own row by eps class.

    complex64's component eps equals fp32's -> accumulating headroom;
    complex32 (component eps ~9.8e-4, storage-rounding class) must take the
    few-ULP budget -- deriving it at 512 ULP produced rtol 0.5, a row that
    would bless 40% corruption (the docstring's own incident).
    """

    rtol64, _ = _tolerances_for_dtype(torch.complex64)
    assert rtol64 == pytest.approx(512.0 * float(torch.finfo(torch.complex64).eps))
    if hasattr(torch, "complex32"):
        rtol32, _ = _tolerances_for_dtype(torch.complex32)
        assert rtol32 == pytest.approx(4.0 * float(torch.finfo(torch.complex32).eps))
        assert rtol32 < 1.0 / 8.0
