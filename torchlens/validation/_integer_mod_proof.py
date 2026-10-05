"""Value proof for an integer dividend modulo +-1 (``remainder`` / ``fmod``).

``integer_mod_by_unit_is_identically_zero`` decides, from the saved call only,
that the result is zero for every dividend value, so the dividend cannot reach
the output; ``exemptions._integer_mod_unit_divisor_decision`` adds the
perturbed-slot condition and emits the posthoc decision. Split from
``validation/exemptions.py`` (R43 file-size ratchet); the checks are unchanged.
"""

from typing import Any

import torch


def _is_unsigned_integer_dtype(dtype: torch.dtype) -> bool:
    """Return whether ``dtype`` is an unsigned integer dtype (bool excluded).

    Parameters
    ----------
    dtype:
        Torch dtype to classify.

    Returns
    -------
    bool
        True for ``uint8``/``uint16``/``uint32``/``uint64``.
    """

    return (
        dtype != torch.bool
        and not dtype.is_floating_point
        and not dtype.is_complex
        and not dtype.is_signed
    )


def _float_dtype_holds_integer_range(float_dtype: torch.dtype, int_dtype: torch.dtype) -> bool:
    """Return whether every value of ``int_dtype`` is finite in ``float_dtype``.

    Parameters
    ----------
    float_dtype:
        Floating result dtype.
    int_dtype:
        Integer or bool dividend dtype.

    Returns
    -------
    bool
        True when ``finfo(float_dtype).max >= iinfo(int_dtype).max``.
    """

    int_max = 1 if int_dtype == torch.bool else torch.iinfo(int_dtype).max
    return float(torch.finfo(float_dtype).max) >= float(int_max)


def _mod_divisor_values_as_torch_sees_them(divisor: Any) -> torch.Tensor | None:
    """Return the divisor's values widened so no wraparound hides them.

    A uint8 ``255`` compares equal to ``-1`` in its own dtype; widening to
    int64 (integer/bool) or float64 (floating) compares the true values.

    Parameters
    ----------
    divisor:
        Saved divisor argument (tensor or Python number).

    Returns
    -------
    torch.Tensor | None
        Widened divisor values, or None when the divisor is not a non-empty
        real tensor or a Python int/float.
    """

    if isinstance(divisor, torch.Tensor):
        if divisor.numel() == 0 or divisor.dtype.is_complex:
            return None
        wide = torch.float64 if divisor.dtype.is_floating_point else torch.int64
        return divisor.detach().to(device="cpu", dtype=wide)
    if isinstance(divisor, bool) or not isinstance(divisor, (int, float)):
        return None
    return torch.tensor(float(divisor), dtype=torch.float64)


def integer_mod_by_unit_is_identically_zero(dividend: Any, divisor: Any, out: Any) -> bool:
    """Return whether a saved integer ``% +-1`` call is identically zero.

    For integers, ``remainder(a, d)`` and ``fmod(a, d)`` equal ``a - d * q``
    with ``q`` an integer quotient; when ``|d| == 1`` the quotient is ``a``
    itself (``a / +-1`` is exact), so the result is 0 for EVERY integer ``a``.
    A float divisor of +-1.0 gives the same: an integer-valued float has no
    fractional part, so its remainder by 1.0 is +-0.0. The dividend's values
    therefore cannot reach the output. This is what ``torch.distributions``
    integer-support checks compute (``value % 1 == 0``) on sampled indices.

    The proof is taken from the saved call only (the caller adds the
    perturbed-slot condition): the dividend must be an integer or bool tensor,
    the divisor a Python int/float or a tensor whose every element (widened to
    int64/float64, so a uint8 255 is not -1) is exactly +1 or -1, or exactly +1
    when any operand, the result or the computation dtype is unsigned (there -1
    wraps to the dtype's max); a floating result or computation dtype must hold
    the dividend dtype's whole range (float16 does not: large integers become
    inf and the remainder nan), and the saved output must be all zeros. A float
    dividend or any other divisor value falls through to the failure path.

    Parameters
    ----------
    dividend:
        Saved dividend argument.
    divisor:
        Saved divisor argument.
    out:
        Saved output of the call.

    Returns
    -------
    bool
        True only when every saved-call condition above holds.
    """

    if not isinstance(dividend, torch.Tensor):
        return False
    if dividend.dtype.is_floating_point or dividend.dtype.is_complex:
        return False
    if not isinstance(out, torch.Tensor) or bool(torch.count_nonzero(out)):
        return False
    divisor_values = _mod_divisor_values_as_torch_sees_them(divisor)
    if divisor_values is None:
        return False
    # Torch computes in the operands' promoted dtype, then casts into ``out=``;
    # the proof must hold in both (float16 math written to a float32 buffer is
    # still float16 math).
    compute_dtype = torch.result_type(dividend, divisor)
    divisor_dtype = divisor.dtype if isinstance(divisor, torch.Tensor) else torch.int64
    operand_dtypes = (dividend.dtype, out.dtype, compute_dtype, divisor_dtype)
    if not _divisor_is_unit_for_dtypes(divisor_values, operand_dtypes):
        return False
    # A floating result must hold every value of the dividend's dtype; float16
    # turns integers above 65504 into inf and the remainder into nan.
    return not any(
        dtype.is_floating_point and not _float_dtype_holds_integer_range(dtype, dividend.dtype)
        for dtype in (out.dtype, compute_dtype)
    )


def _divisor_is_unit_for_dtypes(
    divisor_values: torch.Tensor, operand_dtypes: tuple[torch.dtype, ...]
) -> bool:
    """Return whether every widened divisor value is a unit the dtypes keep exact.

    Parameters
    ----------
    divisor_values:
        Divisor values widened by ``_mod_divisor_values_as_torch_sees_them``.
    operand_dtypes:
        Dividend, result, computation and divisor dtypes.

    Returns
    -------
    bool
        True when every value is +-1, or exactly +1 when any dtype is unsigned.
    """

    # An unsigned operand or result wraps -1 to the dtype's max (uint8: 255),
    # so ``x % -1`` there is ``x % 255``; only +1 keeps the proof.
    unsigned_involved = any(_is_unsigned_integer_dtype(dtype) for dtype in operand_dtypes)
    allowed = (divisor_values == 1) if unsigned_involved else (divisor_values.abs() == 1)
    return bool(allowed.all())
