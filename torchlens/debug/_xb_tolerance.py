"""Tolerance policy for comparing two recorded values in ``tl.debug`` comparisons.

One function, :func:`tolerance_for`, decides how two paired values are judged, from
one constants table. Two runs on the same backend keep ``torch.allclose`` with
``rtol=1e-5, atol=1e-8`` exactly. Two runs on different backends are judged per
module output by relative RMS error, ``rms(a - b) <= atol + rtol * rms(a)``, with
``a`` the reference run (the first trace), both values compared at the wider dtype,
and the constants taken from the NARROWER dtype of the pair. Integer and boolean
values must match exactly either way.

The cross-backend constants were measured on a real port (SmolLM2-135M and -360M in
mlx-lm against the Hugging Face PyTorch reference, MLX on Metal, torch on CPU):
every module of a correct port passes with a margin of at least 2.3x in every dtype,
and in float32 19 of 22 injected one-percent-sized bugs are caught at their own
module. Half precision separates only gross bugs; that is the precision floor, not
a tuning gap, so half-precision rows say to compare float32 runs. float64 has no
measured port; it uses the float32 constant (a looser bound than float64 needs).

The cross-backend ``atol`` is 0, except when the reference value is all zero (its rms is
0, so the relative bound would demand exact zeros): then ``atol = c * eps(narrower dtype)
* s``, where ``s`` is the smallest rms among the module's finite, non-zero floating inputs
on side a (1.0 when there is none), never the output being judged. Taking the smallest
input keeps a causal-mask-like input (huge or infinite values) from widening the bound.
``c`` comes from :data:`CROSS_BACKEND_ATOL_FLOOR` by the narrower dtype, measured on
one small model with three seeds (a correct float32 LayerNorm over constant rows needs
``c >= 163``); a smaller LayerNorm eps would need a larger float32 ``c``, and in
bfloat16 no ``c`` separates small bugs on all-zero outputs from rounding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from .._utils._torch_symbols import torch_attr

#: Same-backend defaults: ``torch.allclose`` exactly as ``compare`` always used.
SAME_BACKEND_RTOL = 1e-5
SAME_BACKEND_ATOL = 1e-8

#: Cross-backend relative-RMS constants keyed by the narrower dtype of a pair.
CROSS_BACKEND_RTOL: dict[str, float] = {
    "bfloat16": 7.5e-1,
    "float16": 5e-2,
    "float32": 1e-4,
    "float64": 1e-4,
}
#: Multiplier ``c`` of the all-zero-reference floor ``c * eps * input rms``, keyed by the
#: narrower dtype of a pair. float64 has no measured port (uncalibrated, as for rtol).
CROSS_BACKEND_ATOL_FLOOR: dict[str, float] = {
    "bfloat16": 20.0,
    "float16": 10.0,
    "float32": 500.0,
    "float64": 500.0,
}

#: Mantissa width orders floating dtypes from narrowest to widest precision.
_PRECISION_BITS: dict[str, int] = {"bfloat16": 8, "float16": 11, "float32": 24, "float64": 53}
_HALF_PRECISION = frozenset({"bfloat16", "float16"})
HALF_PRECISION_NOTE = (
    "half precision: small port bugs sit below its precision floor and only gross "
    "bugs separate; compare float32 runs of both sides to hunt small bugs"
)

ToleranceForm = Literal["allclose", "relative_rms", "exact"]


@dataclass(frozen=True)
class Tolerance:
    """How one paired value is judged.

    Parameters
    ----------
    form:
        ``"allclose"`` (same backend), ``"relative_rms"`` (cross backend) or
        ``"exact"`` (integer and boolean values).
    rtol, atol:
        Effective tolerances printed on the row.
    narrower:
        The narrower floating dtype name of the pair, or ``None`` for exact values.
    note:
        Disclosure for the row's ``reason`` (half-precision floor), or ``""``.
    """

    form: ToleranceForm
    rtol: float
    atol: float
    narrower: str | None
    note: str


def _dtype_name(dtype: torch.dtype) -> str:
    """Return ``"float32"`` for ``torch.float32``."""

    return str(dtype).removeprefix("torch.")


def _narrower_float(dtype_a: torch.dtype, dtype_b: torch.dtype) -> str | None:
    """Return the narrower floating dtype name of the pair, or ``None`` if neither floats."""

    names = [_dtype_name(dtype) for dtype in (dtype_a, dtype_b) if dtype.is_floating_point]
    known = [name for name in names if name in _PRECISION_BITS]
    if not known:
        return None
    return min(known, key=_PRECISION_BITS.__getitem__)


def _atol_floor(narrower: str | None, scale: float) -> float:
    """``CROSS_BACKEND_ATOL_FLOOR[narrower] * eps(narrower dtype) * scale``."""

    eps = torch.finfo(torch_attr(narrower or "float32")).eps
    return CROSS_BACKEND_ATOL_FLOOR[narrower or "float32"] * eps * scale


def tolerance_for(
    dtype_a: torch.dtype,
    dtype_b: torch.dtype,
    *,
    cross_backend: bool,
    rtol: float | None,
    atol: float | None,
    zero_scale: float | None = None,
) -> Tolerance:
    """Return the tolerance for one paired value.

    Parameters
    ----------
    dtype_a, dtype_b:
        Recorded dtypes of the two values (before widening).
    cross_backend:
        Whether the two values come from runs on different backends.
    rtol, atol:
        User overrides, or ``None`` for the default. Across backends an override
        replaces the constant inside the same relative-RMS form.
    zero_scale:
        ``None`` when the reference value has a non-zero element (``atol`` stays 0 across
        backends); otherwise the scale of the floor (the module's smallest finite, non-zero
        side-a input rms, or 1.0). A user ``atol`` replaces either.

    Returns
    -------
    Tolerance
        Form, effective tolerances and the row disclosure.
    """

    exact = not any(d.is_floating_point or d.is_complex for d in (dtype_a, dtype_b))
    if exact:
        return Tolerance("exact", 0.0, 0.0, None, "")
    narrower = _narrower_float(dtype_a, dtype_b)
    if not cross_backend:
        return Tolerance(
            "allclose",
            SAME_BACKEND_RTOL if rtol is None else rtol,
            SAME_BACKEND_ATOL if atol is None else atol,
            narrower,
            "",
        )
    default_rtol = CROSS_BACKEND_RTOL.get(narrower or "float32", CROSS_BACKEND_RTOL["float32"])
    note = HALF_PRECISION_NOTE if narrower in _HALF_PRECISION and rtol is None else ""
    return Tolerance(
        "relative_rms",
        default_rtol if rtol is None else rtol,
        (0.0 if zero_scale is None else _atol_floor(narrower, zero_scale))
        if atol is None
        else atol,
        narrower,
        note,
    )


def judge(cmp_a: torch.Tensor, cmp_b: torch.Tensor, tol: Tolerance) -> tuple[float, float, bool]:
    """Return ``(max_abs, mean_abs, agree)`` for two same-shaped values at one dtype.

    Parameters
    ----------
    cmp_a, cmp_b:
        Aligned values (same shape, same device, the wider dtype); ``cmp_a`` is
        the reference for the relative-RMS scale.
    tol:
        The tolerance from :func:`tolerance_for`.

    Returns
    -------
    tuple[float, float, bool]
        Largest and mean absolute difference, and whether the pair agrees.
    """

    wide = torch.complex128 if torch.is_complex(cmp_a) else torch.float64
    delta = torch.abs(cmp_a.to(wide) - cmp_b.to(wide))
    max_abs = float(delta.max().item()) if delta.numel() else 0.0
    mean_abs = float(delta.mean().item()) if delta.numel() else 0.0
    if tol.form == "exact":
        return max_abs, mean_abs, bool(torch.equal(cmp_a, cmp_b))
    if tol.form == "allclose":
        return max_abs, mean_abs, bool(torch.allclose(cmp_a, cmp_b, rtol=tol.rtol, atol=tol.atol))
    if not delta.numel():
        return max_abs, mean_abs, True
    error = float(torch.sqrt(torch.mean(delta**2)).item())
    scale = float(torch.sqrt(torch.mean(torch.abs(cmp_a.to(wide)) ** 2)).item())
    return max_abs, mean_abs, error <= tol.atol + tol.rtol * scale
