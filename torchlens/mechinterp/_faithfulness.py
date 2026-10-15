"""Circuit faithfulness areas: CPR and CMD over a faithfulness curve.

A circuit sweep keeps the top fraction ``x`` of edges and measures the
faithfulness ``f(x) = (M_circuit - M_empty) / (M_full - M_empty)``. The MIB
circuit track summarizes the curve by two trapezoidal areas over the swept
fractions: CPR, the area under ``f`` (higher is better), and CMD, the area
under ``|1 - f|`` (lower is better). These functions reproduce that loop
exactly, from the values a sweep produced; they run no model.

Spellings DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from ._errors import refuse

__all__ = ["circuit_model_difference", "circuit_performance_ratio"]

#: The MIB circuit track's edge fractions (``evaluate_area_under_curve``).
MIB_FRACTIONS: tuple[float, ...] = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)


def _curve(
    faithfulness: Any, fractions: Sequence[float] | None, log_scale: bool, caller: str
) -> tuple[list[float], list[float]]:
    """Return ``(xs, fs)`` from a faithfulness sequence or ``{fraction: f}`` mapping."""

    if isinstance(faithfulness, Mapping):
        if fractions is not None:
            refuse(
                code="mi_faithfulness_curve_invalid",
                message=f"{caller} got a {{fraction: faithfulness}} mapping and fractions=; "
                "the mapping already names its fractions.",
                remedy=f"call {caller}(curve) with the mapping alone",
            )
        pairs = sorted((float(x), float(f)) for x, f in faithfulness.items())
        xs, fs = [x for x, _f in pairs], [f for _x, f in pairs]
    else:
        fs = [float(f) for f in faithfulness]
        xs = [float(x) for x in (MIB_FRACTIONS if fractions is None else fractions)]
    if len(xs) != len(fs) or len(fs) < 2:
        refuse(
            code="mi_faithfulness_curve_invalid",
            message=f"{caller} got {len(fs)} faithfulness values for {len(xs)} fractions; "
            "the curve needs one value per fraction and at least two points.",
            remedy=f"pass one value per fraction, for example {caller}(values) with the 10 "
            "MIB fractions, or fractions=[...] of the same length",
            n_values=len(fs),
            n_fractions=len(xs),
        )
    if any(not math.isfinite(v) for v in xs + fs) or any(b <= a for a, b in zip(xs, xs[1:])):
        refuse(
            code="mi_faithfulness_curve_invalid",
            message=f"{caller} needs finite values and strictly increasing fractions.",
            remedy="pass the sweep's fractions in increasing order with finite faithfulness",
        )
    if log_scale:
        if xs[0] <= 0:
            refuse(
                code="mi_faithfulness_curve_invalid",
                message=f"{caller}(log_scale=True) needs positive fractions.",
                remedy="drop log_scale=True or pass positive fractions",
            )
        xs = [math.log(x) for x in xs]
    return xs, fs


def _trapezoid(xs: list[float], heights: list[float]) -> float:
    """Trapezoidal area in the MIB loop's order of operations."""

    area = 0.0
    for i in range(len(heights) - 1):
        area += (xs[i + 1] - xs[i]) * ((heights[i] + heights[i + 1]) / 2)
    return area


def circuit_performance_ratio(
    faithfulness: Sequence[float] | Mapping[float, float],
    *,
    fractions: Sequence[float] | None = None,
    log_scale: bool = False,
) -> float:
    """CPR: the trapezoidal area under the faithfulness curve.

    Parameters
    ----------
    faithfulness:
        Faithfulness at each swept edge fraction: a sequence (one value per
        fraction) or a ``{fraction: faithfulness}`` mapping.
    fractions:
        Keyword-only. The edge fractions of a sequence; ``None`` means the
        ten MIB fractions (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1,
        0.2, 0.5, 1).
    log_scale:
        Keyword-only. Integrate over ``log(fraction)`` (MIB's ``log_scale``).

    Returns
    -------
    float
        The area under ``f`` (MIB ``area_under``); higher is better.
    """

    xs, fs = _curve(faithfulness, fractions, log_scale, "circuit_performance_ratio")
    return _trapezoid(xs, fs)


def circuit_model_difference(
    faithfulness: Sequence[float] | Mapping[float, float],
    *,
    fractions: Sequence[float] | None = None,
    log_scale: bool = False,
) -> float:
    """CMD: the trapezoidal area under ``|1 - faithfulness|``.

    Parameters
    ----------
    faithfulness:
        Faithfulness at each swept edge fraction: a sequence (one value per
        fraction) or a ``{fraction: faithfulness}`` mapping.
    fractions:
        Keyword-only. The edge fractions of a sequence; ``None`` means the
        ten MIB fractions (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1,
        0.2, 0.5, 1).
    log_scale:
        Keyword-only. Integrate over ``log(fraction)`` (MIB's ``log_scale``).

    Returns
    -------
    float
        The area under ``|1 - f|`` (MIB ``area_from_1``); lower is better.
    """

    xs, fs = _curve(faithfulness, fractions, log_scale, "circuit_model_difference")
    return _trapezoid(xs, [abs(1.0 - f) for f in fs])
