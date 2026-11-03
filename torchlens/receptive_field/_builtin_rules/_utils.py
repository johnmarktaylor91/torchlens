"""Typed captured-parameter normalization for receptive-field rules."""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from numbers import Integral, Real
from typing import Any, cast

from .._rules import ReceptiveFieldRuleContext


def spatial_rank(context: ReceptiveFieldRuleContext, parameter: object) -> int | None:
    """Infer the spatial rank from a captured parameter or operation family.

    Parameters
    ----------
    context:
        Rule context for the captured operation.
    parameter:
        Captured scalar or spatial sequence, usually ``kernel_size``.

    Returns
    -------
    int | None
        Spatial rank, or ``None`` when it cannot be determined.
    """

    if isinstance(parameter, Sequence) and not isinstance(parameter, (str, bytes)):
        return len(parameter) or None
    name = context.op.func_name or ""
    for rank in (1, 2, 3):
        if name.endswith(f"{rank}d"):
            return rank
    if context.in_shapes and len(context.in_shapes[0]) >= 3:
        return len(context.in_shapes[0]) - 2
    return None


def int_tuple(value: object, rank: int | None = None) -> tuple[int, ...] | None:
    """Normalize a captured integer scalar or sequence.

    Parameters
    ----------
    value:
        Loosely typed captured value.
    rank:
        Required tuple rank. A scalar is repeated to this rank.

    Returns
    -------
    tuple[int, ...] | None
        Normalized values, or ``None`` when the capture is malformed.
    """

    raw_values: tuple[object, ...]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        raw_values = tuple(value)
    else:
        raw_values = (value,)
    if not raw_values:
        return None
    normalized: list[int] = []
    for item in raw_values:
        if not isinstance(item, Integral):
            return None
        normalized.append(int(cast("Any", item)))
    result = tuple(normalized)
    if rank is not None and len(result) == 1:
        result *= rank
    if rank is not None and len(result) != rank:
        return None
    return result


def int_config(
    context: ReceptiveFieldRuleContext,
    key: str,
    rank: int,
    *,
    default: object,
) -> tuple[int, ...] | None:
    """Read one captured config parameter as a fixed-rank integer tuple.

    Parameters
    ----------
    context:
        Rule context containing the salient configuration.
    key:
        Configuration key to read.
    rank:
        Required spatial rank.
    default:
        PyTorch default used when the salient key is absent.

    Returns
    -------
    tuple[int, ...] | None
        Normalized parameter, or ``None`` for malformed captured data.
    """

    return int_tuple(context.op.func_config.get(key, default), rank)


def number_tuple(value: object, rank: int) -> tuple[int | float | Fraction, ...] | None:
    """Normalize a captured real-number scalar or sequence to fixed rank.

    Parameters
    ----------
    value:
        Loosely typed captured value.
    rank:
        Required tuple rank.

    Returns
    -------
    tuple[int | float | Fraction, ...] | None
        Typed numeric values, or ``None`` when the capture is malformed.
    """

    raw_values: tuple[object, ...]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        raw_values = tuple(value)
    else:
        raw_values = (value,)
    if not raw_values:
        return None
    normalized_numbers: list[int | float | Fraction] = []
    for item in raw_values:
        if isinstance(item, Fraction):
            normalized_numbers.append(item)
        elif isinstance(item, Integral):
            normalized_numbers.append(int(cast("Any", item)))
        elif isinstance(item, Real):
            normalized_numbers.append(float(cast("Any", item)))
        else:
            return None
    result = tuple(normalized_numbers)
    if len(result) == 1:
        result *= rank
    if len(result) != rank:
        return None
    return result
