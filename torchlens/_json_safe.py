"""Strict-JSON projection shared by the agent envelope and the agent report."""

from __future__ import annotations

import math
from typing import Any

__all__ = ["json_safe"]


def json_safe(value: Any) -> Any:
    """Recursively project a value into strict-JSON-safe form.

    Non-finite floats become tagged records (``{"nonfinite": "nan"}``) so the
    canonical serializer's ``allow_nan=False`` never sees a bare ``NaN`` --
    the defect the agent memo (3.6) names as otherwise ARRIVING with the
    stats tool. Dict keys are coerced to strings; tuples become lists.

    Parameters
    ----------
    value:
        Arbitrary JSON-adjacent value.

    Returns
    -------
    Any
        Strict-JSON-serializable projection.

    Examples
    --------
    >>> json_safe({"loss": float("nan"), "shape": (1, 3)})
    {'loss': {'nonfinite': 'nan'}, 'shape': [1, 3]}
    >>> json_safe([1, "a"]) == [1, "a"]
    True
    """

    if isinstance(value, float):
        if math.isnan(value):
            return {"nonfinite": "nan"}
        return {"nonfinite": "inf" if value > 0 else "-inf"} if math.isinf(value) else value
    if isinstance(value, bool | int | str) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]
    return repr(value)
