"""Profile table row assembly: one row's summed and listed op values."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .._utils._record_group_table import record_member

if TYPE_CHECKING:
    pass


def _sum_optional(ops: list[Any], field: str) -> int | float | None:
    """Sum a metric without turning wholly unavailable metadata into zero.

    Parameters
    ----------
    ops:
        Operations to aggregate.
    field:
        Numeric metadata field.

    Returns
    -------
    int | float | None
        Sum when at least one value is available, otherwise ``None``.
    """

    values = [record_member(op, field, None) for op in ops]
    if field == "func_duration":
        available = [float(value) for value in values if value is not None]
    else:
        available = [int(value) for value in values if value is not None]
    return sum(available) if available else None


def _values(ops: list[Any], field: str) -> str | None:
    """Format the distinct non-null values of one operation metadata field.

    Parameters
    ----------
    ops:
        Operations to inspect.
    field:
        Metadata field name.

    Returns
    -------
    str | None
        One value or a comma-separated stable set, if available.
    """

    values = sorted(
        {str(value) for op in ops if (value := record_member(op, field, None)) is not None}
    )
    return ", ".join(values) if values else None


def _row(name: str, kind: str, ops: list[Any], *, param_count: int | None) -> dict[str, Any]:
    """Build one consistent profile row from operation metadata.

    Parameters
    ----------
    name:
        Stable row label.
    kind:
        Row granularity label.
    ops:
        Operations represented by the row.
    param_count:
        Parameter count owned by the corresponding module scope.

    Returns
    -------
    dict[str, Any]
        Profile row.
    """

    saved = [bool(getattr(op, "has_saved_activation", False)) for op in ops]
    saved_activation: bool | str | None
    if not saved:
        saved_activation = None
    elif all(saved):
        saved_activation = True
    elif not any(saved):
        saved_activation = False
    else:
        saved_activation = "partial"
    return {
        "name": name,
        "kind": kind,
        "op_count": len(ops),
        "time": _sum_optional(ops, "func_duration"),
        "flops": _sum_optional(ops, "flops_forward"),
        "activation_memory": _sum_optional(ops, "activation_memory"),
        "saved_activation": saved_activation,
        "param_count": param_count,
        "dtype": _values(ops, "dtype"),
        # ``Op`` has no ``device`` attribute; the canonical field is ``device_ref``
        # (a ``DeviceRef`` whose ``str`` is the device name, e.g. ``"cpu"``).
        # Reading the nonexistent ``device`` left this column permanently ``None``
        # on every model via the ``getattr(op, field, None)`` default.
        "device": _values(ops, "device_ref"),
    }
