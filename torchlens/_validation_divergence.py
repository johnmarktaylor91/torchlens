"""First-divergence hints between two validation traces of one model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._record_type import record_type
from ._utils._record_group_table import record_member

if TYPE_CHECKING:
    from ._data_classes.trace import Trace


def _layer_signature(layer: Any) -> tuple[Any, ...]:
    """Cheap structural signature of one layer record for divergence hints."""

    return (
        record_type(layer, None),
        getattr(layer, "func_name", None),
        len(getattr(layer, "parents", ()) or ()),
        bool(getattr(layer, "is_output", False)),
        bool(record_member(layer, "is_buffer", False)),
    )


def _first_reproducibility_divergence(left: Trace, right: Trace) -> str | None:
    """Return a concise first-difference hint for two validation traces.

    Parameters
    ----------
    left:
        First validation trace.
    right:
        Second validation trace.

    Returns
    -------
    str | None
        Human-readable first divergence hint, or ``None`` when no cheap hint is
        available.
    """

    for index, (left_layer, right_layer) in enumerate(zip(left._layer_list, right._layer_list)):
        if _layer_signature(left_layer) != _layer_signature(right_layer):
            return (
                f"first divergence at op index {index}: "
                f"{getattr(left_layer, 'func_name', None)!r} vs "
                f"{getattr(right_layer, 'func_name', None)!r}"
            )
    if len(left._layer_list) != len(right._layer_list):
        return f"op count changed: {len(left._layer_list)} vs {len(right._layer_list)}"
    return None
