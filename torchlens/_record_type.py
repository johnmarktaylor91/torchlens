"""Operation type of an ``Op`` or of a ``Layer``-like record.

An ``Op`` names its operation type ``type``; ``Layer`` and the other
layer-shaped records (capture contexts, events, injected-op records) name it
``layer_type``. Readers that receive either kind call :func:`record_type`.
"""

from __future__ import annotations

from typing import Any

__tl_layer__ = "L0"

_MISSING: Any = object()


def record_type(record: Any, default: Any = _MISSING) -> Any:
    """Return ``record.type`` for an ``Op``, else ``record.layer_type``.

    Parameters
    ----------
    record:
        An ``Op``, a ``Layer`` or any record carrying ``layer_type``.
    default:
        Returned when a non-``Op`` record has no ``layer_type``; without it
        the missing attribute raises ``AttributeError``.

    Returns
    -------
    Any
        The operation type token (for example ``"linear"``).
    """

    from ._data_classes.op import Op

    if isinstance(record, Op):
        return record.type
    if default is _MISSING:
        return record.layer_type
    return getattr(record, "layer_type", default)
