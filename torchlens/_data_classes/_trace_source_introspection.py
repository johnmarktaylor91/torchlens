"""Source-model introspection fallbacks for ``Trace``'s ``_source_*`` fields.

When a trace's stored class or callable metadata is absent, the ``_source_*``
properties on ``Trace`` inspect the live source model, which the trace holds only
through a weak reference. Both helpers return ``None`` once that model is gone or
the trace was cleaned up.
"""

from __future__ import annotations

import inspect
from typing import Any


def source_model_class(trace: Any) -> type[Any] | None:
    """Return the live source model class if it is still retained.

    Parameters
    ----------
    trace:
        Trace whose ``_source_model_ref`` weakref names the source model.

    Returns
    -------
    type[Any] | None
        Source model class, or ``None`` if the weakref is unavailable.
    """

    source_ref = None if trace.__dict__.get("_tl_cleaned_up", False) else trace._source_model_ref
    model = source_ref() if source_ref is not None else None
    return None if model is None else type(model)


def inspect_source_attr(trace: Any, attr_name: str) -> str | None:
    """Inspect one source-model attribute when stored metadata is absent.

    Parameters
    ----------
    trace:
        Trace whose source model is inspected.
    attr_name:
        One of the Trace source-introspection field names.

    Returns
    -------
    str | None
        Inspected metadata, or ``None`` when the source model is gone or
        the callable cannot be inspected.
    """

    model_cls = source_model_class(trace)
    if model_cls is None:
        return None
    if attr_name == "_source_class_docstring":
        return model_cls.__doc__
    if attr_name in {"_source_init_signature", "_source_init_docstring"}:
        target = getattr(model_cls, "__init__", None)
    else:
        target = getattr(model_cls, "forward", None)
    if target is None:
        return None
    if attr_name.endswith("_docstring"):
        return getattr(target, "__doc__", None)
    try:
        return str(inspect.signature(target))
    except (TypeError, ValueError):
        return None
