"""``dataclasses.replace`` with the per-class field scan done once.

``dataclasses.replace`` walks the class's field table in Python on every call
(one ``getattr`` and three tests per field) before it calls the class. The capture
door evolves its many-field resolution state about ten times per capture, so that
scan is a measurable share of the fixed per-capture cost. :func:`replace_fields`
reads the same field values in C (``operator.attrgetter``) and calls the same
class ``__init__`` with the same keyword arguments, so the new object, and every
error a bad call raises, are those of ``dataclasses.replace``. A class whose
field table holds anything but plain ``init`` fields (an ``InitVar`` or an
``init=False`` field, where ``replace`` adds checks) is passed to
``dataclasses.replace`` itself.
"""

from __future__ import annotations

import dataclasses
from operator import attrgetter
from typing import Any, TypeVar

__all__ = ["replace_fields"]

_T = TypeVar("_T")

#: class -> (the field table the plan was read from, field names, getter), or
#: ``None`` for a class ``dataclasses.replace`` must handle.
_PLANS: dict[type, tuple[dict[str, Any], tuple[str, ...], Any] | None] = {}


def _plan_for(cls: type) -> tuple[dict[str, Any], tuple[str, ...], Any] | None:
    """Return the replace plan for ``cls``, read from its current field table."""

    table = getattr(cls, "__dataclass_fields__", None)
    if not isinstance(table, dict):
        return None
    plan = _PLANS.get(cls)
    if plan is not None and plan[0] is table:
        return plan
    plain = getattr(dataclasses, "_FIELD", None)
    classvar = getattr(dataclasses, "_FIELD_CLASSVAR", None)
    if plain is None or classvar is None:
        return None
    names: list[str] = []
    for field in table.values():
        kind = getattr(field, "_field_type", None)
        if kind is classvar:
            continue
        if kind is not plain or not field.init:
            _PLANS[cls] = None
            return None
        names.append(field.name)
    if not names:
        _PLANS[cls] = None
        return None
    plan = (table, tuple(names), attrgetter(*names))
    _PLANS[cls] = plan
    return plan


def replace_fields(obj: _T, /, **changes: Any) -> _T:
    """Return ``dataclasses.replace(obj, **changes)``.

    Parameters
    ----------
    obj:
        Dataclass instance to copy.
    **changes:
        Field values that differ in the copy.

    Returns
    -------
    _T
        A new instance built by ``type(obj)``'s ``__init__`` from ``obj``'s field
        values overlaid with ``changes``.
    """

    cls = type(obj)
    plan = _plan_for(cls)
    if plan is None:
        return dataclasses.replace(obj, **changes)  # type: ignore[type-var]
    _table, names, getter = plan
    try:
        values = getter(obj)
    except AttributeError:
        # An unset field ``replace`` would not read (it is in ``changes``) or would
        # report itself: let ``replace`` decide.
        return dataclasses.replace(obj, **changes)  # type: ignore[type-var]
    kwargs = dict(zip(names, values if len(names) > 1 else (values,), strict=True))
    kwargs.update(changes)
    return cls(**kwargs)
