"""Deferred resolution for thin rows: observe at the call, resolve when the row fills.

The thin emit (:mod:`._thin_emit`) serves a plain admitted call. For a deferred family
the call-time step records only what the resolver needs, read while the call's state
exists, under one private fields key (:data:`DEFERRED`); the row's fill
(:meth:`._thin_record._ThinRow._fill`, on the first read that needs a settle field, or
at the settle) pops the key and runs each resolver on its recorded inputs before
anything reads the fields. The resolver is the builder's own expression on the
recorded objects, so the values are the ones today's emit writes at the call.

The deferred family is the argument template of a call that records one (a call that
records none resolves at the call: its resolve writes three constants).
``_set_arg_template_fields`` observes the facts at the call (``observe_template``: labels, capture layouts, literal values,
copies of unlabeled tensors, the function key) and :func:`resolve_arg_template_fields`,
a pure function of those facts and the layout plan, resolves them. The layout plan is
built once per function and layout and shared by every row that uses it; the per-row
part is the labels and literals. This module imports nothing from the capture import
cycle.
"""

from __future__ import annotations

from typing import Any

from ._arg_template_resolve import resolve_arg_template_fields

__all__ = ("DEFERRED", "defer_arg_template", "resolve_deferred")

DEFERRED = "_thin_deferred"
"""The private fields key holding a thin row's recorded template facts until its fill."""


def defer_arg_template(fields: dict[str, Any], observed: tuple[Any, ...], parts: Any) -> None:
    """Defer ``resolve_arg_template_fields(fields, observed, parts)`` to the row's fill."""

    fields[DEFERRED] = (observed, parts)


def resolve_deferred(fields: dict[str, Any]) -> None:
    """Resolve and drop a thin row's deferred argument template, if it has one."""

    deferred = fields.pop(DEFERRED, None)
    if deferred is not None:
        resolve_arg_template_fields(fields, *deferred)
