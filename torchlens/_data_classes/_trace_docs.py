"""Bind the full ``Trace`` method docstrings kept in the ``_trace_docs_*`` tables.

``trace.py`` is capped at its recorded size, so the record keeps one-line summaries and
the reference text lives in four sibling tables. ``attach_trace_docs(Trace)`` runs once at
class definition and sets ``__doc__`` on the matching function, property, cached property
or static method, so ``help(Trace.draw)`` shows the full text.
"""

from __future__ import annotations

import functools
from typing import Any

from ._trace_docs_intervention import DOCS as _INTERVENTION
from ._trace_docs_output import DOCS as _OUTPUT
from ._trace_docs_stats import DOCS as _STATS
from ._trace_docs_validation import DOCS as _VALIDATION

__all__ = ["TRACE_DOCS", "attach_trace_docs"]

TRACE_DOCS: dict[str, str] = {**_STATS, **_OUTPUT, **_INTERVENTION, **_VALIDATION}


def _doc_target(attr: Any) -> Any:
    """Return the object whose ``__doc__`` ``help()`` reads for a class attribute."""

    if isinstance(attr, (staticmethod, classmethod)):
        return attr.__func__
    return attr


def attach_trace_docs(cls: type) -> None:
    """Set the full docstring on every ``cls`` attribute named in the tables."""

    for name, doc in TRACE_DOCS.items():
        attr = cls.__dict__.get(name)
        if attr is None:
            raise AttributeError(
                f"{cls.__name__} has no attribute {name!r} for its docstring table"
            )
        target = _doc_target(attr)
        if isinstance(target, functools.cached_property):
            target.__doc__ = doc
            target.func.__doc__ = doc
        elif isinstance(target, property):
            cls.__dict__[name].fget.__doc__ = doc
            setattr(cls, name, property(target.fget, target.fset, target.fdel, doc))
        else:
            target.__doc__ = doc
