"""Module-call scope of a render node with its own atomic-module call dropped."""

from __future__ import annotations

from typing import Any

from .._utils._record_group_table import record_member


def _enclosing_module_calls(node: Any) -> list[Any]:
    """Return a node's module calls without its own atomic-module call.

    An atomic (single-op) module's output renders in its parent's scope, so the
    innermost module call is dropped when the node is one.

    Parameters
    ----------
    node:
        Render node or record; ``None`` yields an empty scope.

    Returns
    -------
    list[Any]
        Enclosing module calls, outermost first.
    """

    modules = list(getattr(node, "modules", ()) or ())
    if record_member(node, "is_atomic_module", False) and modules:
        modules = modules[:-1]
    return modules
