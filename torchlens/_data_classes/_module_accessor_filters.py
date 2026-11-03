"""Collection views of ``ModuleAccessor`` that read through the owning trace.

A mixin so the module record file stays within its size cap; it assumes the
``Accessor`` attributes (``_list``) of the class it is mixed into.
"""

from __future__ import annotations

from typing import Any


class ModuleAccessorFilters:
    """Filters and the composition card of a Module collection."""

    _list: list[Any]

    @property
    def uncalled(self) -> list[str]:
        """Addresses of registered modules the captured pass never called.

        The collection spelling of ``Trace.uncalled_modules``: an uncalled module
        has no ``Module`` record, so the answer is its addresses. Empty when no
        member still reaches its trace.
        """

        for module in self._list:
            trace = module._source_trace
            if trace is not None:
                return trace.uncalled_modules
        return []

    def _composition_note(self) -> str | None:
        """Composition breakdown: total calls across modules."""

        calls = sum(getattr(ml, "num_calls", 0) or 0 for ml in self._list)
        if calls <= len(self._list):
            return None
        return f"{calls} calls"
