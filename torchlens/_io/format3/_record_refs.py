"""Resolve the closed record-reference and pending-value grammar."""

from __future__ import annotations

from typing import Any

from ..format_contract import BlobRef
from .decode import CELL_CODE, PendingRecord, build_component, cell_error
from .restore import settle_mapping, settle_tuple
from .values import RowRef


class _RecordResolver:
    """Settle references against one artifact's allocated row identities."""

    objects: dict[str, list[Any]]
    _deferred_trace_refs: int

    def _settle(self, value: Any, at: str) -> Any:
        """Replace the references inside a decoded value with the built objects."""

        # Essential complexity: one branch per decoded carrier; the carrier set is closed.
        if isinstance(value, RowRef):
            return self._lookup(value, at)
        if isinstance(value, PendingRecord):
            before = self._deferred_trace_refs
            fields = {k: self._settle(v, f"{at}.{k}") for k, v in value.fields.items()}
            if self._deferred_trace_refs > before:
                return PendingRecord(value.cls, fields, value.at)
            return build_component(value.cls, fields, at=at)
        if isinstance(value, list):
            return [self._settle(item, f"{at}[{i}]") for i, item in enumerate(value)]
        if isinstance(value, BlobRef):
            return value
        if isinstance(value, tuple):
            items = [self._settle(item, f"{at}[{i}]") for i, item in enumerate(value)]
            return settle_tuple(value, items, at)
        if isinstance(value, (set, frozenset)):
            return type(value)(self._settle(item, at) for item in value)
        if isinstance(value, dict):
            mapping = {key: self._settle(item, f"{at}.{key}") for key, item in value.items()}
            return settle_mapping(value, mapping)
        return value

    def _lookup(self, reference: RowRef, at: str) -> Any:
        """Return the object a row reference names, refusing a dangling one."""

        rows = self.objects.get(reference.record)
        if reference.record == "trace" and not rows and reference.row == 0:
            self._deferred_trace_refs += 1
            return reference
        if rows is None:
            raise cell_error(
                at, f"references the {reference.record} table, which is absent", code=CELL_CODE
            )
        if not 0 <= reference.row < len(rows):
            raise cell_error(
                at, f"row {reference.row} is outside the {reference.record} table", code=CELL_CODE
            )
        return rows[reference.row]
