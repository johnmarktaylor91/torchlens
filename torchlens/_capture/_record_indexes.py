"""The public lookup indexes of a Recording, built core by core.

``RecordingProjector.project`` (``projectors``) projects each sealed core's
retained activation records and hands them here. The indexes read only each
record's context, so this module sits outside the capture import cycle.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from .._ir.predicate import _distinct_label_index_keys

if TYPE_CHECKING:
    from ..recording.types import ActivationRecord

__all__ = ("_RecordIndexes",)


class _RecordIndexes:
    """The Recording's public lookup indexes, built core by core."""

    def __init__(self) -> None:
        self.records: list[ActivationRecord] = []
        self.by_pass: dict[int, list[int]] = {}
        self.by_label: dict[str, list[tuple[int, int]]] = {}
        self.by_address: dict[str, list[int]] = {}

    def add_records(self, records: Iterable[ActivationRecord]) -> None:
        """Append one core's records and index each by pass, label and address.

        Parameters
        ----------
        records
            The core's retained activation records, in capture order. The
            indexes store each record's position in ``self.records`` (which
            spans every core added so far): ``by_pass`` and ``by_address``
            map to positions, ``by_label`` to ``(pass number, position)``
            under the record's label and, when it differs, its raw label.
        """

        for record in records:
            index = len(self.records)
            self.records.append(record)
            self.by_pass.setdefault(record.ctx.pass_number, []).append(index)
            for label_key in _distinct_label_index_keys(record.ctx.label, record.ctx.raw_label):
                self.by_label.setdefault(label_key, []).append((record.ctx.pass_number, index))
            if record.ctx.address is not None:
                self.by_address.setdefault(record.ctx.address, []).append(index)
