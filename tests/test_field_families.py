"""The capture field table classifies every op-store column.

A column added to the op row store without a row in
``torchlens._trace_core.field_table.FIELD_TABLE`` fails here, and so does a
row whose family, source, slot or settle phase is not a declared value.
"""

from __future__ import annotations

import torchlens._trace_core.field_table as ft
from torchlens._data_classes.op import _OP_STORE_LAYOUT
from torchlens._postprocess._pipeline import PIPELINE


def test_every_store_column_has_one_row() -> None:
    """The table's keys are exactly the op store's columns."""
    assert ft.unclassified_fields(_OP_STORE_LAYOUT.names) == ()
    assert sorted(set(ft.FIELD_TABLE) - set(_OP_STORE_LAYOUT.names)) == []
    assert len(ft._ROWS) == len(ft.FIELD_TABLE), "a field has two rows"


def test_a_new_column_without_a_row_is_reported() -> None:
    """The lookup the gate uses names an unknown column."""
    names = (*_OP_STORE_LAYOUT.names, "column_added_without_a_row")
    assert ft.unclassified_fields(names) == ("column_added_without_a_row",)


def test_rows_use_declared_values() -> None:
    """Family, source, slots and settle phase come from the declared sets."""
    phases = {*ft.FIXED_PHASES, *(step.name for step in PIPELINE), ""}
    for row in ft.FIELD_TABLE.values():
        assert row.family in ft.FAMILIES, row
        assert row.source in ft.SOURCES, row
        assert set(row.slots) <= ft.SLOT_NAMES, row
        assert row.settled in phases, row


def test_source_and_slots_agree() -> None:
    """A call-time or derived field names its slots; untracked sources name none."""
    for row in ft.FIELD_TABLE.values():
        if row.source in ("call", "derived"):
            assert row.slots, row
        elif row.source != "graph":
            assert row.slots == (), row
        if row.source == "call":
            assert ft.TRACE_SLOT not in row.slots, row
            assert row.settled, row


def test_call_time_fields_cover_every_call_row() -> None:
    """The grouped call-time list holds each call field under each of its slots."""
    grouped = ft.call_time_fields()
    listed = {name for names in grouped.values() for name in names}
    assert listed == set(ft.fields_with_source("call"))
    order = [name for name, _ in ft.EVENT_SLOTS]
    assert list(grouped) == [slot for slot in order if slot in grouped]


def test_unstored_observations_feed_known_fields() -> None:
    """Each unstored observation sits in a declared slot and feeds real columns."""
    names = [obs.name for obs in ft.UNSTORED_OBSERVATIONS]
    assert len(names) == len(set(names))
    for obs in ft.UNSTORED_OBSERVATIONS:
        assert obs.slot in ft.SLOT_NAMES, obs
        assert obs.name not in ft.FIELD_TABLE, obs
        assert set(obs.feeds) <= set(ft.FIELD_TABLE), obs
