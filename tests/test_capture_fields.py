"""The capture field table classifies every op-store column.

A column added to the op row store without a row in
``torchlens._trace_core.capture_fields.FIELD_TABLE`` fails here, and so does a
row whose family, source, slot or settle phase is not a declared value.
"""

from __future__ import annotations

from torchlens._data_classes.op import _OP_STORE_LAYOUT
from torchlens._postprocess._pipeline import PIPELINE
from torchlens._trace_core import capture_fields as cf


def test_every_store_column_has_one_row() -> None:
    """The table's keys are exactly the op store's columns."""
    assert cf.unclassified_fields(_OP_STORE_LAYOUT.names) == ()
    assert sorted(set(cf.FIELD_TABLE) - set(_OP_STORE_LAYOUT.names)) == []
    assert len(cf._ROWS) == len(cf.FIELD_TABLE), "a field has two rows"


def test_a_new_column_without_a_row_is_reported() -> None:
    """The lookup the gate uses names an unknown column."""
    names = (*_OP_STORE_LAYOUT.names, "column_added_without_a_row")
    assert cf.unclassified_fields(names) == ("column_added_without_a_row",)


def test_rows_use_declared_values() -> None:
    """Family, source, slots and settle phase come from the declared sets."""
    phases = {*cf.FIXED_PHASES, *(step.name for step in PIPELINE), ""}
    for row in cf.FIELD_TABLE.values():
        assert row.family in cf.FAMILIES, row
        assert row.source in cf.SOURCES, row
        assert set(row.slots) <= cf.SLOT_NAMES, row
        assert row.settled in phases, row


def test_source_and_slots_agree() -> None:
    """A call-time or derived field names its slots; untracked sources name none."""
    for row in cf.FIELD_TABLE.values():
        if row.source in ("call", "derived"):
            assert row.slots, row
        elif row.source != "graph":
            assert row.slots == (), row
        if row.source == "call":
            assert cf.TRACE_SLOT not in row.slots, row
            assert row.settled, row


def test_call_time_fields_cover_every_call_row() -> None:
    """The grouped call-time list holds each call field under each of its slots."""
    grouped = cf.call_time_fields()
    listed = {name for names in grouped.values() for name in names}
    assert listed == set(cf.fields_with_source("call"))
    order = [name for name, _ in cf.EVENT_SLOTS]
    assert list(grouped) == [slot for slot in order if slot in grouped]


def test_unstored_observations_feed_known_fields() -> None:
    """Each unstored observation sits in a declared slot and feeds real columns."""
    names = [obs.name for obs in cf.UNSTORED_OBSERVATIONS]
    assert len(names) == len(set(names))
    for obs in cf.UNSTORED_OBSERVATIONS:
        assert obs.slot in cf.SLOT_NAMES, obs
        assert obs.name not in cf.FIELD_TABLE, obs
        assert set(obs.feeds) <= set(cf.FIELD_TABLE), obs
