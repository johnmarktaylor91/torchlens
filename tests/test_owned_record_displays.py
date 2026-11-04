"""Owned record displays preserve tuple semantics and portable wire encoding."""

from __future__ import annotations

import pickle
from dataclasses import dataclass

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._data_classes._accessor_base import Accessor
from torchlens._data_classes.func_call_location import FuncCallLocation
from torchlens._extraction.trace_safetensors import _encode
from torchlens._io._portability_preflight import (
    canonical_metadata_bytes,
    sanitize_annotation_tensors,
)
from torchlens._io.scrub import _rebuild_tuple_value, scrub_for_save
from torchlens._utils._display_bounded import RecordTuple


@pytest.fixture
def trace():
    captured = tl.trace(
        nn.Sequential(nn.Linear(4, 4), nn.Tanh()),
        torch.ones(2, 4),
        capture=tl.options.CaptureOptions(aten_ops=True),
    )
    yield captured
    captured.cleanup()


def test_owned_tuples_keep_value_semantics(trace):
    for rows in (
        trace.aten_ops,
        trace.op_events,
        trace.backward_events,
        trace.sibling_events,
        trace.transforms,
    ):
        assert isinstance(rows, RecordTuple)
        assert rows == tuple(rows)
        assert tuple(rows) == rows
        assert rows[:] == tuple(rows)
        assert type(rows[:]) is tuple
        assert len(repr(rows)) <= 100
        assert len(str(rows).splitlines()) <= 8
    labels = RecordTuple(("one", "two"), "Labels", "labels")
    assert hash(labels) == hash(tuple(labels))
    restored = pickle.loads(pickle.dumps(labels))
    assert type(restored) is RecordTuple
    assert restored == labels and repr(restored) == repr(labels)


def test_owned_tuple_wire_bytes_match_plain_tuple():
    plain = (1, "two", (3, 4))
    owned = RecordTuple(plain, "Rows", "rows")
    rebuilt = _rebuild_tuple_value(owned, owned)
    assert type(rebuilt) is tuple
    assert canonical_metadata_bytes(rebuilt) == canonical_metadata_bytes(plain)
    assert canonical_metadata_bytes(owned) == canonical_metadata_bytes(plain)
    sanitized = sanitize_annotation_tensors({"rows": owned})
    assert type(sanitized["rows"]) is tuple
    assert _encode(owned) == _encode(plain)


def test_save_scrubs_owned_tuple_as_before(trace):
    trace.annotations["rows"] = (1, "two", (3, 4))
    plain, _, _ = scrub_for_save(trace)
    trace.annotations["rows"] = RecordTuple(trace.annotations["rows"], "Rows", "rows")
    bounded, _, _ = scrub_for_save(trace)
    assert type(bounded["annotations"]["rows"]) is tuple
    assert canonical_metadata_bytes(plain) == canonical_metadata_bytes(bounded)


def test_trace_display_never_derives_health_or_mutates(trace, monkeypatch):
    from torchlens.report import _health

    def refuse(*args, **kwargs):
        pytest.fail("display attempted to derive health")

    monkeypatch.setattr(_health, "_derive", refuse)
    before = dict(trace.annotations)
    assert "NaN/Inf not checked" in repr(trace)
    text = str(trace)
    assert text.splitlines()[0] == repr(trace)
    assert len(text.splitlines()) <= 13
    html = trace._repr_html_()
    assert len(html) < 20_000
    assert ".summary()" in html and "<table>" in html
    assert trace.annotations == before


def test_accessor_renders_only_five_members():
    calls = []

    @dataclass
    class Member:
        index: int

        def __repr__(self):
            calls.append(self.index)
            return f"Member {self.index}"

    accessor = Accessor({i: Member(i) for i in range(20)})
    assert repr(accessor) == "Accessor: 20 items"
    assert not calls
    text = str(accessor)
    assert calls == list(range(5))
    assert "... and 15 more" in text


def test_frame_display_does_not_load_source(monkeypatch):
    frame = FuncCallLocation(file="missing.py", line_number=12, func_name="forward")
    monkeypatch.setattr(frame, "_ensure_source_loaded", lambda: pytest.fail("source read"))
    assert repr(frame) == "FuncCallLocation: missing.py:12 in forward"
    assert "  function  forward" in str(frame)
    frame.file = "/very/long/" * 20 + "model/forward.py"
    assert len(repr(frame)) <= 100
    assert "model/forward.py:12" in repr(frame)


def test_aten_records_and_event_stream_are_bounded(trace):
    assert trace.aten_ops
    row = trace.aten_ops[0]
    assert repr(row).startswith(f"AtenOp {row.label}:")
    assert "_display_owner_keys" not in row.__getstate__()
    assert len(repr(row)) <= 100
    assert "  schema" in str(row)
    assert len(repr(row.output_tensor_facts[0])) <= 100
    assert len(repr(trace.event_stream)) <= 100
    assert len(repr(trace.event_stream.live_index)) <= 100
    assert all(len(repr(event)) <= 100 for event in trace.op_events)


def test_trace_edit_honesty_survives_default_state(trace):
    trace._has_direct_writes = True
    assert "(edited)" in repr(trace)
    assert "pristine" not in repr(trace).lower()
    assert len(repr(trace)) <= 100


def test_explicit_health_check_is_reused_without_a_scan(trace, monkeypatch):
    from torchlens.report import _health

    _ = trace.health_facts
    before = dict(trace.annotations)
    monkeypatch.setattr(_health, "_derive", lambda *args: pytest.fail("display derived health"))
    assert "no NaN/Inf (exact)" in repr(trace)
    assert trace.annotations == before


def test_trace_ops_display_uses_the_counted_view(trace):
    from torchlens._utils._display_collections import member_repr

    shown = list(trace.ops)[:5]
    lines = str(trace.ops).splitlines()
    assert lines[1 : len(shown) + 1] == ["  " + member_repr(op) for op in shown]
    assert not any("input_1" in line for line in lines[1:])
    remainder = len(trace.ops) - len(shown)
    if remainder:
        assert f"  ... and {remainder} more" in lines
    else:
        assert not any("... and" in line for line in lines)
