"""The lean recording option (PROVISIONAL SPELLING ``lean=``).

A default ``Recording`` holds exactly what it held before the option existed
(golden produced on the pre-option commit by ``tests/_record_lean_facts.py``);
a lean one keeps the selected records, journals nothing else, says what it
omitted and refuses the reads that would answer from a partial journal.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from _record_lean_facts import SCENARIOS, default_facts, record_default, scenario_input

import torchlens as tl
from torchlens import errors
from torchlens.recording.types import LEAN_OMISSIONS, Recording

_GOLDEN = Path(__file__).parent / "data" / "record_lean_default_golden.json"
_NAMES = [name for name, *_ in SCENARIOS]


def _scenario(name: str) -> tuple[torch.nn.Module, torch.Tensor, str]:
    scenario = next(s for s in SCENARIOS if s[0] == name)
    return scenario[1](), scenario_input(scenario[2]), scenario[3]


def _record_lean(name: str) -> Recording:
    model, x, site = _scenario(name)
    with torch.no_grad():
        return tl.record(model, x, save=tl.module(site), lean=True)


def _events(rec: Recording):
    return object.__getattribute__(rec, "_capture_events")


@pytest.mark.parametrize("name", _NAMES)
def test_default_recording_holds_exactly_the_pre_lean_content(name: str) -> None:
    """A default recording's facts equal the golden taken before the option existed."""

    golden = json.loads(_GOLDEN.read_text())[name]
    assert default_facts(record_default(name)) == golden


@pytest.mark.parametrize("name", _NAMES)
def test_default_recording_is_not_lean(name: str) -> None:
    rec = record_default(name)
    assert rec.options.lean is False
    assert rec.lean_omissions == ()
    assert rec.raw_metadata()  # the full-journal reads answer on a default recording
    assert rec.op_events


@pytest.mark.parametrize("name", _NAMES)
def test_lean_keeps_the_selected_records_and_journals_nothing_else(name: str) -> None:
    """Lean kept records equal the default ones; the journal holds only kept events."""

    default = record_default(name)
    lean = _record_lean(name)
    assert lean.options.lean is True
    assert lean.lean_omissions == LEAN_OMISSIONS
    assert len(lean.records) == len(default.records) == 1
    for d, ln in zip(default.records, lean.records, strict=True):
        assert (d.ctx.label, d.ctx.raw_label, d.ctx.pass_number) == (
            ln.ctx.label,
            ln.ctx.raw_label,
            ln.ctx.pass_number,
        )
        assert torch.equal(d.ram_payload, ln.ram_payload)
    assert torch.equal(default.output, lean.output)
    assert lean.num_ops == default.num_ops
    assert len(lean) == len(default)
    kept = {e.label_raw for e in _events(lean).amended_op_records()}
    assert kept == {r.ctx.raw_label for r in lean.records}
    assert all(e.predicate_matched for e in _events(lean).amended_op_records())
    assert len(_events(lean).op_events) < len(_events(default).op_events)


@pytest.mark.parametrize("name", _NAMES)
def test_lean_records_based_reads_are_complete(name: str) -> None:
    lean = _record_lean(name)
    default = record_default(name)
    assert [r.ctx.label for r in lean.records_in_pass(0)] == [
        r.ctx.label for r in default.records_in_pass(0)
    ]
    assert isinstance(lean.summary(), str)
    pd = pytest.importorskip("pandas")
    frame = lean.to_pandas()
    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == len(default.to_pandas())


@pytest.mark.parametrize("read", ["event_stream", "op_events", "lazy_trace"])
def test_lean_refuses_the_property_reads_that_need_the_full_journal(read: str) -> None:
    lean = _record_lean("small_block2")
    with pytest.raises(errors.CaptureNotActiveError) as exc:
        getattr(lean, read)
    assert exc.value.fields["code"] == "recording_lean_omitted"
    assert f"Recording.{read}" in str(exc.value)
    assert "lean=True" in str(exc.value)


@pytest.mark.parametrize("read", ["raw_metadata", "to_trace"])
def test_lean_refuses_the_method_reads_that_need_the_full_journal(read: str) -> None:
    lean = _record_lean("conv_block1")
    with pytest.raises(errors.CaptureNotActiveError) as exc:
        getattr(lean, read)()
    assert exc.value.fields["code"] == "recording_lean_omitted"
    for omission in LEAN_OMISSIONS:
        assert omission in str(exc.value)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"capture": tl.options.CaptureOptions(backward_ready=True)},
        {"capture": tl.options.CaptureOptions(save_grads=True)},
        {"default_grad": True},
        {"narrate": True},
        {"on_forward_error": "return_partial"},
        {"on_forward_error": "attach_partial"},
    ],
    ids=lambda kw: next(iter(kw)) + "=" + str(next(iter(kw.values())))[:40],
)
def test_lean_refuses_options_that_read_the_dropped_events(kwargs: dict) -> None:
    model, x, site = _scenario("small_block2")
    with pytest.raises(errors.RecordingConfigError) as exc:
        tl.record(model, x, save=tl.module(site), lean=True, **kwargs)
    assert exc.value.fields["code"] == "record_lean_unsupported"
    assert "lean=True" in str(exc.value)


def test_lean_must_be_a_bool() -> None:
    model, x, site = _scenario("small_block2")
    with pytest.raises(errors.InvalidArgumentError) as exc:
        tl.record(model, x, save=tl.module(site), lean="yes")  # type: ignore[arg-type]
    assert exc.value.fields["code"] == "recording_option_type_invalid"
    assert exc.value.fields["argument"] == "lean"


def test_lean_marker_survives_replace() -> None:
    lean = _record_lean("small_head")
    copy = lean._replace(num_ops=lean.num_ops)
    assert copy.options.lean is True
    assert copy.lean_omissions == LEAN_OMISSIONS
    assert [r.ctx.label for r in copy.records] == [r.ctx.label for r in lean.records]
    with pytest.raises(errors.CaptureNotActiveError) as exc:
        copy.raw_metadata()
    assert exc.value.fields["code"] == "recording_lean_omitted"


def test_recorder_door_takes_the_lean_keyword() -> None:
    model, x, site = _scenario("small_block2")
    recorder = tl.recording.Recorder(model, save=tl.module(site), lean=True)
    with torch.no_grad(), recorder:
        recorder(x)
        recorder(x + 1)
    rec = recorder.recording
    assert rec.options.lean is True
    assert len(rec.records) == 2
    assert [len(list(rec.records_in_pass(i))) for i in (0, 1)] == [1, 1]
    assert all(e.predicate_matched for e in _events(rec).amended_op_records())


def test_recording_options_carry_the_lean_field() -> None:
    assert tl.options.RecordingOptions().lean is False
    assert tl.options.RecordingOptions(lean=True).lean is True
