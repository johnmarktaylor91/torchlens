"""``tl.record`` on a backend producer: the torch recorder's lifecycle, kept.

Repeated passes reuse model discovery but never a taken branch, halts stop the
producer, forward failures follow ``on_forward_error``, and predicate failures
follow ``on_predicate_error``, each as the torch recorder does for the same model.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from _parity_b_record_fake import (
    FAKE_BACKEND,
    FakeProducer,
    assert_same_records,
    fake_backend,
    payload_rows,
)
from torch import nn

import torchlens as tl
from torchlens.errors import PredicateError, RecorderStateError, TraceCaptureError
from torchlens.recording import RecordContext, Recorder
from torchlens.recording._backend_recorder import backend_recorder_factory

pytestmark = [pytest.mark.backend_parity]


class _Branchy(nn.Module):
    """Takes a relu or a tanh branch depending on the input's sign."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc(x)
        if float(x.sum()) > 0:
            return torch.relu(h) * 2
        return torch.tanh(h) + torch.sigmoid(h)


class _FailingAfterOps(nn.Module):
    """Runs three ops and then raises."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.add(x, 1)
        x = torch.relu(x)
        x = torch.mul(x, 3)
        raise RuntimeError("intentional forward boom")


def _ops(ctx: RecordContext) -> bool:
    return ctx.kind == "op"


def _inputs() -> list[torch.Tensor]:
    return [torch.ones(2, 4), -torch.ones(2, 4), torch.ones(2, 4)]


def test_repeated_passes_follow_each_pass_topology() -> None:
    """One recorder, three passes with different branches, discovery once."""

    model = _Branchy().eval()
    producer = FakeProducer()
    with backend_recorder_factory(producer)(model, save=_ops) as recorder:
        outputs = [recorder(x) for x in _inputs()]
    with Recorder(model, save=_ops) as torch_recorder:
        torch_outputs = [torch_recorder(x) for x in _inputs()]

    assert producer.prepare_calls == 1
    assert producer.passes == 3
    assert_same_records(recorder.recording, torch_recorder.recording)
    per_pass = {
        n: {r.ctx.layer_type for r in recorder.recording.records if r.ctx.pass_number == n}
        for n in (1, 2, 3)
    }
    assert "relu" in per_pass[1] and "tanh" not in per_pass[1]
    assert "tanh" in per_pass[2] and "relu" not in per_pass[2]
    assert per_pass[3] == per_pass[1]
    for got, want in zip(outputs, torch_outputs, strict=True):
        assert torch.equal(got, want)


def test_repeated_one_shot_records_never_reuse_a_branch() -> None:
    """Back-to-back ``tl.record`` calls each see their own input's branch."""

    model = _Branchy().eval()
    with fake_backend(FakeProducer()):
        recordings = [tl.record(model, x, save=_ops, backend=FAKE_BACKEND) for x in _inputs()]
    for recording, x in zip(recordings, _inputs(), strict=True):
        assert_same_records(recording, tl.record(model, x, save=_ops))


def test_halt_stops_the_producer_as_torch_stops() -> None:
    """``stop_after`` ends the pass after the matching op; later ops are not emitted."""

    model = _Branchy().eval()
    x = torch.ones(2, 4)

    def stop(ctx: RecordContext) -> bool:
        return ctx.kind == "op" and ctx.layer_type == "relu"

    with fake_backend(FakeProducer()) as producer:
        recording = tl.record(model, x, save=_ops, stop_after=stop, backend=FAKE_BACKEND)
    torch_recording = tl.record(model, x, save=_ops, stop_after=stop)

    assert_same_records(recording, torch_recording)
    assert recording.status == torch_recording.status
    assert recording.records[-1].ctx.layer_type == "relu"
    assert producer.events_emitted < len(tl.record(model, x, save=lambda ctx: True).records)


@pytest.mark.parametrize("mode", ["return_partial", "attach_partial"])
def test_forward_failure_partial_matches_torch(mode: str) -> None:
    """A producer whose forward raised yields torch's partial recording."""

    options = tl.options.RecordingOptions(on_forward_error=mode)
    recordings: dict[str, Any] = {}
    with fake_backend(FakeProducer()):
        for name, backend in (("fake", FAKE_BACKEND), ("torch", None)):
            kwargs = {"backend": backend} if backend else {}
            if mode == "return_partial":
                recordings[name] = tl.record(
                    _FailingAfterOps(), torch.tensor(1.0), save=_ops, recording=options, **kwargs
                )
                continue
            with pytest.raises(TraceCaptureError, match="intentional forward boom") as info:
                tl.record(
                    _FailingAfterOps(), torch.tensor(1.0), save=_ops, recording=options, **kwargs
                )
            assert type(info.value.__cause__) is RuntimeError
            recordings[name] = info.value.partial_recording  # type: ignore[attr-defined]

    fake, torch_partial = recordings["fake"], recordings["torch"]
    assert [row[:2] for row in payload_rows(fake)] == [
        row[:2] for row in payload_rows(torch_partial)
    ]
    for name in ("status", "is_failed", "num_committed_ops", "last_successful_op_label"):
        assert getattr(fake, name) == getattr(torch_partial, name), name
    assert "boom" in fake.error_traceback
    with pytest.raises(RuntimeError, match="failed partial Recording"):
        fake.to_trace()


def test_forward_failure_raise_mode_matches_torch() -> None:
    """The default disposition raises torch's error type, chained from the model's."""

    with (
        fake_backend(FakeProducer()),
        pytest.raises(TraceCaptureError, match="intentional forward boom") as info,
    ):
        tl.record(_FailingAfterOps(), torch.tensor(1.0), save=_ops, backend=FAKE_BACKEND)
    assert type(info.value.__cause__) is RuntimeError
    assert not hasattr(info.value, "partial_recording")


def test_failed_recorder_refuses_another_pass() -> None:
    """After a returned partial the recorder is failed, as the torch recorder is."""

    options = tl.options.RecordingOptions(on_forward_error="return_partial")
    build = backend_recorder_factory(FakeProducer())
    with build(_FailingAfterOps(), save=_ops, recording=options) as recorder:
        assert recorder(torch.tensor(1.0)) is None
        with pytest.raises(RecorderStateError, match="Recorder is in a failed state"):
            recorder(torch.tensor(1.0))
    assert recorder.recording.is_failed is True


def _boom_on_relu(ctx: RecordContext) -> bool:
    if ctx.kind == "op" and ctx.layer_type == "relu":
        raise ValueError("predicate boom")
    return ctx.kind == "op"


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("accumulate", PredicateError), ("fail-fast", TraceCaptureError)],
)
def test_predicate_errors_follow_on_predicate_error(mode: str, expected: type) -> None:
    """A raising save predicate surfaces as torch surfaces it, in both modes.

    Accumulate raises ``PredicateError`` at the end of the block; fail-fast
    raises from inside the pass, so the forward-failure door wraps it.
    """

    model, x = _Branchy().eval(), torch.ones(2, 4)
    options = tl.options.RecordingOptions(on_predicate_error=mode)
    errors = []
    with fake_backend(FakeProducer()):
        for kwargs in ({"backend": FAKE_BACKEND}, {}):
            with pytest.raises(expected, match="predicate") as info:
                tl.record(model, x, save=_boom_on_relu, recording=options, **kwargs)
            errors.append(info.value)
    assert type(errors[0]) is type(errors[1])
    assert type(errors[0].__cause__) is type(errors[1].__cause__)
    assert errors[0].fields.get("code") == errors[1].fields.get("code")


def test_backend_recording_refuses_to_trace() -> None:
    """A producer recording keeps no journal, so ``to_trace()`` refuses typed."""

    model = _Branchy().eval()
    with fake_backend(FakeProducer()):
        recording = tl.record(model, torch.ones(2, 4), save=_ops, backend=FAKE_BACKEND)
    with pytest.raises(RuntimeError) as info:
        recording.to_trace()
    assert info.value.fields["code"] == "recording_events_not_retained"  # type: ignore[attr-defined]
