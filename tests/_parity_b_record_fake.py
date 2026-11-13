"""A fake non-torch backend whose ``tl.record`` runs through the neutral event interface.

The fake producer stands in for a real backend's record producer (the jax and
tinygrad producers live in their backends). Each pass it runs the torch model
under a save-everything torch recording, then replays that pass to the sink as
:class:`~torchlens.recording._backend_events.BackendOpEvent` metadata, handing a
value over only when the sink requests it. A forward that raised replays the
events it reached and then raises the model's own exception, as a backend whose
forward failed part-way would.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
from torch import nn

import torchlens as tl
from torchlens.errors import TraceCaptureError
from torchlens.recording._backend_events import BackendOpEvent, PayloadRequest, RecordSink

FAKE_BACKEND = "fake_record_backend"


class FakeCodec:
    """Payload codec that counts what it copies."""

    def __init__(self) -> None:
        """Start both counters at zero."""

        self.ram_copies = 0
        self.disk_copies = 0

    def ram_copy(self, value: Any) -> Any:
        """Return an independent copy for the in-memory store."""

        self.ram_copies += 1
        return value.detach().clone()

    def to_disk_tensor(self, value: Any) -> torch.Tensor:
        """Return a CPU tensor for the disk bundle."""

        self.disk_copies += 1
        return value.detach().cpu().clone()


class FakeProducer:
    """Record producer replaying a torch pass as backend events.

    Parameters
    ----------
    lazy:
        Deliver every requested value at the end of the pass, newest first, as a
        lazy framework that realizes values only on demand would.
    """

    backend = FAKE_BACKEND
    supported_options: frozenset[str] = frozenset()

    def __init__(self, *, lazy: bool = False) -> None:
        """Set up the codec and the counters the tests read."""

        self.codec = FakeCodec()
        self.lazy = lazy
        self.prepare_calls = 0
        self.passes = 0
        self.events_emitted = 0

    def prepare(self, model: nn.Module) -> nn.Module:
        """Count model discovery; the module itself is the prepared program."""

        self.prepare_calls += 1
        return model

    def run_pass(
        self, prepared: nn.Module, input_args: Any, input_kwargs: Any, sink: RecordSink
    ) -> Any:
        """Replay one pass to the sink and return the model output."""

        self.passes += 1
        records, error, output = _run_torch_pass(prepared, input_args, input_kwargs)
        waiting: list[tuple[PayloadRequest, Any]] = []
        for record in records:
            decision = sink.emit(_event_from_ctx(record.ctx))
            self.events_emitted += 1
            if decision.request is not None:
                waiting.append((decision.request, record.ram_payload))
                if not self.lazy:
                    sink.deliver(*waiting.pop())
            if decision.stop:
                break
        for request, value in reversed(waiting):
            sink.deliver(request, value)
        if error is not None and not sink.stop:
            raise error
        return output


def _run_torch_pass(model: nn.Module, input_args: Any, input_kwargs: Any) -> tuple[Any, Any, Any]:
    """Return the save-everything records of one torch pass, its error and its output."""

    try:
        recording = tl.record(
            model,
            input_args,
            input_kwargs=dict(input_kwargs) or None,
            save=lambda ctx: True,
            recording=tl.options.RecordingOptions(on_forward_error="attach_partial"),
        )
    except TraceCaptureError as failed:
        partial = failed.partial_recording  # type: ignore[attr-defined]
        return list(partial.records), failed.__cause__, None
    return list(recording.records), None, recording.output


def _event_from_ctx(ctx: Any) -> BackendOpEvent:
    """Turn a torch record context into the metadata-only backend event."""

    return BackendOpEvent(
        raw_label=ctx.raw_label or ctx.label,
        layer_type=ctx.layer_type,
        type_index=ctx.type_index,
        raw_index=ctx.raw_index,
        parent_labels=tuple(ctx.parent_labels),
        func_name=ctx.func_name,
        shape=ctx.shape,
        dtype=ctx.dtype,
        device=ctx.tensor_device,
        requires_grad=None,
        output_index=ctx.output_index,
        is_bottom_level_func=bool(ctx.is_bottom_level_func),
        module_stack=tuple(ctx.module_stack),
        func_call_id=ctx.func_call_id,
        kind=str(ctx.kind),
    )


@contextmanager
def fake_backend(producer: Any) -> Iterator[Any]:
    """Register the fake backend binding ``producer`` as its fastlog surface."""

    torch_spec = tl.backends.spec("torch")
    implementations = dict(torch_spec.capability_implementations or {})
    implementations["fastlog"] = lambda: producer
    spec = dataclasses.replace(
        torch_spec,
        name=FAKE_BACKEND,
        aliases=(),
        can_handle=lambda model, args, kwargs: False,
        capability_implementations=implementations,
        protocol=None,
    )
    tl.backends.register(spec, replace=True)
    try:
        yield producer
    finally:
        tl.backends.unregister(FAKE_BACKEND)


def payload_rows(recording: Any) -> list[tuple[str, str | None, Any]]:
    """Return ``(label, raw_label, payload)`` for every record, in order."""

    return [(r.ctx.label, r.ctx.raw_label, r.ram_payload) for r in recording.records]


def assert_same_records(backend_recording: Any, torch_recording: Any) -> None:
    """Assert the two recordings hold the same keys, contexts and values."""

    got = payload_rows(backend_recording)
    want = payload_rows(torch_recording)
    assert [row[:2] for row in got] == [row[:2] for row in want]
    for (label, _, value), (_, _, expected) in zip(got, want, strict=True):
        if expected is None:
            assert value is None, label
        else:
            assert torch.equal(value, expected), label
    assert set(backend_recording.by_label) == set(torch_recording.by_label)
    assert set(backend_recording.by_address) == set(torch_recording.by_address)
    assert dict(backend_recording.by_pass) == dict(torch_recording.by_pass)
    for got_record, want_record in zip(
        backend_recording.records, torch_recording.records, strict=True
    ):
        for name in ("kind", "layer_type", "type_index", "pass_number", "address"):
            assert getattr(got_record.ctx, name) == getattr(want_record.ctx, name), name
