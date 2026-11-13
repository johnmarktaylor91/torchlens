"""``tl.record`` on a backend that binds a record producer: values, keys and storage.

A fake producer replays a torch pass as backend events (``_parity_b_record_fake``).
The recording it yields must hold what the torch recorder holds for the same
selection, with values equal to ``tl.trace``'s, copying only selected values.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import pytest
import torch
from _parity_b_record_fake import (
    FAKE_BACKEND,
    FakeProducer,
    assert_same_records,
    fake_backend,
)
from torch import nn

import torchlens as tl
from torchlens.recording import CaptureSpec, RecordContext

pytestmark = [pytest.mark.backend_parity]


def _model() -> nn.Module:
    """Return a small nested model with linear, relu and add ops."""

    torch.manual_seed(0)
    return nn.Sequential(
        OrderedDict(
            stem=nn.Linear(4, 4),
            block=nn.Sequential(OrderedDict(fc=nn.Linear(4, 4), act=nn.ReLU())),
            head=nn.Linear(4, 2),
        )
    ).eval()


def _linear_ops(ctx: RecordContext) -> bool:
    """Select the linear ops."""

    return ctx.kind == "op" and ctx.layer_type == "linear"


_SELECTIONS = {
    "linear": _linear_ops,
    "relu_selector": tl.func("relu"),
    "all_ops": lambda ctx: ctx.kind == "op",
}


@pytest.mark.parametrize("selection", sorted(_SELECTIONS))
def test_record_values_equal_trace_and_torch_record(selection: str) -> None:
    """Same selection: same records and keys as torch, values equal to tl.trace."""

    model, x = _model(), torch.randn(3, 4)
    save = _SELECTIONS[selection]
    with fake_backend(FakeProducer()) as producer:
        recording = tl.record(model, x, save=save, backend=FAKE_BACKEND)
    torch_recording = tl.record(model, x, save=save)
    trace = tl.trace(model, x)

    assert recording.records, selection
    assert_same_records(recording, torch_recording)
    for layer_type in {r.ctx.layer_type for r in recording.records}:
        saved = [r.ram_payload for r in recording.records if r.ctx.layer_type == layer_type]
        traced = [op.out for op in trace.ops if op.type == layer_type]
        assert len(saved) == len(traced), layer_type
        for value, expected in zip(saved, traced, strict=True):
            assert torch.equal(value, expected), layer_type
    assert producer.codec.ram_copies == len(recording.records)
    assert producer.events_emitted > len(recording.records) or selection == "all_ops"
    assert recording.status == torch_recording.status == "complete"
    assert torch.equal(recording.output, torch_recording.output)


def test_metadata_only_selection_copies_no_value() -> None:
    """A metadata-only decision keeps the context and never asks for the value."""

    def metadata_only(ctx: RecordContext) -> CaptureSpec | bool:
        if ctx.kind == "op" and ctx.layer_type == "relu":
            return CaptureSpec(save_out=False, save_metadata=True)
        return False

    model, x = _model(), torch.randn(3, 4)
    with fake_backend(FakeProducer()) as producer:
        recording = tl.record(model, x, save=metadata_only, backend=FAKE_BACKEND)
    torch_recording = tl.record(model, x, save=metadata_only)

    assert_same_records(recording, torch_recording)
    assert [r.ram_payload for r in recording.records] == [None]
    assert producer.codec.ram_copies == 0


def test_lazy_out_of_order_delivery_keeps_event_order() -> None:
    """Values delivered late and newest first still land in event order."""

    model, x = _model(), torch.randn(3, 4)
    save = _SELECTIONS["all_ops"]
    with fake_backend(FakeProducer(lazy=True)):
        lazy = tl.record(model, x, save=save, backend=FAKE_BACKEND)
    assert_same_records(lazy, tl.record(model, x, save=save))


def test_disk_storage_writes_selected_values_through_the_codec(tmp_path: Path) -> None:
    """``storage=`` routes each selected value through the codec to the bundle."""

    model, x = _model(), torch.randn(3, 4)
    storage = tl.options.StorageOptions(path=tmp_path / "fake_bundle", retain_in_memory=True)
    with fake_backend(FakeProducer()) as producer:
        recording = tl.record(model, x, save=_linear_ops, storage=storage, backend=FAKE_BACKEND)
    torch_storage = tl.options.StorageOptions(path=tmp_path / "torch_bundle", retain_in_memory=True)
    torch_recording = tl.record(model, x, save=_linear_ops, storage=torch_storage)

    assert_same_records(recording, torch_recording)
    assert producer.codec.disk_copies == len(recording.records) == 3
    assert all(r.metadata.get("blob_id") is not None for r in recording.records)
    assert [sorted(r.metadata) for r in recording.records] == [
        sorted(r.metadata) for r in torch_recording.records
    ]


def test_default_torch_record_still_runs_the_torch_recorder() -> None:
    """With no backend, ``tl.record`` keeps the torch recorder and its journal."""

    model, x = _model(), torch.randn(3, 4)
    recording = tl.record(model, x, save=_linear_ops)
    assert len(recording.to_trace().ops) > 0
