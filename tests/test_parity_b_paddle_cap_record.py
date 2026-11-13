"""Paddle ``tl.record`` producer: values and keys equal ``tl.trace``; stops end the forward.

The fixture applies the registry hunk pp-merge lands: Paddle's ``fastlog`` capability
bound to :func:`torchlens.backends.paddle._record.paddle_record_producer`.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest

paddle = pytest.importorskip("paddle")

import numpy as np  # noqa: E402

import torchlens as tl  # noqa: E402
from torchlens.backends import registry  # noqa: E402
from torchlens.backends.paddle import _record  # noqa: E402

pytestmark = [pytest.mark.backend_paddle, pytest.mark.rare]

_PRODUCERS: list[_record.PaddleRecordProducer] = []


def _bound_producer() -> _record.PaddleRecordProducer:
    """Return a producer and remember it for counter checks."""

    producer = _record.paddle_record_producer()
    _PRODUCERS.append(producer)
    return producer


@pytest.fixture(autouse=True)
def _paddle_fastlog_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    """The registry hunk pp-merge applies: Paddle's fastlog bound to the record producer."""

    spec = registry.get_backend_spec("paddle")
    replaced = dataclasses.replace(
        spec,
        capabilities=dataclasses.replace(spec.capabilities, fastlog=True),
        capability_implementations={
            **(spec.capability_implementations or {}),
            "fastlog": _bound_producer,
        },
    )
    monkeypatch.setitem(registry._REGISTRY, "paddle", replaced)
    _PRODUCERS.clear()


def _chain(x: Any) -> Any:
    """Run a short chain of distinct Paddle ops."""

    h = paddle.multiply(x, x)
    h = paddle.nn.functional.relu(h - 0.25)
    h = paddle.tanh(h)
    h = paddle.add(h, x)
    return paddle.exp(paddle.nn.functional.relu(h) * 0.5)


def _mlp() -> Any:
    """Return a seeded two-layer Paddle MLP."""

    paddle.seed(0)
    return paddle.nn.Sequential(
        paddle.nn.Linear(6, 8), paddle.nn.ReLU(), paddle.nn.Linear(8, 3), paddle.nn.Tanh()
    )


def _x() -> Any:
    """Return a deterministic input tensor."""

    return paddle.arange(24, dtype="float32").reshape([4, 6]) / 12.0 - 0.5


def _ops(ctx: Any) -> bool:
    """Save every op."""

    return bool(ctx.kind == "op")


def _relu_ops(ctx: Any) -> bool:
    """Save the relu ops."""

    return bool(ctx.kind == "op" and "relu" in str(ctx.layer_type))


def _trace_payloads(model: Any, x: Any) -> dict[str, Any]:
    """Map each raw label of a full ``tl.trace`` capture to its saved payload."""

    trace = tl.trace(model, x, backend="paddle")
    return {
        str(op._label_raw): op.out
        for op in trace._layer_list
        if getattr(op, "_label_raw", None) is not None and op.out is not None
    }


def _assert_records_match_trace(recording: Any, payloads: dict[str, Any]) -> None:
    """Every record has a ``tl.trace`` op with the same raw label and equal value."""

    assert recording.records
    for record in recording.records:
        label = record.ctx.raw_label
        assert label in payloads, f"record {label!r} has no tl.trace op"
        assert record.ram_payload is not None, label
        np.testing.assert_array_equal(record.ram_payload.numpy(), payloads[label].numpy())


@pytest.mark.parametrize("save", [_ops, _relu_ops], ids=["all_ops", "relu_ops"])
def test_record_values_and_keys_equal_trace_on_a_chain(save: Any) -> None:
    """Every recorded value equals the ``tl.trace`` payload with the same raw label."""

    recording = tl.record(_chain, _x(), save=save, backend="paddle")

    _assert_records_match_trace(recording, _trace_payloads(_chain, _x()))
    if save is _relu_ops:
        assert len(recording.records) == 2
    np.testing.assert_array_equal(recording.output.numpy(), _chain(_x()).numpy())


def test_record_covers_every_trace_op_on_a_module_model() -> None:
    """A Layer root records one value per ``tl.trace`` op, with module frames."""

    model = _mlp()
    recording = tl.record(model, _x(), save=_ops, backend="paddle")
    trace = tl.trace(model, _x(), backend="paddle")
    traced = [
        str(op._label_raw)
        for op in trace._layer_list
        if op.layer_type not in ("input", "output") and op.out is not None
    ]

    _assert_records_match_trace(recording, _trace_payloads(model, _x()))
    assert [r.ctx.raw_label for r in recording.records] == traced
    assert _PRODUCERS[-1].ops_emitted == len(traced)
    assert any(r.ctx.module_stack for r in recording.records)


def test_input_events_carry_the_trace_input_label() -> None:
    """Input leaves are ``input`` events keyed as ``tl.trace`` keys them."""

    x = _x()
    recording = tl.record(_chain, x, save=lambda ctx: ctx.kind == "input", backend="paddle")

    assert [r.ctx.kind for r in recording.records] == ["input"]
    _assert_records_match_trace(recording, _trace_payloads(_chain, x))


def test_stop_after_ends_the_forward() -> None:
    """``stop_after`` ends the pass at the first tanh: no later op runs."""

    full = tl.record(_chain, _x(), save=_ops, backend="paddle")
    full_ops = _PRODUCERS[-1].ops_emitted

    def stop(ctx: Any) -> bool:
        return bool(ctx.kind == "op" and ctx.layer_type == "tanh")

    halted = tl.record(_chain, _x(), save=_ops, stop_after=stop, backend="paddle")

    assert halted.records[-1].ctx.layer_type == "tanh"
    assert _PRODUCERS[-1].ops_emitted < full_ops
    prefix = [r.ctx.raw_label for r in full.records[: len(halted.records)]]
    assert [r.ctx.raw_label for r in halted.records] == prefix
    _assert_records_match_trace(halted, _trace_payloads(_chain, _x()))


def test_repeated_passes_keep_labels_and_release_wrappers() -> None:
    """Two recordings label identically and leave Paddle unwrapped."""

    from torchlens.backends.paddle.wrappers import is_paddle_wrapped

    first = tl.record(_chain, _x(), save=_ops, backend="paddle")
    second = tl.record(_chain, _x(), save=_ops, backend="paddle")

    assert [r.ctx.raw_label for r in first.records] == [r.ctx.raw_label for r in second.records]
    assert not is_paddle_wrapped()


def test_disk_storage_writes_selected_values(tmp_path: Path) -> None:
    """``storage=`` routes each selected value through the Paddle codec to the bundle."""

    storage = tl.options.StorageOptions(path=tmp_path / "paddle_bundle", retain_in_memory=True)
    recording = tl.record(_chain, _x(), save=_relu_ops, storage=storage, backend="paddle")

    assert len(recording.records) == 2
    assert all(r.metadata.get("blob_id") is not None for r in recording.records)
    _assert_records_match_trace(recording, _trace_payloads(_chain, _x()))


def test_codec_ram_copy_is_independent() -> None:
    """The RAM copy does not follow later in-place writes to the live tensor."""

    value = paddle.to_tensor([1.0, 2.0, 3.0])
    copy = _record.PaddlePayloadCodec().ram_copy(value)
    value.scale_(2.0)

    np.testing.assert_array_equal(copy.numpy(), np.asarray([1.0, 2.0, 3.0], dtype=np.float32))


def test_codec_disk_tensor_matches_values() -> None:
    """The disk codec gives a CPU torch tensor with the same numbers and dtype."""

    value = paddle.to_tensor([1.5, -2.25, 3.0], dtype="float32")
    tensor = _record.PaddlePayloadCodec().to_disk_tensor(value)

    assert str(tensor.dtype) == "torch.float32"
    np.testing.assert_array_equal(tensor.numpy(), value.numpy())
