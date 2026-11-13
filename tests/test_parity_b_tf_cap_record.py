"""TensorFlow ``tl.record`` producer: values and keys equal ``tl.trace``; a stop unwinds the forward.

The fixture applies the registry hunk pp-merge lands: TensorFlow's ``fastlog``
capability bound to :func:`torchlens.backends.tf._record.tf_record_producer`.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

import torchlens as tl  # noqa: E402
from torchlens.backends import registry  # noqa: E402
from torchlens.backends.tf import _record  # noqa: E402

pytestmark = [pytest.mark.tf_backend, pytest.mark.rare]

_PRODUCERS: list[_record.TFRecordProducer] = []
_X = np.random.default_rng(0).standard_normal((2, 8)).astype(np.float32)


def _bound_producer() -> _record.TFRecordProducer:
    producer = _record.tf_record_producer()
    _PRODUCERS.append(producer)
    return producer


@pytest.fixture(autouse=True)
def _tf_fastlog_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    """The registry hunk pp-merge applies: TensorFlow's fastlog bound to the record producer."""

    spec = registry.get_backend_spec("tf")
    replaced = dataclasses.replace(
        spec,
        capabilities=dataclasses.replace(spec.capabilities, fastlog=True),
        capability_implementations={
            **(spec.capability_implementations or {}),
            "fastlog": _bound_producer,
        },
    )
    monkeypatch.setitem(registry._REGISTRY, "tf", replaced)
    _PRODUCERS.clear()


def _chain(x: Any) -> Any:
    """Three ``relu((h + 1) * 0.5)`` blocks."""

    h = x
    for _ in range(3):
        h = tf.nn.relu(tf.multiply(tf.add(h, 1.0), 0.5))
    return h


class _Mlp(tf.keras.Model):
    """Two dense layers with a tanh between them."""

    def __init__(self) -> None:
        super().__init__(name="mlp")
        self.first = tf.keras.layers.Dense(6, name="first")
        self.second = tf.keras.layers.Dense(3, name="second")

    def call(self, x: Any) -> Any:
        return self.second(tf.tanh(self.first(x)))


def _ops(ctx: Any) -> bool:
    return bool(ctx.kind == "op")


def _relu_ops(ctx: Any) -> bool:
    return bool(ctx.kind == "op" and ctx.func_name == "Relu")


def _trace_payloads(model: Any, x: Any) -> dict[str, Any]:
    """Map each raw label of a full ``tl.trace`` capture to its saved payload."""

    trace = tl.trace(model, x, backend="tf")
    return {
        str(op._label_raw): op.out
        for op in trace._layer_list
        if getattr(op, "_label_raw", None) is not None
    }


def _assert_records_match_trace(recording: Any, payloads: dict[str, Any]) -> None:
    assert recording.records
    for record in recording.records:
        label = record.ctx.raw_label
        assert label in payloads, f"record {label!r} has no tl.trace op"
        assert record.ram_payload is not None, label
        np.testing.assert_array_equal(np.asarray(record.ram_payload), np.asarray(payloads[label]))


@pytest.mark.parametrize("save", [_ops, _relu_ops], ids=["all_ops", "relu_ops"])
def test_record_values_and_keys_equal_trace(save: Any) -> None:
    """Every recorded value equals the ``tl.trace`` payload with the same raw label."""

    x = tf.constant(_X)
    recording = tl.record(_chain, x, save=save, backend="tf")

    _assert_records_match_trace(recording, _trace_payloads(_chain, x))
    if save is _relu_ops:
        assert [r.ctx.func_name for r in recording.records] == ["Relu"] * 3
    np.testing.assert_array_equal(np.asarray(recording.output), _chain(x).numpy())


def test_every_trace_op_is_recorded() -> None:
    """Saving every op records one value per ``tl.trace`` op that is not an input."""

    x = tf.constant(_X)
    recording = tl.record(_chain, x, save=_ops, backend="tf")
    trace = tl.trace(_chain, x, backend="tf")
    trace_ops = [op for op in trace._layer_list if op.func_name != "input"]

    assert [r.ctx.raw_label for r in recording.records] == [str(op._label_raw) for op in trace_ops]


def test_keras_model_records_module_frames_and_values() -> None:
    """A Keras model's ops carry their layer frames and match ``tl.trace`` values."""

    model = _Mlp()
    x = tf.constant(_X)
    model(x)
    recording = tl.record(model, x, save=_ops, backend="tf")

    _assert_records_match_trace(recording, _trace_payloads(model, x))
    frames = [f for r in recording.records for f in r.ctx.module_stack]
    assert frames
    assert {f.module_type for f in frames} >= {"Dense"}
    assert _PRODUCERS[-1].passes == 1


def test_stop_after_unwinds_the_forward() -> None:
    """``stop_after`` ends the pass at the first Relu: no later op runs."""

    x = tf.constant(_X)
    full = tl.record(_chain, x, save=_ops, backend="tf")
    full_ops = _PRODUCERS[-1].ops_run

    def stop(ctx: Any) -> bool:
        return bool(ctx.kind == "op" and ctx.func_name == "Relu")

    halted = tl.record(_chain, x, save=_ops, stop_after=stop, backend="tf")

    assert halted.records[-1].ctx.func_name == "Relu"
    assert _PRODUCERS[-1].ops_run < full_ops
    prefix = [r.ctx.raw_label for r in full.records[: len(halted.records)]]
    assert [r.ctx.raw_label for r in halted.records] == prefix
    _assert_records_match_trace(halted, _trace_payloads(_chain, x))


def test_disk_storage_writes_selected_values(tmp_path: Path) -> None:
    """``storage=`` routes each selected value through the TensorFlow codec to the bundle."""

    x = tf.constant(_X)
    storage = tl.options.StorageOptions(path=tmp_path / "tf_bundle", retain_in_memory=True)
    recording = tl.record(_chain, x, save=_relu_ops, storage=storage, backend="tf")

    assert len(recording.records) == 3
    assert all(r.metadata.get("blob_id") is not None for r in recording.records)
    _assert_records_match_trace(recording, _trace_payloads(_chain, x))


def test_codec_round_trips_bfloat16() -> None:
    """The disk codec keeps bfloat16 values exactly."""

    value = tf.constant([1.5, -2.25, 3.0], dtype=tf.bfloat16)
    tensor = _record.TFPayloadCodec().to_disk_tensor(value)

    assert str(tensor.dtype) == "torch.bfloat16"
    np.testing.assert_array_equal(tensor.float().numpy(), tf.cast(value, tf.float32).numpy())
