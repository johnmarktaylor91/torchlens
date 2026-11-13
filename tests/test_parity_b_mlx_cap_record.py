"""MLX ``tl.record`` producer: values and keys equal ``tl.trace``; stops end the forward.

The fixture applies the registry hunk pp-merge lands: MLX's ``fastlog`` capability
bound to :func:`torchlens.backends.mlx._record.mlx_record_producer`.
"""

from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path
from typing import Any

import pytest

if sys.platform != "darwin" and os.environ.get("TORCHLENS_RUN_UNSTABLE_MLX_TESTS") != "1":
    pytest.skip(
        "MLX exec-level tests are skipped on non-Darwin platforms by default; "
        "set TORCHLENS_RUN_UNSTABLE_MLX_TESTS=1 to opt in.",
        allow_module_level=True,
    )

mlx = pytest.importorskip("mlx")
pytest.importorskip("torchlens.recording._backend_events")

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import numpy as np  # noqa: E402

import torchlens as tl  # noqa: E402
from torchlens.backends import registry  # noqa: E402
from torchlens.backends.mlx import _record  # noqa: E402

pytestmark = [pytest.mark.backend_mlx, pytest.mark.rare]

_PRODUCERS: list[_record.MLXRecordProducer] = []
_WEIGHT = mx.array(np.linspace(-1.0, 1.0, 36, dtype=np.float32).reshape(6, 6))


def _bound_producer() -> _record.MLXRecordProducer:
    """Return a producer and remember it for counter checks."""

    producer = _record.mlx_record_producer()
    _PRODUCERS.append(producer)
    return producer


@pytest.fixture(autouse=True)
def _mlx_fastlog_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    """The registry hunk pp-merge applies: MLX's fastlog bound to the record producer."""

    spec = registry.get_backend_spec("mlx")
    replaced = dataclasses.replace(
        spec,
        capabilities=dataclasses.replace(spec.capabilities, fastlog=True),
        capability_implementations={
            **(spec.capability_implementations or {}),
            "fastlog": _bound_producer,
        },
    )
    monkeypatch.setitem(registry._REGISTRY, "mlx", replaced)
    _PRODUCERS.clear()


def _chain(x: Any) -> Any:
    """Run a short chain of distinct MLX ops, one of them on a closure constant."""

    h = mx.multiply(x, x)
    h = mx.maximum(h - 0.25, 0.0)
    h = mx.tanh(mx.matmul(h, _WEIGHT))
    h = mx.add(h, x)
    return mx.exp(mx.maximum(h, 0.0) * 0.5)


class _MLP(nn.Module):
    """A two-layer MLX MLP."""

    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.Linear(6, 8)
        self.l2 = nn.Linear(8, 3)

    def __call__(self, x: Any) -> Any:
        return mx.tanh(self.l2(nn.relu(self.l1(x))))


def _mlp() -> _MLP:
    """Return a seeded two-layer MLX MLP."""

    mx.random.seed(0)
    model = _MLP()
    mx.eval(model.parameters())
    return model


def _x() -> Any:
    """Return a deterministic input array."""

    return mx.arange(24, dtype=mx.float32).reshape(4, 6) / 12.0 - 0.5


def _ops(ctx: Any) -> bool:
    """Save every op."""

    return bool(ctx.kind == "op")


def _maximum_ops(ctx: Any) -> bool:
    """Save the maximum ops."""

    return bool(ctx.kind == "op" and "maximum" in str(ctx.layer_type))


def _trace_payloads(model: Any, x: Any) -> dict[str, Any]:
    """Map each raw label of a full ``tl.trace`` capture to its saved payload."""

    trace = tl.trace(model, x, backend="mlx")
    return {
        str(op._label_raw): op.out
        for op in trace._layer_list
        if getattr(op, "_label_raw", None) is not None and op.out is not None
    }


def _assert_records_match_trace(recording: Any, payloads: dict[str, Any]) -> None:
    """Every record has a ``tl.trace`` op with the same raw label and an equal value."""

    assert recording.records
    for record in recording.records:
        label = record.ctx.raw_label
        assert label in payloads, f"record {label!r} has no tl.trace op"
        assert record.ram_payload is not None, label
        assert record.ram_payload.dtype == payloads[label].dtype, label
        assert bool(mx.array_equal(record.ram_payload, payloads[label], equal_nan=True)), label


@pytest.mark.parametrize("save", [_ops, _maximum_ops], ids=["all_ops", "maximum_ops"])
def test_record_values_and_keys_equal_trace_on_a_chain(save: Any) -> None:
    """Every recorded value equals the ``tl.trace`` payload with the same raw label."""

    recording = tl.record(_chain, _x(), save=save, backend="mlx")

    _assert_records_match_trace(recording, _trace_payloads(_chain, _x()))
    if save is _maximum_ops:
        assert len(recording.records) == 2
    else:
        assert any(r.ctx.layer_type == "constant" for r in recording.records)
    assert _PRODUCERS[-1].values_delivered == len(recording.records)
    assert bool(mx.array_equal(recording.output, _chain(_x())))


def test_record_covers_every_trace_op_on_a_module_model() -> None:
    """A module root records one value per ``tl.trace`` op, with module frames."""

    model = _mlp()
    recording = tl.record(model, _x(), save=_ops, backend="mlx")
    trace = tl.trace(model, _x(), backend="mlx")
    traced = [
        str(op._label_raw)
        for op in trace._layer_list
        if op.type not in ("input", "output") and op.out is not None
    ]

    _assert_records_match_trace(recording, _trace_payloads(model, _x()))
    assert [r.ctx.raw_label for r in recording.records] == traced
    assert _PRODUCERS[-1].ops_emitted == len(traced)
    assert any(r.ctx.module_stack for r in recording.records)


def test_unselected_values_are_never_delivered() -> None:
    """Only requested values reach the sink; the rest stay unrealized for the recording."""

    model = _mlp()
    recording = tl.record(
        model, _x(), save=lambda ctx: "relu" in str(ctx.layer_type), backend="mlx"
    )

    producer = _PRODUCERS[-1]
    assert len(recording.records) == 1
    assert producer.values_delivered == 1
    assert producer.ops_emitted > 1


def test_input_events_carry_the_trace_input_label() -> None:
    """Input leaves are ``input`` events keyed as ``tl.trace`` keys them."""

    x = _x()
    recording = tl.record(_chain, x, save=lambda ctx: ctx.kind == "input", backend="mlx")

    assert [r.ctx.kind for r in recording.records] == ["input"]
    _assert_records_match_trace(recording, _trace_payloads(_chain, x))


def test_stop_after_ends_the_forward() -> None:
    """``stop_after`` ends the pass at the first tanh: no later op runs."""

    full = tl.record(_chain, _x(), save=_ops, backend="mlx")
    full_ops = _PRODUCERS[-1].ops_emitted

    def stop(ctx: Any) -> bool:
        return bool(ctx.kind == "op" and ctx.layer_type == "tanh")

    halted = tl.record(_chain, _x(), save=_ops, stop_after=stop, backend="mlx")

    assert halted.records[-1].ctx.layer_type == "tanh"
    assert _PRODUCERS[-1].ops_emitted < full_ops
    prefix = [r.ctx.raw_label for r in full.records[: len(halted.records)]]
    assert [r.ctx.raw_label for r in halted.records] == prefix
    _assert_records_match_trace(halted, _trace_payloads(_chain, _x()))


def test_repeated_passes_keep_labels_and_release_wrappers() -> None:
    """Two recordings label identically and leave MLX unwrapped."""

    from torchlens.backends.mlx.wrappers import is_mlx_wrapped

    model = _mlp()
    first = tl.record(model, _x(), save=_ops, backend="mlx")
    second = tl.record(model, _x(), save=_ops, backend="mlx")

    assert [r.ctx.raw_label for r in first.records] == [r.ctx.raw_label for r in second.records]
    assert not is_mlx_wrapped()


def test_disk_storage_writes_selected_values(tmp_path: Path) -> None:
    """``storage=`` routes each selected value through the MLX codec to the bundle."""

    storage = tl.options.StorageOptions(path=tmp_path / "mlx_bundle", retain_in_memory=True)
    recording = tl.record(_chain, _x(), save=_maximum_ops, storage=storage, backend="mlx")

    assert len(recording.records) == 2
    assert all(r.metadata.get("blob_id") is not None for r in recording.records)
    _assert_records_match_trace(recording, _trace_payloads(_chain, _x()))


def test_codec_ram_copy_is_independent() -> None:
    """The RAM copy does not follow later writes to the live array."""

    value = mx.array([1.0, 2.0, 3.0])
    copy = _record.MLXPayloadCodec().ram_copy(value)
    value[0] = 9.0

    np.testing.assert_array_equal(np.array(copy), np.asarray([1.0, 2.0, 3.0], dtype=np.float32))


@pytest.mark.parametrize(
    ("dtype", "torch_dtype"),
    [("float32", "torch.float32"), ("bfloat16", "torch.bfloat16"), ("int32", "torch.int32")],
)
def test_codec_disk_tensor_matches_values(dtype: str, torch_dtype: str) -> None:
    """The disk codec gives a CPU torch tensor with the same numbers and dtype."""

    value = mx.array([1.5, -2.25, 3.0]).astype(getattr(mx, dtype))
    tensor = _record.MLXPayloadCodec().to_disk_tensor(value)

    assert str(tensor.dtype) == torch_dtype
    np.testing.assert_array_equal(
        tensor.float().numpy(), np.array(value.astype(mx.float32), copy=True)
    )
