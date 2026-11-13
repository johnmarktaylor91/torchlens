"""JAX ``tl.record`` producer: values and keys equal ``tl.trace``, stops unwind the interpreter.

The fixture applies the registry hunk pp-merge lands: JAX's ``fastlog`` capability
bound to :func:`torchlens.backends.jax._record.jax_record_producer`.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("jax")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import torchlens as tl  # noqa: E402
from torchlens.backends import registry  # noqa: E402
from torchlens.backends.jax import _record  # noqa: E402

_PRODUCERS: list[_record.JaxRecordProducer] = []


def _bound_producer() -> _record.JaxRecordProducer:
    producer = _record.jax_record_producer()
    _PRODUCERS.append(producer)
    return producer


@pytest.fixture(autouse=True)
def _jax_fastlog_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    """The registry hunk pp-merge applies: JAX's fastlog bound to the record producer."""

    spec = registry.get_backend_spec("jax")
    replaced = dataclasses.replace(
        spec,
        capabilities=dataclasses.replace(spec.capabilities, fastlog=True),
        capability_implementations={
            **(spec.capability_implementations or {}),
            "fastlog": _bound_producer,
        },
    )
    monkeypatch.setitem(registry._REGISTRY, "jax", replaced)
    _PRODUCERS.clear()


def _params(seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    return {
        "w1": jnp.asarray(rng.standard_normal((4, 8)), dtype=jnp.float32),
        "w2": jnp.asarray(rng.standard_normal((8, 3)), dtype=jnp.float32),
    }


def _mlp(params: dict[str, Any], x: jax.Array) -> jax.Array:
    hidden = jnp.tanh(x @ params["w1"])
    return jnp.tanh(hidden @ params["w2"]) * 2.0


def _branchy(params: dict[str, Any], x: jax.Array) -> jax.Array:
    hidden = x @ params["w1"]
    hidden = jax.lax.cond(hidden.sum() > 0, jnp.tanh, jnp.sin, hidden)
    return hidden @ params["w2"]


def _x(seed: int = 1, sign: float = 1.0) -> jax.Array:
    rng = np.random.default_rng(seed)
    return sign * jnp.abs(jnp.asarray(rng.standard_normal((2, 4)), dtype=jnp.float32))


def _ops(ctx: Any) -> bool:
    return bool(ctx.kind == "op")


def _tanh_ops(ctx: Any) -> bool:
    return bool(ctx.kind == "op" and ctx.layer_type == "tanh")


def _trace_payloads(fn: Any, params: Any, x: Any) -> dict[str, Any]:
    """Map each raw label of a full ``tl.trace`` capture to its saved payload."""

    trace = tl.trace(fn, (params, x), backend="jax")
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


@pytest.mark.parametrize("save", [_ops, _tanh_ops], ids=["all_ops", "tanh_ops"])
def test_record_values_and_keys_equal_trace(save: Any) -> None:
    """Every recorded value equals the ``tl.trace`` payload with the same raw label."""

    params, x = _params(), _x()
    recording = tl.record(_mlp, (params, x), save=save, backend="jax")

    _assert_records_match_trace(recording, _trace_payloads(_mlp, params, x))
    if save is _tanh_ops:
        assert [r.ctx.layer_type for r in recording.records] == ["tanh", "tanh"]
    np.testing.assert_array_equal(np.asarray(recording.output), np.asarray(_mlp(params, x)))


def test_all_trace_equations_are_recorded() -> None:
    """Saving every op records one value per interpreted equation of ``tl.trace``."""

    params, x = _params(), _x()
    recording = tl.record(_mlp, (params, x), save=_ops, backend="jax")
    trace = tl.trace(_mlp, (params, x), backend="jax")

    _assert_records_match_trace(recording, _trace_payloads(_mlp, params, x))
    assert _PRODUCERS[-1].equations_bound == len(trace.jax_equation_captures)
    assert len(recording.records) == len(trace.jax_equation_captures)


def test_record_pass_releases_every_primitive_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    """The record pass keeps no capture values, yet every selected value is delivered."""

    results: list[Any] = []
    interpret = _record.interpret_closed_jaxpr_with_inlining

    def spying_interpret(*args: Any, **kwargs: Any) -> Any:
        result = interpret(*args, **kwargs)
        results.append(result)
        return result

    monkeypatch.setattr(_record, "interpret_closed_jaxpr_with_inlining", spying_interpret)
    params, x = _params(), _x()
    recording = tl.record(_mlp, (params, x), save=_ops, backend="jax")

    primitive = [i for i, c in enumerate(results[-1].captures) if c.kind == "primitive"]
    assert primitive
    assert set(primitive) <= set(results[-1].released_capture_indices)
    _assert_records_match_trace(recording, _trace_payloads(_mlp, params, x))


def test_value_dependent_branch_records_the_taken_branch_per_pass() -> None:
    """A concrete ``cond`` predicate picks each pass's branch, as ``tl.trace`` does."""

    params = _params()
    for sign, taken in ((1.0, "tanh"), (-1.0, "sin")):
        x = _x(sign=sign)
        if float((x @ params["w1"]).sum()) <= 0 and taken == "tanh":
            pytest.skip("fixture input did not take the positive branch")
        recording = tl.record(_branchy, (params, x), save=_ops, backend="jax")
        layer_types = {r.ctx.layer_type for r in recording.records}
        payloads = _trace_payloads(_branchy, params, x)
        _assert_records_match_trace(recording, payloads)
        assert taken in layer_types


def test_stop_after_unwinds_the_interpreter() -> None:
    """``stop_after`` ends the pass at the first tanh: no later equation binds."""

    params, x = _params(), _x()
    full = tl.record(_mlp, (params, x), save=_ops, backend="jax")
    full_bound = _PRODUCERS[-1].equations_bound

    def stop(ctx: Any) -> bool:
        return bool(ctx.kind == "op" and ctx.layer_type == "tanh")

    halted = tl.record(_mlp, (params, x), save=_ops, stop_after=stop, backend="jax")
    halted_bound = _PRODUCERS[-1].equations_bound

    assert halted.records[-1].ctx.layer_type == "tanh"
    assert halted_bound < full_bound
    assert len(halted.records) < len(full.records)
    prefix = [r.ctx.raw_label for r in full.records[: len(halted.records)]]
    assert [r.ctx.raw_label for r in halted.records] == prefix
    _assert_records_match_trace(halted, _trace_payloads(_mlp, params, x))


def test_disk_storage_writes_selected_values(tmp_path: Path) -> None:
    """``storage=`` routes each selected value through the JAX codec to the bundle."""

    params, x = _params(), _x()
    storage = tl.options.StorageOptions(path=tmp_path / "jax_bundle", retain_in_memory=True)
    recording = tl.record(_mlp, (params, x), save=_tanh_ops, storage=storage, backend="jax")

    assert len(recording.records) == 2
    assert all(r.metadata.get("blob_id") is not None for r in recording.records)
    _assert_records_match_trace(recording, _trace_payloads(_mlp, params, x))


def test_codec_round_trips_bfloat16() -> None:
    """The disk codec keeps bfloat16 values exactly."""

    value = jnp.asarray([1.5, -2.25, 3.0], dtype=jnp.bfloat16)
    tensor = _record.JaxPayloadCodec().to_disk_tensor(value)

    assert str(tensor.dtype) == "torch.bfloat16"
    np.testing.assert_array_equal(tensor.float().numpy(), np.asarray(value.astype(jnp.float32)))
