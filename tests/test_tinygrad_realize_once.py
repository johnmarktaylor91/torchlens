"""Realize-once capture on the tinygrad backend.

A capture realizes every recorded value in one tinygrad evaluation after the
forward, so payloads come from that single realization and random-number state
is consumed as one eager forward consumes it. Random-init ``tinygrad.nn``
layers capture and validate; random numbers drawn inside the forward refuse
``rng_state_in_graph`` before anything is realized.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import torchlens as tl
from torchlens.errors import BackendUnsupportedError

tinygrad = pytest.importorskip("tinygrad")
Tensor = tinygrad.Tensor


pytestmark = [pytest.mark.backend_tinygrad, pytest.mark.rare]

_X = np.asarray([[1.0, -0.5, 0.25, 2.0], [-1.5, 0.75, 1.0, -0.25]], dtype=np.float32)


def _rng_state() -> list[int]:
    """tinygrad's seed and per-device RNG counters, read back to the host."""

    values = [int(Tensor._seed)]
    for device in sorted(Tensor._device_rng_counters):
        values.extend(int(v) for v in Tensor._device_rng_counters[device].numpy().reshape(-1))
    return values


def _random_linear() -> Any:
    """A seeded default-init ``nn.Linear`` whose weights are still lazy RNG graphs."""

    from tinygrad import nn

    Tensor.manual_seed(0)
    return nn.Linear(4, 3)


def _op_payloads(trace: Any) -> list[tuple[str, np.ndarray]]:
    """``(func_name, payload)`` for every op of a tinygrad trace, in order."""

    return [(str(op.func_name), np.asarray(op.out.numpy())) for op in trace._layer_list]


def test_random_init_layer_captures_like_one_eager_forward() -> None:
    """A random-init layer captures, validates, and consumes RNG like one eager forward."""

    layer = _random_linear()
    trace = tl.trace(lambda value: layer(value).relu(), Tensor(_X), backend="tinygrad")
    after_capture = _rng_state()

    eager_layer = _random_linear()
    eager = eager_layer(Tensor(_X)).relu().numpy()
    after_eager = _rng_state()

    assert trace.tinygrad_payload_policy == "realize_once"
    np.testing.assert_array_equal(trace[trace.output_layers[0]].out.numpy(), eager)
    assert after_capture == after_eager
    assert trace.validate_forward_pass([layer(Tensor(_X)).relu()]) is True


def test_random_init_weights_record_their_realized_values() -> None:
    """The weight ops hold the values the layer's weights were realized to, no RNG subgraph."""

    layer = _random_linear()
    trace = tl.trace(lambda value: layer(value), Tensor(_X), backend="tinygrad")

    payloads = [payload for _name, payload in _op_payloads(trace)]
    weight = layer.weight.numpy()
    bias = layer.bias.numpy()
    assert any(p.shape == weight.shape and np.array_equal(p, weight) for p in payloads)
    assert any(p.shape == bias.shape and np.array_equal(p, bias) for p in payloads)
    recorded_uops = {str((op.annotations or {}).get("tinygrad_uop")) for op in trace._layer_list}
    assert "THREEFRY" not in recorded_uops
    assert not any(p.dtype == np.uint32 for p in payloads)


def test_random_init_capture_equals_capture_after_realizing_weights() -> None:
    """Capturing lazy random weights records what capturing pre-realized ones records.

    The same ops, shapes and dtypes, and the same input and weight values.
    Values computed from the weights are compared with the eager forward in
    ``test_random_init_layer_captures_like_one_eager_forward``: tinygrad
    computes lazy weights inline in their consumers, so a forward over lazy
    weights and one over realized weights may round differently, in eager
    tinygrad as in the capture.
    """

    lazy_layer = _random_linear()
    lazy_trace = tl.trace(lambda value: lazy_layer(value).relu(), Tensor(_X), backend="tinygrad")

    realized_layer = _random_linear()
    Tensor.realize(realized_layer.weight, realized_layer.bias)
    realized_trace = tl.trace(
        lambda value: realized_layer(value).relu(), Tensor(_X), backend="tinygrad"
    )

    lazy_ops = list(lazy_trace._layer_list)
    realized_ops = list(realized_trace._layer_list)
    assert [op.func_name for op in lazy_ops] == [op.func_name for op in realized_ops]
    for left, right in zip(lazy_ops, realized_ops, strict=True):
        left_value, right_value = np.asarray(left.out.numpy()), np.asarray(right.out.numpy())
        assert (left_value.shape, left_value.dtype) == (right_value.shape, right_value.dtype)
        if not left.parents:
            np.testing.assert_array_equal(left_value, right_value)


def test_capture_realizes_the_forward_graph_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """One ``realize_once`` call; nothing outside it evaluates RNG or a recorded op again."""

    from torchlens.backends._protocols.tinygrad import TinygradProtocol

    layer = _random_linear()
    calls: list[int] = []
    inside: list[bool] = []
    outside: list[Any] = []
    original_once = TinygradProtocol.realize_once
    original_realize = Tensor.realize

    def counting_once(self: Any, values: Any) -> None:
        inside.append(True)
        try:
            original_once(self, values)
        finally:
            inside.pop()
            calls.append(len(values))

    def spying_realize(self: Any, *lst: Any, **kwargs: Any) -> Any:
        if not inside:
            for tensor in (self, *lst):
                uop = tensor.uop
                if uop.device is not None and not uop.has_buffer_identity():
                    outside.append(uop)
        return original_realize(self, *lst, **kwargs)

    monkeypatch.setattr(TinygradProtocol, "realize_once", counting_once)
    monkeypatch.setattr(Tensor, "realize", spying_realize)
    trace = tl.trace(lambda value: layer(value).relu(), Tensor(_X), backend="tinygrad")
    monkeypatch.undo()

    assert len(calls) == 1
    recorded = {
        id(capture.uop)
        for capture in trace.tinygrad_uop_captures
        if capture.parent_labels
        and any(node.op.name == "BUFFER" for node in capture.uop.toposort())
    }
    assert recorded
    for uop in outside:
        nodes = uop.toposort()
        assert not any(node.op.name == "THREEFRY" for node in nodes)
        assert not any(id(node) in recorded for node in nodes)


def test_in_forward_random_numbers_refuse_before_realizing() -> None:
    """``Tensor.rand`` inside the traced callable refuses ``rng_state_in_graph``, unrealized."""

    Tensor.manual_seed(0)
    returned: list[Any] = []

    def noisy(value: Any) -> Any:
        out = value * Tensor.rand(*_X.shape)
        returned.append(out)
        return out

    with pytest.raises(BackendUnsupportedError) as info:
        tl.trace(noisy, Tensor(_X), backend="tinygrad")

    assert info.value.code == "rng_state_in_graph"
    assert "THREEFRY" in str(info.value)
    assert not returned[0].uop.has_buffer_identity()


def test_closure_constant_from_ndarray_captures_and_validates() -> None:
    """A numpy-built closure weight realizes with the capture and replays clean."""

    weight = Tensor(np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    bias = Tensor(np.asarray([0.5, -0.5], dtype=np.float32))
    x = Tensor(np.asarray([[1.0, -1.0]], dtype=np.float32))

    def forward(value: Any) -> Any:
        return (value.matmul(weight) + bias).relu()

    trace = tl.trace(forward, x, backend="tinygrad")

    np.testing.assert_allclose(trace[trace.output_layers[0]].out.numpy(), [[0.0, 0.0]])
    assert trace.validate_forward_pass([forward(x)]) is True


def test_unrealizable_value_refuses_typed_naming_a_buffer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A recorded value tinygrad cannot realize refuses ``constant_unrealizable``."""

    from torchlens.backends._protocols.tinygrad import TinygradProtocol

    def refuse(self: Any, values: Any) -> None:
        raise NotImplementedError("need copyin")

    weight = Tensor([[1.0, 2.0], [3.0, 4.0]])
    monkeypatch.setattr(TinygradProtocol, "realize_once", refuse)
    with pytest.raises(BackendUnsupportedError) as info:
        tl.trace(lambda value: value.matmul(weight), Tensor([[1.0, -1.0]]), backend="tinygrad")

    assert info.value.code == "constant_unrealizable"
    assert "BUFFER" in str(info.value)


def test_payload_structure_check_realizes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deciding what a capture records never realizes a value."""

    from torchlens.backends.tinygrad._uop_graph import _is_materializable_uop

    realized: list[Any] = []
    original = Tensor.realize

    def spying(self: Any, *lst: Any, **kwargs: Any) -> Any:
        realized.append(self)
        return original(self, *lst, **kwargs)

    lazy = (Tensor([1.0, 2.0]) + 1.0) * 3.0
    monkeypatch.setattr(Tensor, "realize", spying)
    assert _is_materializable_uop(lazy.uop) is True
    monkeypatch.undo()

    assert realized == []
    assert not lazy.uop.has_buffer_identity()
