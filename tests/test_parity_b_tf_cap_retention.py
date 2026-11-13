"""TensorFlow S1: selective save keeps only selected payloads and no live activations.

A capture that will run no backward pass releases each op's live tensors once
its edges are recorded, and the producer index holds tensors weakly. These
tests check that a sparse capture's payloads equal the same ops of a full
capture, that a plain capture still saves and validates everything, and that
the weak index never hands a dead tensor's label to a new tensor.
"""

from __future__ import annotations

import gc
from typing import Any

import numpy as np
import pytest

import torchlens as tl

tf = pytest.importorskip("tensorflow")

from torchlens.backends.tf._retention import (  # noqa: E402
    WeakProducerIndex,
    release_live_tensors,
    retention_policy,
)

pytestmark = [pytest.mark.tf_backend, pytest.mark.rare]

_X = np.random.default_rng(0).standard_normal(64).astype(np.float32)


def _chain(x: Any) -> Any:
    """Four ``relu((h + 1) * 0.5)`` blocks."""

    h = x
    for _ in range(4):
        h = tf.nn.relu(tf.multiply(tf.add(h, 1.0), 0.5))
    return h


def _payloads(trace: Any) -> dict[str, np.ndarray]:
    """Saved payloads by op label."""

    return {op.label: np.asarray(op.out) for op in trace._layer_list if op.has_saved_activation}


@pytest.mark.parametrize("func_name", ["Relu", "Mul", "AddV2"])
def test_sparse_payloads_equal_full_capture(func_name: str) -> None:
    """``save=tl.func(...)`` keeps exactly the selected ops, with the full capture's values."""

    x = tf.constant(_X)
    full = _payloads(tl.trace(_chain, x))
    sparse_trace = tl.trace(_chain, x, save=tl.func(func_name))
    sparse = _payloads(sparse_trace)
    selected = {op.label for op in sparse_trace._layer_list if op.func_name == func_name}
    inputs = {op.label for op in sparse_trace._layer_list if op.func_name == "input"}
    assert selected
    assert set(sparse) == selected | inputs  # model inputs are always kept
    for label, value in sparse.items():
        np.testing.assert_array_equal(value, full[label])


def test_plain_capture_saves_every_payload_and_releases_live_tensors() -> None:
    """A plain capture still saves every op and validates; its records hold no live outputs."""

    x = tf.constant(_X)
    trace = tl.trace(_chain, x)
    assert all(op.has_saved_activation for op in trace._layer_list)
    np.testing.assert_allclose(trace[trace.output_layers[0]].out, _chain(x).numpy())
    assert trace.validate_forward_pass(_chain(x))
    captures = getattr(trace, "_tf_op_captures", ())
    assert captures
    assert all(capture.output_tensor is None for capture in captures)


def test_weak_index_entry_dies_with_its_tensor() -> None:
    """A dead tensor's label is gone; a new tensor never inherits it."""

    index = WeakProducerIndex()
    first = tf.constant([1.0, 2.0])
    index[first.ref()] = "add_1"
    assert index.get(first.ref()) == "add_1"
    assert len(index) == 1
    del first
    gc.collect()
    assert len(index) == 0
    second = tf.constant([1.0, 2.0])
    assert index.get(second.ref()) is None


def test_weak_index_iterates_live_refs_and_deletes() -> None:
    """Iteration yields ``ref()`` keys for live tensors; ``del`` removes an entry."""

    index = WeakProducerIndex()
    tensors = [tf.constant([float(i)]) for i in range(3)]
    for i, tensor in enumerate(tensors):
        index[tensor.ref()] = f"op_{i}"
    assert {key.deref().numpy()[0] for key in index} == {0.0, 1.0, 2.0}
    del index[tensors[1].ref()]
    assert index.get(tensors[1].ref()) is None
    assert sorted(index.values()) == ["op_0", "op_2"]


class _Session:
    """Minimal stand-in for the eager session's retention switches."""

    def __init__(self, retain: bool, tap: Any = None) -> None:
        self.retain_live_tensors = retain
        self.output_tap = tap


@pytest.mark.parametrize(
    ("retain", "tap", "expected"),
    [(True, None, True), (False, None, False), (False, object(), True)],
)
def test_retention_policy(retain: bool, tap: Any, expected: bool) -> None:
    """Live tensors survive for backward passes and the derived-gradient tap only."""

    assert retention_policy(_Session(retain, tap))(object()) is expected


def test_release_keeps_literal_inputs() -> None:
    """Released records drop producer-linked inputs and keep literal ones as data."""

    from torchlens.backends.tf.op_callback_capture import TFInputCapture, TFOpCapture

    capture = _first_capture_with_literal(TFOpCapture, TFInputCapture)
    released = release_live_tensors(capture)
    assert released.output_tensor is None
    for before, after in zip(capture.inputs, released.inputs, strict=True):
        literal = before.producer_label_raw is None and before.source_label_raw is None
        assert (after.tensor is before.tensor) if literal else after.tensor is None


def _first_capture_with_literal(op_type: Any, input_type: Any) -> Any:
    """A kept op record from a capture that retains live tensors, with a literal input."""

    from torchlens.backends.tf.op_callback_capture import TFEagerCaptureSession

    session = TFEagerCaptureSession(
        tf=tf,
        callable_obj=_chain,
        args=(tf.constant(_X),),
        kwargs={},
        module_tree=None,
        save_payloads=True,
        save_predicate=None,
    )
    assert session.retain_live_tensors
    session.run()
    for capture in session.op_captures:
        assert isinstance(capture, op_type)
        if any(isinstance(item, input_type) and item.tensor is not None for item in capture.inputs):
            return capture
    pytest.fail("no op record with a live input")


def test_op_records_mark_payloads_the_selector_declined() -> None:
    """``payload_declined`` is set exactly for the ops ``save=`` did not select."""

    x = tf.constant(_X)
    assert not any(c.payload_declined for c in tl.trace(_chain, x)._tf_op_captures)
    sparse = tl.trace(_chain, x, save=tl.func("Relu"))
    declined = {c.op_type: c.payload_declined for c in sparse._tf_op_captures}
    assert declined == {"Relu": False, "AddV2": True, "Mul": True}
