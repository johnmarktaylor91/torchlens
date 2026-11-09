"""TensorFlow records what each module call returned, so ``Module.out`` works, also after save and load.

Each case is checked against the value the child actually returned to its caller in a plain
(uncaptured) forward of the same model instance, kept by the parent's ``call``.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from conftest import tensorflow_backend_modules

import torchlens as tl
from torchlens._ir.container import TupleIndex
from torchlens.backends._module_returns import returned_op_labels
from torchlens.intervention.errors import MultiOutputModuleError
from torchlens.validation._invariants import check_metadata_invariants

tf, keras, _TF_BACKEND_SKIP_REASON = tensorflow_backend_modules()

pytestmark = [
    pytest.mark.tf_backend,
    pytest.mark.skipif(
        _TF_BACKEND_SKIP_REASON is not None,
        reason=_TF_BACKEND_SKIP_REASON or "TensorFlow backend stack is supported",
    ),
    pytest.mark.rare,
]

_ACTUAL: dict[str, Any] = {}
_KEEPING: list[bool] = [False]


def _keep(call_label: str, value: Any) -> Any:
    """Keep a child's returned value during the plain forward, then pass it on."""

    if _KEEPING[0]:
        _ACTUAL[call_label] = value
    return value


class Pair(keras.layers.Layer):
    """Return a tuple with a nested tuple and a duplicated leaf."""

    def call(self, x: Any) -> Any:
        """Return ``(a, (b, a))``."""

        a = x + 1.0
        b = x * 2.0
        return a, (b, a)


class PassThrough(keras.layers.Layer):
    """Return the input unchanged."""

    def call(self, x: Any) -> Any:
        """Return ``x``."""

        return x


class Echo(keras.layers.Layer):
    """Return the input next to a value computed from it."""

    def call(self, x: Any) -> Any:
        """Return ``(x, tanh(x))``."""

        return x, tf.tanh(x)


class Square(keras.layers.Layer):
    """Square the input."""

    def call(self, x: Any) -> Any:
        """Return ``x * x``."""

        return x * x


class ReturnsModel(keras.Model):
    """Exercise every recorded-return case in one forward."""

    def __init__(self) -> None:
        """Build the children; ``unused`` is never called."""

        super().__init__(name="returns_model")
        self.lin = keras.layers.Dense(4, name="lin")
        self.pair = Pair(name="pair")
        self.ident = PassThrough(name="ident")
        self.echo = Echo(name="echo")
        self.sq = Square(name="sq")
        self.side = Square(name="side")
        self.unused = keras.layers.Dense(4, name="unused")

    def call(self, x: Any) -> Any:
        """Run the children; ``side``'s return is never consumed."""

        h = _keep("lin:1", self.lin(x))
        a, (b, _a) = _keep("pair:1", self.pair(h))
        i = _keep("ident:1", self.ident(b))
        e_in, e_out = _keep("echo:1", self.echo(i))
        _keep("side:1", self.side(h))
        s = _keep("sq:2", self.sq(_keep("sq:1", self.sq(e_out))))
        return s + a + e_in


@pytest.fixture(scope="module")
def captured(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Any, Any, dict[str, Any]]]:
    """Capture the model, save and load it, and keep the actual returns."""

    tf.random.set_seed(0)
    model = ReturnsModel()
    x = tf.random.uniform((2, 4), seed=0)
    model(x)
    trace = tl.trace(model, x, backend="tf")
    path = Path(tmp_path_factory.mktemp("tf_returns")) / "returns.tlspec"
    trace.save(path)
    loaded = tl.load(path)
    _ACTUAL.clear()
    _KEEPING[0] = True
    try:
        model(x)
    finally:
        _KEEPING[0] = False
    yield trace, loaded, dict(_ACTUAL)
    trace.cleanup()
    loaded.cleanup()


@pytest.fixture(params=["live", "loaded"])
def trace_and_actual(
    request: pytest.FixtureRequest, captured: tuple[Any, Any, dict[str, Any]]
) -> tuple[Any, dict[str, Any]]:
    """Yield the live or the loaded trace with the actual returns."""

    trace, loaded, actual = captured
    return (trace if request.param == "live" else loaded), actual


def _np(value: Any) -> np.ndarray:
    """Return a NumPy copy of a TensorFlow tensor or array-like."""

    return value.numpy() if hasattr(value, "numpy") else np.asarray(value)


def test_single_return_resolves_module_out(trace_and_actual: tuple[Any, dict[str, Any]]) -> None:
    """A layer returning one computed tensor answers ``Module.out`` with that value."""

    trace, actual = trace_and_actual
    np.testing.assert_allclose(_np(trace.modules["lin"].out), _np(actual["lin:1"]), rtol=1e-6)
    assert trace.module_calls["lin:1"].output_paths == ((),)


def test_nested_tuple_with_duplicate_leaf(trace_and_actual: tuple[Any, dict[str, Any]]) -> None:
    """Nested tuple leaves keep their paths; a duplicated leaf is one op with two paths."""

    trace, actual = trace_and_actual
    call = trace.module_calls["pair:1"]
    a, (b, _a) = actual["pair:1"]
    assert len(call.output_ops) == 2
    np.testing.assert_allclose(_np(call.outs[0]), _np(a))
    np.testing.assert_allclose(_np(call.outs[1]), _np(b))
    assert call.output_paths == (
        (TupleIndex(0),),
        (TupleIndex(1), TupleIndex(0)),
        (TupleIndex(1), TupleIndex(1)),
    )


def test_pass_through_records_no_output_op(trace_and_actual: tuple[Any, dict[str, Any]]) -> None:
    """A layer returning its input keeps no output op; the returned roster names the input."""

    trace, actual = trace_and_actual
    call = trace.module_calls["ident:1"]
    assert list(call.output_ops) == []
    assert call.output_paths == ((),)
    with pytest.raises(MultiOutputModuleError):
        trace.modules["ident"].out
    (label,) = returned_op_labels(trace, "ident:1")
    np.testing.assert_allclose(_np(trace.layers[label].out), _np(actual["ident:1"]))


def test_returned_input_beside_computed_value(trace_and_actual: tuple[Any, dict[str, Any]]) -> None:
    """A returned input stays outside ``output_ops``; the computed leaf is the only output."""

    trace, actual = trace_and_actual
    call = trace.module_calls["echo:1"]
    returned_input, computed = actual["echo:1"]
    np.testing.assert_allclose(_np(trace.modules["echo"].out), _np(computed))
    assert call.output_paths == ((TupleIndex(0),), (TupleIndex(1),))
    roster = returned_op_labels(trace, "echo:1")
    assert len(roster) == 2
    assert roster[0] == returned_op_labels(trace, "ident:1")[0]
    np.testing.assert_allclose(_np(trace.layers[roster[0]].out), _np(returned_input))


def test_repeated_calls_record_each_return(trace_and_actual: tuple[Any, dict[str, Any]]) -> None:
    """Each call of a reused layer has its own recorded output."""

    trace, actual = trace_and_actual
    assert trace.modules["sq"].num_calls == 2
    for call_label in ("sq:1", "sq:2"):
        np.testing.assert_allclose(_np(trace.module_calls[call_label].out), _np(actual[call_label]))


def test_unconsumed_return_is_recorded(trace_and_actual: tuple[Any, dict[str, Any]]) -> None:
    """A layer whose return nothing downstream reads still records it."""

    trace, actual = trace_and_actual
    np.testing.assert_allclose(_np(trace.modules["side"].out), _np(actual["side:1"]))


def test_unused_child_has_no_call(trace_and_actual: tuple[Any, dict[str, Any]]) -> None:
    """A child that never ran has no module call."""

    trace, _actual = trace_and_actual
    assert not any(label.startswith("unused:") for label in trace.module_calls.keys())


def test_metadata_invariants_hold(trace_and_actual: tuple[Any, dict[str, Any]]) -> None:
    """Recorded returns keep the module-call invariants intact."""

    trace, _actual = trace_and_actual
    assert check_metadata_invariants(trace) is True


def test_raw_tf_module_records_returns() -> None:
    """A raw ``tf.Module`` child with its own ``__call__`` records its return too."""

    class Affine(tf.Module):
        """Raw module computing ``x @ w + b``."""

        def __init__(self) -> None:
            """Create the variables."""

            super().__init__(name="affine")
            self.w = tf.Variable([[2.0], [3.0]], name="w")
            self.b = tf.Variable([1.0], name="b")

        def __call__(self, x: Any) -> Any:
            """Return ``x @ w + b``."""

            return tf.matmul(x, self.w) + self.b

    class Root(tf.Module):
        """Raw root calling one child."""

        def __init__(self) -> None:
            """Create the child."""

            super().__init__(name="root")
            self.affine = Affine()

        def __call__(self, x: Any) -> Any:
            """Return ``relu(affine(x))``."""

            return tf.nn.relu(self.affine(x))

    root = Root()
    x = tf.ones((1, 2))
    trace = tl.trace(root, x, backend="tf")
    try:
        np.testing.assert_allclose(_np(trace.modules["affine"].out), _np(root.affine(x)))
    finally:
        trace.cleanup()
