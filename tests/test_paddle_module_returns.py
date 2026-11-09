"""Paddle records what each layer call returned, so ``Module.out`` works, also after save and load.

Each case is checked against the value the layer actually returned in a plain (uncaptured)
forward of the same model instance, read with Paddle's own forward post-hooks.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

paddle = pytest.importorskip("paddle")

import torchlens as tl  # noqa: E402
from torchlens._ir.container import TupleIndex  # noqa: E402
from torchlens.backends._module_returns import returned_op_labels  # noqa: E402
from torchlens.intervention.errors import MultiOutputModuleError  # noqa: E402
from torchlens.validation._invariants import check_metadata_invariants  # noqa: E402

pytestmark = [pytest.mark.backend_paddle, pytest.mark.rare]


class Pair(paddle.nn.Layer):
    """Return a tuple with a nested tuple and a duplicated leaf."""

    def forward(self, x: Any) -> Any:
        """Return ``(a, (b, a))``."""

        a = x + 1.0
        b = x * 2.0
        return a, (b, a)


class PassThrough(paddle.nn.Layer):
    """Return the input unchanged."""

    def forward(self, x: Any) -> Any:
        """Return ``x``."""

        return x


class Echo(paddle.nn.Layer):
    """Return the input next to a value computed from it."""

    def forward(self, x: Any) -> Any:
        """Return ``(x, tanh(x))``."""

        return x, paddle.tanh(x)


class Square(paddle.nn.Layer):
    """Square the input."""

    def forward(self, x: Any) -> Any:
        """Return ``x * x``."""

        return x * x


class ReturnsModel(paddle.nn.Layer):
    """Exercise every recorded-return case in one forward."""

    def __init__(self) -> None:
        """Build the children; ``unused`` is never called."""

        super().__init__()
        self.lin = paddle.nn.Linear(4, 4)
        self.pair = Pair()
        self.ident = PassThrough()
        self.echo = Echo()
        self.sq = Square()
        self.side = Square()
        self.unused = paddle.nn.Linear(4, 4)

    def forward(self, x: Any) -> Any:
        """Run the children; ``side``'s return is never consumed."""

        h = self.lin(x)
        a, (b, _a) = self.pair(h)
        i = self.ident(b)
        e_in, e_out = self.echo(i)
        self.side(h)
        s = self.sq(self.sq(e_out))
        return s + a + e_in


def _np(value: Any) -> np.ndarray:
    """Return a NumPy copy of a Paddle tensor or array-like."""

    return value.numpy() if hasattr(value, "numpy") else np.asarray(value)


def _actual_returns(model: Any, x: Any) -> dict[str, Any]:
    """Return each layer call's returned value from a plain forward, keyed ``address:call``."""

    seen: dict[str, Any] = {}
    counts: dict[str, int] = {}
    handles = []
    for address, layer in model.named_sublayers():

        def hook(_layer: Any, _args: Any, output: Any, address: str = address) -> None:
            counts[address] = counts.get(address, 0) + 1
            seen[f"{address}:{counts[address]}"] = output

        handles.append(layer.register_forward_post_hook(hook))
    try:
        model(x)
    finally:
        for handle in handles:
            handle.remove()
    return seen


@pytest.fixture(scope="module")
def captured(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Any, Any, dict[str, Any]]]:
    """Capture the model, save and load it, and read the actual returns."""

    paddle.seed(0)
    model = ReturnsModel()
    model.eval()
    x = paddle.rand([2, 4], dtype="float32")
    trace = tl.trace(model, x, backend="paddle")
    path = Path(tmp_path_factory.mktemp("paddle_returns")) / "returns.tlspec"
    trace.save(path)
    loaded = tl.load(path)
    yield trace, loaded, _actual_returns(model, x)
    trace.cleanup()
    loaded.cleanup()


@pytest.fixture(params=["live", "loaded"])
def trace_and_actual(
    request: pytest.FixtureRequest, captured: tuple[Any, Any, dict[str, Any]]
) -> tuple[Any, dict[str, Any]]:
    """Yield the live or the loaded trace with the actual returns."""

    trace, loaded, actual = captured
    return (trace if request.param == "live" else loaded), actual


def test_single_return_resolves_module_out(trace_and_actual: tuple[Any, dict[str, Any]]) -> None:
    """A layer returning one computed tensor answers ``Module.out`` with that value."""

    trace, actual = trace_and_actual
    np.testing.assert_allclose(_np(trace.modules["lin"].out), _np(actual["lin:1"]))
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


def test_unused_child_records_nothing(trace_and_actual: tuple[Any, dict[str, Any]]) -> None:
    """A child that never ran has no recorded output.

    The preview module-log builder lists one call for every discovered module, so ``unused:1``
    may exist; it predates recorded returns, and nothing may be recorded on it.
    """

    trace, _actual = trace_and_actual
    calls = [label for label in trace.module_calls.keys() if label.startswith("unused:")]
    for call_label in calls:
        assert list(trace.module_calls[call_label].output_ops) == []
        assert not trace.module_calls[call_label].output_paths
        assert returned_op_labels(trace, call_label) == []


def test_metadata_invariants_hold(trace_and_actual: tuple[Any, dict[str, Any]]) -> None:
    """Recorded returns keep the module-call invariants intact."""

    trace, _actual = trace_and_actual
    assert check_metadata_invariants(trace) is True
