"""Native backward capture on Paddle (F5): records, the backward validator and refusals.

Until the Paddle capture calls the hooks itself, :func:`_capture` stands in for
it: it wraps ``tl.trace`` in ``backward_ready_scope``, notes every labelled
tensor where the capture labels it, keeps the live output, and retains the
state at the end. ``record_backward`` then runs through ``Trace.backward`` with
the registry's Paddle entry pointed at the native surface, as the dispatch
tests of the shared builder do.
"""

from __future__ import annotations

import dataclasses
import types
from typing import Any

import pytest

paddle = pytest.importorskip("paddle")

import torchlens as tl  # noqa: E402
from torchlens.backends import registry  # noqa: E402
from torchlens.backends._backward_native import (  # noqa: E402
    NativeBackwardSurface,
    native_backward_surface,
    paddle as native,  # noqa: E402
)
from torchlens.backends._replay_adapters import paddle as adapter  # noqa: E402
from torchlens.backends._reverse_engine import graph_from_recipes  # noqa: E402
from torchlens.backends.paddle.backend import PaddleBackend  # noqa: E402
from torchlens.backends.paddle.tensor_store import PaddleTensorLabelStore  # noqa: E402
from torchlens.errors import RunCapabilityUnavailableError  # noqa: E402
from torchlens.validation._backward_neutral import (  # noqa: E402
    RecordedBackward,
    check_backward_records,
    recorded_backward_from_trace,
)

pytestmark = [pytest.mark.backend_paddle, pytest.mark.rare]

F = paddle.nn.functional


@pytest.fixture
def ready(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Patch in the capture's hook calls and the registry's backward entry."""

    box: list[native.BackwardScope] = []
    set_label = PaddleTensorLabelStore.set_label
    mark_outputs = PaddleBackend._mark_outputs

    def noting_set_label(self: Any, tensor: Any, label: str) -> None:
        set_label(self, tensor, label)
        if box:
            box[0].note(label, tensor)

    def keeping_mark_outputs(self: Any, trace: Any, output: Any) -> None:
        if box:
            box[0].output = output
        mark_outputs(self, trace, output)

    monkeypatch.setattr(PaddleTensorLabelStore, "set_label", noting_set_label)
    monkeypatch.setattr(PaddleBackend, "_mark_outputs", keeping_mark_outputs)
    spec = registry.get_backend_spec("paddle")
    monkeypatch.setitem(
        registry._REGISTRY,
        "paddle",
        dataclasses.replace(
            spec,
            capabilities=dataclasses.replace(spec.capabilities, backward_capture=True),
            capability_implementations={
                **(spec.capability_implementations or {}),
                "backward_capture": lambda: NativeBackwardSurface("paddle", native),
            },
        ),
    )

    def capture(model: Any, *inputs: Any) -> Any:
        session = types.SimpleNamespace()
        with native.backward_ready_scope(session) as scope:
            box.append(scope)
            try:
                trace = tl.trace(model, inputs[0] if len(inputs) == 1 else inputs, backend="paddle")
            finally:
                box.clear()
        native.retain_backward_state(trace, session)
        return trace

    return capture


def _leaf(shape: list[int], seed: int) -> Any:
    paddle.seed(seed)
    value = paddle.randn(shape)
    value.stop_gradient = False
    return value


def _raw(trace: Any) -> dict[str, str]:
    return {
        op.layer_label: str(getattr(op, "_label_raw", op.layer_label)) for op in trace._layer_list
    }


def _validate(trace: Any, pass_number: int, leaves: dict[str, Any]) -> Any:
    """Run the shared backward validator on one recorded pass.

    The validator's graph comes from the adapter's recipes, which name ops by
    raw label; the trace's records name them by final label. Only the keys are
    translated here; every comparison is the validator's own.
    """

    raw = _raw(trace)
    recorded = recorded_backward_from_trace(trace, pass_number)
    rekeyed = RecordedBackward(
        calls={raw[label]: call for label, call in recorded.calls.items()},
        grads={raw[label]: grad for label, grad in recorded.grads.items()},
    )
    ops = [op for op in trace._layer_list if not op.is_input]
    values = {raw[op.layer_label]: op.out for op in trace._layer_list}
    graph = graph_from_recipes(trace, adapter, ops, values.__getitem__)
    inputs = {
        raw[op.layer_label]: grad
        for op in trace._layer_list
        if op.is_input
        for grad in [leaves.get(op.layer_label)]
        if grad is not None
    }
    return check_backward_records(trace, adapter, graph, rekeyed, inputs)


def _close(left: Any, right: Any) -> bool:
    return bool(paddle.allclose(left, right, rtol=1e-6, atol=1e-7))


class Diamond(paddle.nn.Layer):
    """A tensor used three times, one of them twice by one call."""

    def __init__(self) -> None:
        super().__init__()
        paddle.seed(1)
        self.linear = paddle.nn.Linear(3, 4)

    def forward(self, x: Any) -> Any:
        h = paddle.tanh(self.linear(x))
        return h * h + paddle.sin(h)


class SharedLinear(paddle.nn.Layer):
    """One parameter set reached by two calls."""

    def __init__(self) -> None:
        super().__init__()
        paddle.seed(2)
        self.linear = paddle.nn.Linear(4, 4)

    def forward(self, x: Any) -> Any:
        return self.linear(F.relu(self.linear(x)))


class MultiOutput(paddle.nn.Layer):
    """A three-way split with one part unused, and top-k values with integer indices."""

    def forward(self, x: Any) -> Any:
        a, b, _unused = paddle.split(x, 3, axis=1)
        values, _indices = paddle.topk(paddle.exp(a), k=1, axis=1)
        return values * paddle.sin(b)


class Gathered(paddle.nn.Layer):
    """A gather whose integer index Paddle gives no gradient slot."""

    def forward(self, x: Any, index: Any) -> Any:
        return paddle.gather(paddle.tanh(x), index)


class Concat(paddle.nn.Layer):
    """A concat whose elements have one consumer each."""

    def forward(self, x: Any) -> Any:
        return paddle.concat([paddle.tanh(x), paddle.sin(x)], axis=0)


class SharedConcat(paddle.nn.Layer):
    """A concat element that also feeds another op."""

    def forward(self, x: Any) -> Any:
        a = paddle.tanh(x)
        return paddle.concat([a, paddle.sin(x)], axis=0).sum() + a.sum()


def _sum(output: Any) -> Any:
    return output.sum()


def _differentiable(model: Any, inputs: tuple[Any, ...]) -> list[Any]:
    """The inputs and parameters Paddle differentiates, in a fixed order."""

    tensors = [*inputs, *model.parameters()]
    return [tensor for tensor in tensors if not tensor.stop_gradient]


def _reference(model: Any, inputs: tuple[Any, ...], loss: Any) -> tuple[Any, list[Any]]:
    """The untraced output and Paddle's own leaf gradients for the same inputs."""

    fresh = tuple(value.detach() for value in inputs)
    for item, value in zip(fresh, inputs, strict=True):
        item.stop_gradient = value.stop_gradient
    output = model(*fresh)
    return output, list(paddle.grad([loss(output)], _differentiable(model, fresh)))


def _run(ready: Any, model: Any, *inputs: Any, loss: Any = _sum) -> tuple[Any, dict[str, Any]]:
    """Capture, check the forward is unchanged, record one pass, check leaf gradients."""

    expected_output, expected = _reference(model, inputs, loss)
    trace = ready(model, *inputs)
    assert _close(trace[trace.output_layers[0]].out, expected_output)
    trace.backward(loss)
    got = [tensor.grad for tensor in _differentiable(model, inputs)]
    assert all(_close(left, right) for left, right in zip(got, expected, strict=True))
    input_ops = [op for op in trace._layer_list if op.is_input]
    pairs = zip(input_ops, inputs, strict=False)
    return trace, {op.layer_label: item.grad for op, item in pairs if not item.stop_gradient}


@pytest.mark.parametrize(
    ("model", "shapes"),
    [
        (Diamond(), [[2, 3]]),
        (SharedLinear(), [[2, 4]]),
        (MultiOutput(), [[2, 6]]),
        (Concat(), [[2, 3]]),
    ],
    ids=["diamond", "shared-linear", "multi-output", "concat"],
)
def test_records_pass_the_backward_validator(
    ready: Any, model: Any, shapes: list[list[int]]
) -> None:
    inputs = tuple(_leaf(shape, seed) for seed, shape in enumerate(shapes))
    trace, leaves = _run(ready, model, *inputs)
    result = _validate(trace, 1, leaves)
    assert result.findings == ()
    assert result.nodes_checked > 0 and result.leaves_checked == len(leaves)


def test_integer_inputs_get_no_slot_and_validate(ready: Any) -> None:
    x = _leaf([4, 3], 3)
    index = paddle.to_tensor([0, 2, 2])
    trace, leaves = _run(ready, Gathered(), x, index)
    result = _validate(trace, 1, leaves)
    assert result.findings == () and result.nodes_checked > 0


def test_grad_fns_carry_paddle_names_and_op_calls(ready: Any) -> None:
    trace, _leaves = _run(ready, SharedLinear(), _leaf([2, 4], 4))
    records = list(trace._grad_fn_logs.values())
    names = {record.class_name for record in records}
    assert "GradNodeAccumulation" in names
    assert any(name.startswith("Matmul") for name in names)
    assert all(record.class_qualname == record.class_name for record in records)
    with_calls = [record for record in records if record.op_label is not None]
    assert with_calls
    assert all(trace[record.op_label].grad is not None for record in with_calls)


def test_second_pass_with_retained_graph_validates(ready: Any) -> None:
    model = Diamond()
    x = _leaf([2, 3], 5)
    trace = ready(model, x)
    trace.backward(_sum, retain_graph=True)
    trace.backward(_sum)
    assert sorted(trace._backward_pass_logs) == [1, 2]
    for number in (1, 2):
        assert _validate(trace, number, {}).findings == ()


def test_unhooked_list_element_with_other_consumers_is_refused(ready: Any) -> None:
    trace = ready(SharedConcat(), _leaf([2, 3], 6))
    with pytest.raises(RunCapabilityUnavailableError, match="_register_post_hook"):
        trace.backward(None)


def test_scalar_output_is_its_own_loss(ready: Any) -> None:
    trace = ready(SharedLinear(), _leaf([2, 4], 7))
    with pytest.raises(RunCapabilityUnavailableError, match="scalar"):
        trace.backward(None)
    scalar = ready(lambda x: paddle.tanh(x).sum(), _leaf([3], 8))
    scalar.backward(None)
    assert sorted(scalar._backward_pass_logs) == [1]


def test_refusals_are_typed(ready: Any) -> None:
    plain = tl.trace(Diamond(), _leaf([2, 3], 9), backend="paddle")
    with pytest.raises(RunCapabilityUnavailableError, match="backward_ready"):
        native.record_backward(plain, _sum)
    trace = ready(Diamond(), _leaf([2, 3], 10))
    with pytest.raises(RunCapabilityUnavailableError, match="retain_graph"):
        native.record_backward(trace, _sum, create_graph=True)
    with pytest.raises(RunCapabilityUnavailableError, match="scalar"):
        native.record_backward(trace, lambda out: out)


def test_registry_factory_finds_the_paddle_module() -> None:
    surface = native_backward_surface("paddle")
    assert surface is not None
    assert surface.module is native
