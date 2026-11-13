"""The reverse gradient engine on the NumPy fake adapter equals finite differences."""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from tests.support import replay_fake_adapter as fake
from torchlens.backends._reverse_engine import (
    EngineNode,
    ReplayGraph,
    graph_from_recipes,
    reverse_gradients,
)
from torchlens.errors import RunCapabilityUnavailableError

_INPUTS = {"a": np.array([0.5, -1.25, 2.0]), "b": np.array([1.5, 0.75, -0.5])}


def _ops() -> list[fake.FakeOp]:
    """A graph with a repeated parent, reuse across consumers and a two-output op."""

    return [
        fake.FakeOp("a", "input", (), ("a",)),
        fake.FakeOp("b", "input", (), ("b",)),
        fake.FakeOp("c", "mul", ("a", "b"), ("c",)),
        fake.FakeOp("m", "mul", ("a", "a"), ("m",)),
        fake.FakeOp("d", "add", ("c", "m"), ("d",)),
        fake.FakeOp("e", "add", ("d", "a"), ("e",)),
        fake.FakeOp("s", "split2", ("e",), ("s", "s:1"), attrs={"index": 1}),
        fake.FakeOp("u", "scale", ("s",), ("u",), literals=((("factor",), 3.0),)),
        fake.FakeOp("us", "sum", ("u",), ("us",), attrs={"axis": 0}),
        fake.FakeOp("t", "sum", ("s:1",), ("t",), attrs={"axis": 0}),
        fake.FakeOp("n", "neg", ("t",), ("n",)),
        fake.FakeOp("loss", "add", ("us", "n"), ("loss",)),
    ]


def _loss(inputs: Mapping[str, np.ndarray]) -> float:
    return float(fake.build_trace(_ops(), inputs).payloads["loss"])


def _finite_difference(name: str, eps: float = 1e-6) -> np.ndarray:
    grad = np.zeros_like(_INPUTS[name])
    for index in range(grad.size):
        up = {key: value.copy() for key, value in _INPUTS.items()}
        down = {key: value.copy() for key, value in _INPUTS.items()}
        up[name][index] += eps
        down[name][index] -= eps
        grad[index] = (_loss(up) - _loss(down)) / (2 * eps)
    return grad


def _graph(trace: fake.FakeTrace, *, with_inputs: bool = True) -> ReplayGraph:
    ops = [op for op in trace.ops.values() if with_inputs or op.op_identity != "input"]
    return graph_from_recipes(trace, fake, ops, trace.payloads.__getitem__)


@pytest.mark.parametrize("with_inputs", [True, False])
def test_engine_gradients_equal_finite_differences(with_inputs: bool) -> None:
    """Leaf gradients from the engine match central differences of the loss."""

    trace = fake.build_trace(_ops(), _INPUTS)
    grads = reverse_gradients(trace, fake, _graph(trace, with_inputs=with_inputs), {"loss": 1.0})
    for name in ("a", "b"):
        np.testing.assert_allclose(grads[name], _finite_difference(name), rtol=1e-6, atol=1e-8)


def test_engine_sums_over_consumers_and_repeated_occurrences() -> None:
    """``a`` feeds ``c``, ``m`` twice and ``e``: its gradient is the sum of all four."""

    trace = fake.build_trace(_ops(), _INPUTS)
    grads = reverse_gradients(trace, fake, _graph(trace), {"loss": 1.0})
    a, b = _INPUTS["a"], _INPUTS["b"]
    upstream = grads["e"]
    np.testing.assert_allclose(grads["a"], upstream * b + 2 * upstream * a + upstream)
    np.testing.assert_allclose(grads["s"], np.array([3.0]))
    np.testing.assert_allclose(grads["s:1"], np.array([-1.0, -1.0]))


def test_engine_fills_the_unreached_slot_of_a_multi_output_op_with_zeros() -> None:
    """Seeding only one slot of ``split2`` still gives a full-shape parent gradient."""

    trace = fake.build_trace(_ops(), _INPUTS)
    grads = reverse_gradients(trace, fake, _graph(trace), {"s:1": np.ones(2)})
    np.testing.assert_array_equal(grads["e"], np.array([0.0, 1.0, 1.0]))
    assert "us" not in grads


def test_engine_refuses_a_gradient_that_reaches_an_undescribed_op() -> None:
    """An opaque op on the gradient path is refused, never silently skipped."""

    ops = [
        fake.FakeOp("a", "input", (), ("a",)),
        fake.FakeOp("o", "opaque", ("a",), ("o",)),
        fake.FakeOp("loss", "neg", ("o",), ("loss",)),
    ]
    trace = fake.build_trace(ops[:1], {"a": np.ones(2)})
    trace.ops.update({op.label: op for op in ops[1:]})
    trace.payloads.update({"o": np.ones(2), "loss": -np.ones(2)})
    graph = _graph(trace)
    assert graph.node_for("o").blocked == "opaque region"
    with pytest.raises(RunCapabilityUnavailableError, match="opaque region"):
        reverse_gradients(trace, fake, graph, {"loss": np.ones(2)})
    leaf_only = reverse_gradients(trace, fake, graph, {"a": np.ones(2)})
    assert list(leaf_only) == ["a"]


def test_engine_refuses_a_gradient_that_reaches_a_blocked_op_by_its_raw_label() -> None:
    """A recipe-less op whose consumer names its raw label is refused, never taken for a leaf.

    MLX, Paddle and tinygrad recipes name parents by ``_label_raw``; the blocked
    node must carry that spelling or the gradient upstream of it is dropped.
    """

    blocked = SimpleNamespace(label="o", _label_raw="o_raw", op_identity="opaque")
    ops = [fake.FakeOp("a", "input", (), ("a",)), fake.FakeOp("loss", "neg", ("o_raw",), ("loss",))]
    trace = fake.build_trace(ops[:1], {"a": np.ones(2)})
    trace.ops["loss"] = ops[1]
    trace.payloads.update({"o_raw": np.ones(2), "loss": -np.ones(2)})
    adapter = SimpleNamespace(**{name: getattr(fake, name) for name in dir(fake)})
    adapter.recipe = lambda tr, op: None if op is blocked else fake.recipe(tr, op)
    adapter.can_replay = lambda tr, op: (
        "opaque region" if op is blocked else fake.can_replay(tr, op)
    )
    graph = graph_from_recipes(
        trace, adapter, [ops[0], blocked, ops[1]], trace.payloads.__getitem__
    )
    assert graph.node_for("o_raw").blocked == "opaque region"
    with pytest.raises(RunCapabilityUnavailableError, match="opaque region"):
        reverse_gradients(trace, adapter, graph, {"loss": np.ones(2)})


def test_engine_refuses_a_replayable_op_the_adapter_rejects() -> None:
    """``can_replay`` is asked for every reached op with parents."""

    trace = fake.build_trace(_ops(), _INPUTS)
    refusing = SimpleNamespace(
        **{name: getattr(fake, name) for name in ("vjp", "zeros_like", "add", "recipe")},
        can_replay=lambda trace, op: "random op without RNG support" if op.label == "d" else None,
    )
    with pytest.raises(RunCapabilityUnavailableError, match="random op"):
        reverse_gradients(trace, refusing, _graph(trace), {"loss": 1.0})


def test_engine_refuses_nodes_out_of_forward_order() -> None:
    trace = fake.build_trace(_ops(), _INPUTS)
    graph = _graph(trace)
    shuffled = ReplayGraph(nodes=tuple(reversed(graph.nodes)), value=graph.value)
    with pytest.raises(RunCapabilityUnavailableError, match="before it is produced"):
        reverse_gradients(trace, fake, shuffled, {"loss": 1.0})


def test_engine_refuses_a_derivative_with_the_wrong_arity() -> None:
    trace = fake.build_trace(_ops(), _INPUTS)

    def short_vjp(trace: Any, op: Any, parents: Any, cotangents: Any) -> tuple[Any, ...]:
        return fake.vjp(trace, op, parents, cotangents)[:1]

    broken = SimpleNamespace(
        **{name: getattr(fake, name) for name in ("can_replay", "zeros_like", "add")},
        vjp=short_vjp,
    )
    node = EngineNode(op=trace.ops["c"], parents=("a", "b"), outputs=("c",))
    graph = ReplayGraph(nodes=(node,), value=trace.payloads.__getitem__)
    with pytest.raises(RunCapabilityUnavailableError, match="2 parent occurrences"):
        reverse_gradients(trace, broken, graph, {"c": np.ones(3)})
