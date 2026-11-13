"""The neutral backward validator passes a consistent pass and fails corrupted ones."""

from __future__ import annotations

from typing import Any

import numpy as np

from tests.support import replay_fake_adapter as fake
from torchlens.backends._reverse_engine import ReplayGraph, graph_from_recipes, reverse_gradients
from torchlens.validation._backward_neutral import (
    RecordedBackward,
    check_backward_records,
)

_INPUTS = {"a": np.array([0.5, -1.25, 2.0]), "b": np.array([1.5, 0.75, -0.5])}
_OPS = [
    fake.FakeOp("a", "input", (), ("a",)),
    fake.FakeOp("b", "input", (), ("b",)),
    fake.FakeOp("c", "mul", ("a", "b"), ("c",)),
    fake.FakeOp("m", "mul", ("a", "a"), ("m",)),
    fake.FakeOp("d", "add", ("c", "m"), ("d",)),
    fake.FakeOp("s", "split2", ("d",), ("s", "s:1"), attrs={"index": 1}),
    fake.FakeOp("t", "sum", ("s:1",), ("t",), attrs={"axis": 0}),
    fake.FakeOp("loss", "scale", ("t",), ("loss",), literals=((("factor",), 2.0),)),
]


def _setup() -> tuple[fake.FakeTrace, ReplayGraph, RecordedBackward, dict[str, Any]]:
    """A pass recorded the way a native lane journals it, and the true leaf gradients."""

    trace = fake.build_trace(_OPS, _INPUTS)
    ops = [op for op in trace.ops.values() if op.op_identity != "input"]
    graph = graph_from_recipes(trace, fake, ops, trace.payloads.__getitem__)
    grads = reverse_gradients(trace, fake, graph, {"loss": 1.0})
    calls = {}
    for node in graph.nodes:
        incoming = tuple(grads.get(slot) for slot in node.outputs)
        filled = [
            fake.zeros_like(trace.payloads[s]) if g is None else g
            for s, g in zip(node.outputs, incoming, strict=True)
        ]
        outgoing = fake.vjp(trace, node.op, [trace.payloads[p] for p in node.parents], filled)
        calls[node.label] = (incoming, tuple(outgoing))
    recorded = RecordedBackward(calls=calls, grads=dict(grads))
    return trace, graph, recorded, {"a": grads["a"], "b": grads["b"]}


def _with_call(
    recorded: RecordedBackward, label: str, outgoing: tuple[Any, ...]
) -> RecordedBackward:
    calls = dict(recorded.calls)
    calls[label] = (calls[label][0], outgoing)
    return RecordedBackward(calls=calls, grads=recorded.grads)


def test_consistent_pass_passes_all_three_checks() -> None:
    trace, graph, recorded, leaves = _setup()
    result = check_backward_records(trace, fake, graph, recorded, leaves)
    assert result.passed, result.findings
    assert result.nodes_checked == len(graph.nodes)
    assert result.sums_checked >= 5
    assert result.leaves_checked == 2


def test_corrupted_edge_fails_node_and_sum_checks() -> None:
    trace, graph, recorded, leaves = _setup()
    to_a, to_b = recorded.calls["c"][1]
    corrupted = _with_call(recorded, "c", (to_a, to_b + 0.25))
    result = check_backward_records(trace, fake, graph, corrupted, leaves)
    assert not result.passed
    found = {(item.check, item.op_label) for item in result.findings}
    assert ("node", "c") in found
    assert ("sum", "b") in found


def test_swapped_edge_fails() -> None:
    trace, graph, recorded, leaves = _setup()
    to_a, to_b = recorded.calls["c"][1]
    swapped = _with_call(recorded, "c", (to_b, to_a))
    result = check_backward_records(trace, fake, graph, swapped, leaves)
    assert not result.passed
    assert any(item.check == "node" and item.op_label == "c" for item in result.findings)


def test_corrupted_intermediate_gradient_fails_the_sum_check() -> None:
    trace, graph, recorded, leaves = _setup()
    grads = dict(recorded.grads)
    grads["d"] = grads["d"] * 2.0
    result = check_backward_records(
        trace, fake, graph, RecordedBackward(recorded.calls, grads), leaves
    )
    assert {(item.check, item.op_label) for item in result.findings} >= {("sum", "d")}


def test_corrupted_leaf_fails_the_leaf_check() -> None:
    trace, graph, recorded, leaves = _setup()
    wrong = {**leaves, "a": leaves["a"] + 1.0}
    result = check_backward_records(trace, fake, graph, recorded, wrong)
    assert [(item.check, item.op_label) for item in result.findings] == [("leaf", "a")]


def test_missing_leaf_and_empty_pass_fail() -> None:
    trace, graph, recorded, leaves = _setup()
    result = check_backward_records(trace, fake, graph, recorded, {**leaves, "z": np.ones(1)})
    assert ("leaf", "z") in {(item.check, item.op_label) for item in result.findings}
    empty = check_backward_records(trace, fake, graph, RecordedBackward({}, {}), {})
    assert not empty.passed


def test_recorded_call_on_an_undescribed_op_is_a_finding() -> None:
    trace, graph, recorded, leaves = _setup()
    blocked = ReplayGraph(
        nodes=tuple(
            node if node.label != "c" else type(node)(node.op, (), node.outputs, "opaque region")
            for node in graph.nodes
        ),
        value=graph.value,
    )
    result = check_backward_records(trace, fake, blocked, recorded, leaves)
    assert any("opaque region" in item.detail for item in result.findings)
