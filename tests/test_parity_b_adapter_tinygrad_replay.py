"""tinygrad replay adapter: ``forward`` equals the saved payloads, ``vjp`` chains equal backward."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import torchlens as tl
from torchlens.backends._replay_adapters import get_adapter

pytest.importorskip("tinygrad")
from tinygrad import Tensor  # noqa: E402

from tests.conformance._rosters import tinygrad as roster  # noqa: E402

pytestmark = [pytest.mark.backend_tinygrad, pytest.mark.rare]

adapter = get_adapter("tinygrad")

ROWS = roster.config_built_rows() + roster.random_init_rows()
SOURCE_PREFIX = "input source"


def _capture(model: Any, value: Any) -> Any:
    return tl.trace(model, value, backend="tinygrad")


def _ops_by_label(trace: Any) -> dict[str, Any]:
    return {op.label: op for op in trace._layer_list}


def _parent_values(trace: Any, op: Any) -> list[Any]:
    """Saved parent payloads, one per recipe input occurrence."""

    recipe = adapter.recipe(trace, op)
    assert recipe is not None
    ops = _ops_by_label(trace)
    return [ops[occ.parent_label].out for occ in recipe.inputs]


def _refusals(trace: Any) -> list[tuple[str, str, str]]:
    return [
        (op.label, str(op.func_name), reason)
        for op in trace._layer_list
        if (reason := adapter.can_replay(trace, op)) is not None
        and not reason.startswith(SOURCE_PREFIX)
    ]


@pytest.mark.parametrize("row", ROWS, ids=lambda row: row.family)
def test_forward_equals_saved_payload_on_every_roster_op(row: Any) -> None:
    """Every replayable op of every roster model replays to its saved payload."""

    model, value = row.builder()
    trace = _capture(model, value)
    replayed = 0
    for op in trace._layer_list:
        if adapter.can_replay(trace, op) is not None:
            continue
        (out,) = adapter.forward(trace, op, _parent_values(trace, op))
        assert adapter.close(op.out, out), op.label
        replayed += 1
    assert replayed > 0
    assert _refusals(trace) == []


def test_refused_ops_report() -> None:
    """Write the refused roster ops by name (``PARITY_B_TG_REPORT``); the list is empty today."""

    report: dict[str, list[tuple[str, str, str]]] = {}
    for row in ROWS:
        model, value = row.builder()
        report[row.family] = _refusals(_capture(model, value))
    path = os.environ.get("PARITY_B_TG_REPORT")
    if path:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=1)
    assert all(not rows for rows in report.values()), report


def _output_seed_labels(trace: Any) -> list[str]:
    ops = _ops_by_label(trace)
    return [label for label in trace.output_layers if label in ops]


def _chain_input_grad(trace: Any) -> np.ndarray:
    """Compose ``vjp`` in reverse over the trace; seed ones on every output."""

    ops = _ops_by_label(trace)
    grads: dict[str, Any] = {}
    for label in _output_seed_labels(trace):
        seed = Tensor.ones(*ops[label].out.shape, dtype=ops[label].out.dtype)
        grads[label] = seed if label not in grads else adapter.add(grads[label], seed)
    for op in reversed(trace._layer_list):
        cotangent = grads.get(op.label)
        if cotangent is None or adapter.can_replay(trace, op) is not None:
            continue
        recipe = adapter.recipe(trace, op)
        if not recipe.inputs:
            continue
        values = [ops[occ.parent_label].out for occ in recipe.inputs]
        for occ, grad in zip(recipe.inputs, adapter.vjp(trace, op, values, (cotangent,))):
            if grad is None:
                continue
            prior = grads.get(occ.parent_label)
            total = grad if prior is None else adapter.add(prior, grad)
            grads[occ.parent_label] = total.realize()
    (input_op,) = [op for op in trace._layer_list if op.type == "input"]
    return adapter.to_host(grads[input_op.label])


def _backward_input_grad(model: Any, value: Any) -> np.ndarray:
    x = Tensor(value.numpy(), requires_grad=True)
    out = model(x)
    leaves = out if isinstance(out, (tuple, list)) else (out,)
    loss = leaves[0].sum()
    for leaf in leaves[1:]:
        loss = loss + leaf.sum()
    loss.backward()
    return x.grad.numpy()


@pytest.mark.parametrize("row", ROWS, ids=lambda row: row.family)
def test_vjp_chain_equals_tensor_backward(row: Any) -> None:
    """Reverse composition of per-op ``vjp`` equals ``Tensor.backward`` at the input."""

    model, value = row.builder()
    trace = _capture(model, value)
    chained = _chain_input_grad(trace)
    expected = _backward_input_grad(model, value)
    np.testing.assert_allclose(chained, expected, rtol=1e-5, atol=1e-6)


def test_vjp_keeps_repeated_parent_occurrences_apart() -> None:
    """``x * x`` gives one cotangent per occurrence; their sum is the full derivative."""

    trace = _capture(lambda v: v * v, Tensor([1.0, -2.0, 3.0]))
    (mul,) = [op for op in trace._layer_list if op.type == "mul"]
    values = _parent_values(trace, mul)
    assert len(values) == 2 and values[0] is values[1]
    cot = Tensor([1.0, 1.0, 1.0])
    first, second = adapter.vjp(trace, mul, values, (cot,))
    np.testing.assert_allclose(adapter.to_host(first), [1.0, -2.0, 3.0])
    np.testing.assert_allclose(adapter.to_host(second), [1.0, -2.0, 3.0])


def test_forward_is_differentiable_by_tinygrad() -> None:
    """A replayed op stays in tinygrad's graph: its gradient reaches the given parents."""

    trace = _capture(lambda v: (v + 1.0) * v, Tensor([1.0, -2.0, 3.0]))
    (mul,) = [op for op in trace._layer_list if op.type == "mul"]
    values = [Tensor(adapter.to_host(v), requires_grad=True) for v in _parent_values(trace, mul)]
    (out,) = adapter.forward(trace, mul, values)
    out.sum().backward()
    grads = [adapter.to_host(v.grad) for v in values]
    np.testing.assert_allclose(grads[0], adapter.to_host(values[1]))
    np.testing.assert_allclose(grads[1], adapter.to_host(values[0]))


def test_random_op_is_refused_with_a_reason() -> None:
    """A captured UOp drawn by tinygrad's Threefry generator is refused until RNG replay exists."""

    threefry = next(u for u in Tensor.rand(3).uop.toposort() if u.op.name == "THREEFRY")
    capture = SimpleNamespace(
        label_raw="r",
        uop=threefry,
        op_name="THREEFRY",
        parent_labels=("p",),
        parent_arg_positions=((0, "p"),),
        payload_snapshot=None,
    )
    op = SimpleNamespace(
        _label_raw="r", label="r", parents=("p",), parent_arg_positions={"args": {0: "p"}}
    )
    parent = SimpleNamespace(_label_raw="p", label="p", parents=())
    trace = SimpleNamespace(_layer_list=[parent, op], tinygrad_uop_captures=(capture,))
    reason = adapter.can_replay(trace, op)
    assert reason is not None and "random" in reason
    assert adapter.recipe(trace, op) is None
