"""The Paddle replay adapter replays every roster op and differentiates like Paddle.

Proofs (wave B, lane b-adapter-paddle): ``forward`` equals the saved payload on
every replayable op of every Paddle roster row; ``vjp`` composed over the
recorded graph equals Paddle's own autograd of the whole forward; every op
``can_replay`` refuses is reported by name with its reason (written to
``$TL_PARITY_B_REFUSALS`` as JSON lines when set).
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pytest

paddle = pytest.importorskip("paddle")

import torchlens as tl  # noqa: E402
from tests.conformance._rosters import paddle as paddle_roster  # noqa: E402
from torchlens._utils._record_group_table import record_member  # noqa: E402
from torchlens.backends._replay_adapters import ReplayRefusal, get_adapter  # noqa: E402
from torchlens.backends._validation_shared import ops_by_label  # noqa: E402

pytestmark = [pytest.mark.backend_paddle, pytest.mark.rare]

adapter: Any = get_adapter("paddle")
_ROWS = {row.family: row for row in paddle_roster.roster()}
_FAMILIES = sorted(_ROWS)


def _args(example: Any) -> tuple[Any, ...]:
    """The positional arguments of a roster example."""

    return example if isinstance(example, tuple) else (example,)


def _trace(family: str) -> tuple[Any, Any, tuple[Any, ...]]:
    """Build and capture one roster row; return the model, trace and args."""

    model, example = _ROWS[family].builder()
    args = _args(example)
    return model, tl.trace(model, args, backend="paddle"), args


def _report(family: str, trace: Any) -> dict[str, str]:
    """Refused non-input ops of a trace by label, appended to the refusal report."""

    refused = {
        str(op._label_raw): f"{op.func_name}: {reason}"
        for op in trace._layer_list
        if not op.is_input and (reason := adapter.can_replay(trace, op)) is not None
    }
    path = os.environ.get("TL_PARITY_B_REFUSALS")
    if path:
        with open(path, "a", encoding="utf-8") as handle:
            for label, text in refused.items():
                handle.write(json.dumps({"family": family, "op": label, "why": text}) + "\n")
    return refused


def _parents(trace: Any, site: Any) -> list[Any]:
    """Saved parent payloads, one per input occurrence."""

    by_label = ops_by_label(trace)
    return [by_label[label].out for label, _path in adapter._occurrences(site.capture)]


def _sites(trace: Any) -> list[tuple[Any, Any]]:
    """Replayable ``(first op, call site)`` pairs in forward order (each call once)."""

    sites = []
    for op in trace._layer_list:
        if op.is_input or adapter.can_replay(trace, op) is not None:
            continue
        site = adapter._site(trace, op)
        if site.slot_labels[0] == op._label_raw:
            sites.append((op, site))
    return sites


@pytest.mark.parametrize("family", _FAMILIES)
def test_forward_equals_saved_payload_on_every_replayable_op(family: str) -> None:
    """Each replayable call reproduces every output slot's saved payload."""

    _model, trace, _args_ = _trace(family)
    _report(family, trace)
    by_label = ops_by_label(trace)
    sites = _sites(trace)
    assert sites, family
    for op, site in sites:
        outputs = adapter.forward(trace, op, _parents(trace, site))
        assert len(outputs) == len(site.slot_labels)
        for label, value in zip(site.slot_labels, outputs, strict=True):
            assert adapter.close(by_label[label].out, value), f"{family}:{label}"


def _seed_outputs(model: Any, args: tuple[Any, ...]) -> tuple[list[Any], list[Any], list[Any]]:
    """Run the model eagerly on fresh leaves; return leaves, float outputs, seeds."""

    leaves = []
    for arg in args:
        leaf = paddle.to_tensor(arg.numpy())
        leaf.stop_gradient = False
        leaves.append(leaf)
    result = model(*leaves)
    flat = list(result) if isinstance(result, (tuple, list)) else [result]
    outs = [out for out in flat if paddle.is_floating_point(out)]
    seeds = [paddle.to_tensor(np.linspace(0.5, 1.5, out.size, dtype="float32").reshape(out.shape))
             for out in outs]  # fmt: skip
    return leaves, outs, seeds


def _output_labels(trace: Any, outs: list[Any]) -> list[str]:
    """Raw labels of the trace ops whose saved payloads are the model outputs, in order."""

    finals = [op for op in trace._layer_list
              if record_member(op, "is_output_parent", False)
              or record_member(op, "is_final_output", False)]  # fmt: skip
    labels = []
    for out in outs:
        host = out.numpy()
        match = next(
            op
            for op in finals
            if op.out.shape == out.shape and np.array_equal(op.out.numpy(), host)
        )
        labels.append(str(match._label_raw))
    return labels


def _reverse(trace: Any, cot: dict[str, Any]) -> None:
    """Accumulate cotangents back through every replayable call (in place)."""

    by_label = ops_by_label(trace)
    for op, site in reversed(_sites(trace)):
        slot_cots = [cot.get(label) for label in site.slot_labels]
        if all(value is None for value in slot_cots):
            continue
        occurrences = adapter._occurrences(site.capture)
        parents = [by_label[label].out for label, _path in occurrences]
        grads = adapter.vjp(trace, op, parents, slot_cots)
        assert len(grads) == len(occurrences)
        for (label, _path), grad in zip(occurrences, grads, strict=True):
            if grad is not None:
                cot[label] = grad if label not in cot else adapter.add(cot[label], grad)


@pytest.mark.parametrize("family", _FAMILIES)
def test_vjp_composed_over_the_graph_equals_paddle_autograd(family: str) -> None:
    """Reverse accumulation of per-op ``vjp`` equals ``paddle.grad`` of the forward."""

    model, trace, args = _trace(family)
    refused = _report(family, trace)
    if refused:
        pytest.skip(f"{family}: refused ops on the graph: {sorted(refused.values())}")
    leaves, outs, seeds = _seed_outputs(model, args)
    expected = paddle.grad(outs, leaves, grad_outputs=seeds, allow_unused=True)
    cot = dict(zip(_output_labels(trace, outs), seeds, strict=True))
    _reverse(trace, cot)
    inputs = [op for op in trace._layer_list if op.is_input]
    assert len(inputs) == len(leaves)
    for op, reference in zip(inputs, expected, strict=True):
        got = cot.get(str(op._label_raw))
        if reference is None:
            assert got is None, family
            continue
        assert got is not None, f"{family}: no gradient reached {op._label_raw}"
        assert adapter.close(reference, got), f"{family}: {op._label_raw}"


def test_repeated_parent_gets_one_cotangent_per_occurrence() -> None:
    """``v * v`` replays with two occurrences, each with its own cotangent."""

    x = paddle.to_tensor(np.array([1.0, -2.0, 3.0], dtype="float32"))
    trace = tl.trace(lambda v: v * v, (x,), backend="paddle")
    op = next(op for op in trace._layer_list if not op.is_input)
    site = adapter._site(trace, op)
    assert [label for label, _ in adapter._occurrences(site.capture)] == [
        str(next(o for o in trace._layer_list if o.is_input)._label_raw)
    ] * 2
    cot = paddle.to_tensor(np.array([1.0, 0.5, 2.0], dtype="float32"))
    first, second = adapter.vjp(trace, op, [x, x], [cot])
    np.testing.assert_array_equal(first.numpy(), (cot * x).numpy())
    np.testing.assert_array_equal(second.numpy(), (cot * x).numpy())


def test_multi_output_call_replays_atomically() -> None:
    """``paddle.split`` emits several ops from one call; replay returns every slot."""

    x = paddle.to_tensor(np.arange(12, dtype="float32").reshape(4, 3))
    trace = tl.trace(lambda v: [part * 2.0 for part in paddle.split(v, 2)], (x,), backend="paddle")
    split_ops = [op for op in trace._layer_list if op.func_name == "split"]
    assert len(split_ops) == 2
    site = adapter._site(trace, split_ops[1])
    assert site.slot_labels == tuple(str(op._label_raw) for op in split_ops)
    outputs = adapter.forward(trace, split_ops[0], _parents(trace, site))
    for op, value in zip(split_ops, outputs, strict=True):
        assert adapter.close(op.out, value)
    cots = [None, paddle.ones_like(outputs[1])]
    (grad,) = adapter.vjp(trace, split_ops[0], _parents(trace, site), cots)
    np.testing.assert_array_equal(
        grad.numpy(), np.repeat([[0.0], [1.0]], [2, 2], axis=0) * np.ones((4, 3))
    )


def test_training_dropout_is_refused_as_random_and_eval_dropout_replays() -> None:
    """Dropout that draws random numbers waits for the rng lane; p=0 replays."""

    x = paddle.to_tensor(np.ones((4, 4), dtype="float32"))
    functional = paddle.nn.functional
    live = tl.trace(lambda v: functional.dropout(v, p=0.5, training=True), (x,), backend="paddle")
    still = tl.trace(lambda v: functional.dropout(v, p=0.5, training=False) + 1.0, (x,),
                     backend="paddle")  # fmt: skip
    live_op = next(op for op in live._layer_list if "dropout" in op.func_name)
    assert "random op" in str(adapter.can_replay(live, live_op))
    still_ops = [op for op in still._layer_list if not op.is_input]
    assert all(adapter.can_replay(still, op) is None for op in still_ops)


def test_intervened_call_replays_the_function_and_has_no_recipe() -> None:
    """A user-replaced value replays to the pre-hook output and gets no data recipe."""

    x = paddle.to_tensor(np.array([[1.0, -1.0, 2.0]], dtype="float32"))
    trace = tl.trace(
        lambda v: paddle.nn.functional.relu(v) * 3.0,
        (x,),
        backend="paddle",
        intervene=tl.when(tl.func("functional.relu"), tl.zero_ablate()),
    )
    relu = next(op for op in trace._layer_list if op.func_name == "functional.relu")
    site = adapter._site(trace, relu)
    assert adapter.can_replay(trace, relu) is None
    (value,) = adapter.forward(trace, relu, _parents(trace, site))
    np.testing.assert_array_equal(value.numpy(), np.maximum(x.numpy(), 0.0))
    assert adapter.recipe(trace, relu) is None
    assert "intervened" in str(adapter.recipe_refusal(trace, relu))


def test_array_helpers_use_paddle_and_the_validator_comparison() -> None:
    """``zeros_like``, ``add``, ``to_host`` and ``close`` behave as the contract says."""

    x = paddle.to_tensor(np.array([1.0, 2.0], dtype="float32"))
    assert adapter.to_host(adapter.zeros_like(x)).tolist() == [0.0, 0.0]
    assert adapter.to_host(adapter.add(x, x)).tolist() == [2.0, 4.0]
    assert adapter.close(x, x + 0.0)
    assert not adapter.close(x, x + 1.0)
    assert not adapter.close(x, paddle.cast(x, "float64"))


def test_contract_misuse_raises_the_internal_replay_refusal() -> None:
    """Wrong arity and refused ops raise ``ReplayRefusal``, never a coded error."""

    x = paddle.to_tensor(np.array([1.0, 2.0], dtype="float32"))
    trace = tl.trace(lambda v: paddle.nn.functional.relu(v), (x,), backend="paddle")
    relu = next(op for op in trace._layer_list if not op.is_input)
    source = next(op for op in trace._layer_list if op.is_input)
    with pytest.raises(ReplayRefusal, match="arity"):
        adapter.forward(trace, relu, [])
    with pytest.raises(ReplayRefusal, match="arity"):
        adapter.vjp(trace, relu, [x], [])
    with pytest.raises(ReplayRefusal, match="cannot_replay"):
        adapter.forward(trace, source, [])
