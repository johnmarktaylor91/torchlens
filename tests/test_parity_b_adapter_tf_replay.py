"""TensorFlow replay adapter proofs over the TensorFlow conformance roster.

* ``forward`` equals the saved payload on every replayable op of every roster model;
* ``vjp`` composed over the whole graph equals ``tf.GradientTape.gradient``;
* every op ``can_replay`` refuses is reported by op type with its reason.

Set ``TL_ADAPTER_TF_REPORT`` to a file path to also write the refused-op list there.
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Any

import numpy as np
import pytest

from tests.conformance._rosters import tf as tf_roster

tf = pytest.importorskip("tensorflow")

import torchlens as tl  # noqa: E402
from torchlens.backends._replay_adapters import get_adapter  # noqa: E402
from torchlens.backends._validation_shared import (  # noqa: E402
    float_replay_tolerances,
    ops_by_label,
)

pytestmark = [pytest.mark.tf_backend, pytest.mark.rare]

ADAPTER = get_adapter("tf")
_ROWS = {row.family: row for row in tf_roster.roster()}
_ECOSYSTEM = set(tf_roster.ECOSYSTEM_FAMILIES)
_CAPTURES: dict[str, tuple[Any, Any, Any]] = {}
_REFUSED: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))

#: Roster op types that are expected to be refused, with the reason prefix.
_EXPECTED_REFUSALS = {"ReadVariableOp": "resource read"}


def _row_params() -> list[Any]:
    return [
        pytest.param(name, marks=[pytest.mark.slow] if name in _ECOSYSTEM else [], id=name)
        for name in _ROWS
    ]


def _capture(name: str) -> tuple[Any, Any, Any]:
    """``(model, example input, trace)`` for one roster row, captured once per session."""

    if name not in _CAPTURES:
        model, example = _ROWS[name].builder()
        _CAPTURES[name] = (model, example, tl.trace(model, example, backend="tf"))
    return _CAPTURES[name]


def _unique_ops(trace: Any) -> list[Any]:
    """Every recorded op once, in graph (execution) order."""

    return list(trace._layer_list)


def _loss_weights(value: Any) -> Any:
    """Fixed non-uniform loss weights shaped like ``value`` (0.5 to 1.5, row-major).

    A plain sum is a degenerate loss for any model ending in a normalization:
    the sum of a layer-normalized row is constant, so every gradient below it is
    zero and the comparison would only see rounding noise. The weights depend on
    the shape alone, so a value returned twice gets the same weights each time.
    """

    size = int(tf.size(value))
    weights = tf.linspace(0.5, 1.5, size) if size > 1 else tf.ones((size,))
    return tf.cast(tf.reshape(weights, tf.shape(value)), value.dtype)


def _output_seeds(trace: Any) -> dict[str, Any]:
    """Loss weights per model-output op (summed when one value is returned more than once).

    The tape's loss is the weighted sum over the flattened model outputs
    (:func:`_loss_weights`), so its seed on each output op is those weights,
    times the number of times that op's value is returned.
    """

    ops = ops_by_label(trace)
    seeds: dict[str, Any] = {}
    for name in trace.output_layers:
        op = ops[name]
        weights = _loss_weights(op.out)
        seeds[op.label] = (
            weights if op.label not in seeds else ADAPTER.add(seeds[op.label], weights)
        )
    return seeds


def _parent_values(trace: Any, recipe: Any) -> list[Any]:
    ops = ops_by_label(trace)
    return [ops[occurrence.parent_label].out for occurrence in recipe.inputs]


def _write_report() -> None:
    path = os.environ.get("TL_ADAPTER_TF_REPORT")
    if not path:
        return
    with open(path, "w", encoding="utf-8") as handle:
        for row, by_type in sorted(_REFUSED.items()):
            for op_type, reasons in sorted(by_type.items()):
                handle.write(f"{row}\t{op_type}\t{' | '.join(sorted(reasons))}\n")


@pytest.mark.parametrize("name", _row_params())
def test_forward_equals_saved_payload_on_every_replayable_op(name: str) -> None:
    """Replaying each op from its parents' saved payloads reproduces its own payload."""

    _, _, trace = _capture(name)
    replayed = 0
    for op in _unique_ops(trace):
        reason = ADAPTER.can_replay(trace, op)
        if reason is not None:
            _REFUSED[name][str(op.func_name)].add(reason)
            continue
        recipe = ADAPTER.recipe(trace, op)
        (value,) = ADAPTER.forward(trace, op, _parent_values(trace, recipe))
        assert ADAPTER.close(op.out, value), f"{name}:{op.label} ({op.func_name})"
        replayed += 1
    _write_report()
    assert replayed > 0
    for op_type, reasons in _REFUSED[name].items():
        if op_type in {"input", "constant"}:
            continue
        prefix = _EXPECTED_REFUSALS.get(op_type)
        assert prefix is not None, f"{name}: unexpected refusal of {op_type}: {reasons}"
        assert all(reason.startswith(prefix) for reason in reasons)


def _reverse_accumulate(trace: Any) -> dict[str, Any]:
    """Cotangent per op label from ``vjp`` composed in reverse graph order (seed: ones)."""

    cotangent = _output_seeds(trace)
    ops = _unique_ops(trace)
    assert cotangent, "no model-output op to seed"
    for op in reversed(ops):
        seed = cotangent.get(op.label)
        recipe = ADAPTER.recipe(trace, op)
        if seed is None or recipe is None or not recipe.inputs:
            continue
        reason = ADAPTER.can_replay(trace, op)
        assert reason is None, f"{op.label} blocks the reverse chain: {reason}"
        grads = ADAPTER.vjp(trace, op, _parent_values(trace, recipe), [seed])
        assert len(grads) == len(recipe.inputs)
        for occurrence, grad in zip(recipe.inputs, grads, strict=True):
            if grad is None:
                continue
            label = occurrence.parent_label
            label = ops_by_label(trace)[label].label
            previous = cotangent.get(label)
            cotangent[label] = grad if previous is None else ADAPTER.add(previous, grad)
    return cotangent


def _tape_gradients(model: Any, example: Any) -> tuple[Any, list[Any], list[Any]]:
    """``(input gradient or None, variables, variable gradients)`` from one tape.

    The input gradient is taken only for a single floating input tensor (a
    token-id dict, for example, has none).
    """

    inputs = tf.nest.map_structure(tf.convert_to_tensor, example)
    single_float = tf.is_tensor(inputs) and inputs.dtype.is_floating
    variables = list(getattr(model, "trainable_variables", []) or [])
    with tf.GradientTape() as tape:
        if single_float:
            tape.watch(inputs)
        outputs = model(inputs)
        flat = tf.nest.flatten(outputs)
        loss = tf.add_n([tf.reduce_sum(item * _loss_weights(item)) for item in flat])
    grads = tape.gradient(loss, [inputs if single_float else tf.zeros(()), *variables])
    return (grads[0] if single_float else None), variables, list(grads[1:])


def _gap(expected: Any, actual: Any) -> str:
    """Size of a mismatch, for the assertion message."""

    want, got = np.asarray(expected, dtype=np.float64), np.asarray(actual, dtype=np.float64)
    diff = np.abs(want - got)
    return f"max abs diff {diff.max():.3e}, max abs expected {np.abs(want).max():.3e}"


def _dense(value: Any) -> Any:
    """``tape.gradient`` gives ``IndexedSlices`` for an embedding lookup; densify it."""

    return tf.convert_to_tensor(value) if isinstance(value, tf.IndexedSlices) else value


def _composed_close(expected: Any, actual: Any) -> bool:
    """The validator's ``close``, or its relative band taken at the tensor's scale.

    A gradient composed backwards over a whole graph sums float32 contributions
    in a different order from the tape. Over the deep ecosystem graphs (1341
    recorded ops for efficientnet_b0) that moves elements near zero by more than
    their own relative band while the largest gap stays a few float32 ULPs of
    the tensor's largest element. So the per-element replay band applies first,
    and otherwise the largest gap must sit inside that same relative band times
    the largest expected magnitude. This is this test's criterion only; the
    validator and its band are unchanged.
    """

    want, got = _dense(expected), _dense(actual)
    if ADAPTER.close(want, got):
        return True
    want_array, got_array = ADAPTER.to_host(want), ADAPTER.to_host(got)
    if want_array.shape != got_array.shape or want_array.dtype != got_array.dtype:
        return False
    if not np.issubdtype(want_array.dtype, np.floating):
        return False
    rtol, _ = float_replay_tolerances(np.finfo(want_array.dtype))
    gap = np.abs(want_array.astype(np.float64) - got_array.astype(np.float64))
    scale = float(np.abs(want_array.astype(np.float64)).max())
    return bool(np.isfinite(gap).all() and gap.max() <= rtol * scale)


@pytest.mark.parametrize("name", _row_params())
def test_vjp_composed_over_the_graph_equals_tape_gradient(name: str) -> None:
    """Reverse composition of per-op ``vjp`` equals ``tape.gradient`` for input and weights.

    Compared with :func:`_composed_close`; the session test holds the config rows
    to the validator's per-element band.
    """

    from torchlens.backends.tf._tensor_probe import _tensor_ref_key

    model, example, trace = _capture(name)
    cotangent = _reverse_accumulate(trace)
    input_grad, variables, variable_grads = _tape_gradients(model, example)
    if input_grad is not None:
        (input_op,) = [op for op in _unique_ops(trace) if bool(getattr(op, "is_input", False))]
        got = cotangent[input_op.label]
        assert _composed_close(input_grad, got), f"{name}: {_gap(input_grad, got)}"
    reads: dict[object, list[str]] = defaultdict(list)
    for op in _unique_ops(trace):
        recipe = ADAPTER.recipe(trace, op)
        if recipe is not None and op.func_name == "ReadVariableOp":
            capture = _capture_of(trace, op)
            reads[capture.inputs[0].ref_key].append(op.label)
    for variable, expected in zip(variables, variable_grads, strict=True):
        labels = reads.get(_tensor_ref_key(variable.handle), [])
        assert labels, f"{name}: no read of {variable.name}"
        total = None
        for label in labels:
            grad = cotangent.get(label)
            if grad is not None:
                total = grad if total is None else ADAPTER.add(total, grad)
        assert total is not None, f"{name}:{variable.name}"
        assert _composed_close(expected, total), (
            f"{name}:{variable.name}: {_gap(_dense(expected), total)}"
        )


def _capture_of(trace: Any, op: Any) -> Any:
    return {c.label_raw: c for c in trace._tf_op_captures}[op._label_raw]


def _extra_ops_function(v: Any) -> Any:
    """Exercise the adapter's extra kernels, a repeated parent and literal inputs."""

    squared = v * v
    sliced = tf.strided_slice(squared, [0, 1], [2, 3], [1, 1])
    stacked = tf.stack([sliced, tf.identity(sliced)], axis=1)
    gelu = tf.nn.gelu(stacked)
    held = tf.stop_gradient(gelu) + gelu
    sizes = tf.cast(tf.size(held) + tf.rank(held) + tf.shape(held)[0], tf.float32)
    return tf.reduce_sum(held, axis=[1, 2]) * sizes


def test_extra_kernels_and_repeated_parent_replay_and_differentiate() -> None:
    """The adapter's own kernels replay to the payload and ``vjp`` matches the tape."""

    example = tf.constant(np.arange(8, dtype=np.float32).reshape(2, 4) / 7.0 - 0.3)
    trace = tl.trace(_extra_ops_function, example, backend="tf")
    seen_types = set()
    for op in _unique_ops(trace):
        if ADAPTER.can_replay(trace, op) is not None:
            continue
        seen_types.add(str(op.func_name))
        recipe = ADAPTER.recipe(trace, op)
        (value,) = ADAPTER.forward(trace, op, _parent_values(trace, recipe))
        assert ADAPTER.close(op.out, value), op.label
    assert {"Identity", "StopGradient", "StridedSlice", "Pack"} <= seen_types
    mul = next(op for op in _unique_ops(trace) if op.func_name == "Mul")
    mul_recipe = ADAPTER.recipe(trace, mul)
    assert len(mul_recipe.inputs) == 2
    assert mul_recipe.inputs[0].parent_label == mul_recipe.inputs[1].parent_label
    grads = ADAPTER.vjp(trace, mul, _parent_values(trace, mul_recipe), [tf.ones_like(mul.out)])
    assert all(ADAPTER.close(np.asarray(example), g) for g in grads)
    cotangent = _reverse_accumulate(trace)
    with tf.GradientTape() as tape:
        tape.watch(example)
        loss = tf.reduce_sum(_extra_ops_function(example))
    (input_op,) = [op for op in _unique_ops(trace) if bool(getattr(op, "is_input", False))]
    assert ADAPTER.close(tape.gradient(loss, example), cotangent[input_op.label])


def test_forward_is_differentiable_by_tensorflow() -> None:
    """A tape around a replayed two-op cone gives TensorFlow's gradient of that cone."""

    _, example, trace = _capture("chain_branch_function")
    ops = [op for op in _unique_ops(trace) if ADAPTER.can_replay(trace, op) is None]
    source = tf.convert_to_tensor(example)
    values = {op.label: op.out for op in _unique_ops(trace)}
    with tf.GradientTape() as tape:
        tape.watch(source)
        input_op = next(op for op in _unique_ops(trace) if bool(getattr(op, "is_input", False)))
        values[input_op.label] = source
        for op in ops:
            recipe = ADAPTER.recipe(trace, op)
            parents = [values[ops_by_label(trace)[o.parent_label].label] for o in recipe.inputs]
            (values[op.label],) = ADAPTER.forward(trace, op, parents)
        out = tf.reduce_sum(values[ops[-1].label])
    grad = tape.gradient(out, source)
    assert grad is not None and float(tf.reduce_sum(tf.abs(grad))) > 0.0


def test_random_ops_are_refused_until_rng_capture_lands() -> None:
    """A random op names the rng lane; ops downstream of it still replay."""

    def noisy(v: Any) -> Any:
        return tf.nn.relu(v + tf.random.normal(tf.shape(v)))

    trace = tl.trace(noisy, tf.ones((2, 3)), backend="tf")
    reasons = {str(op.func_name): ADAPTER.can_replay(trace, op) for op in _unique_ops(trace)}
    random_types = [name for name in reasons if name.startswith("Random")]
    assert random_types, reasons
    assert all(reasons[name].startswith("random op") for name in random_types)
    assert reasons["Relu"] is None
