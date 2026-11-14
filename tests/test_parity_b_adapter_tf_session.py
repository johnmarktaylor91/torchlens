"""The TensorFlow adapter drives the shared replay session and pullback step.

A whole roster graph replays through :class:`ReplaySession` from the saved
payloads of the ops the adapter refuses (inputs, constants, resource reads);
every replayed value equals the saved payload, and :func:`pullback_step` over
the same steps in reverse gives ``tf.GradientTape.gradient`` for the input.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.conformance._rosters import tf as tf_roster

tf = pytest.importorskip("tensorflow")

import torchlens as tl  # noqa: E402
from torchlens.backends._replay_adapters import (  # noqa: E402
    ReplayPlan,
    ReplaySession,
    ReplayStep,
    get_adapter,
    merge_grads,
    pullback_step,
)
from torchlens.backends._validation_shared import ops_by_label  # noqa: E402

pytestmark = [pytest.mark.tf_backend, pytest.mark.rare]

ADAPTER = get_adapter("tf")
_ROWS = {row.family: row for row in tf_roster.roster()}
_CONFIG_ROWS = [name for name in _ROWS if name not in set(tf_roster.ECOSYSTEM_FAMILIES)]


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


def _plan(trace: Any) -> tuple[list[ReplayStep], dict[str, Any]]:
    """Steps for every replayable op; saved payloads for every refused op."""

    steps: list[ReplayStep] = []
    initial: dict[str, Any] = {}
    for op in _unique_ops(trace):
        if ADAPTER.can_replay(trace, op) is None:
            steps.append(ReplayStep.from_recipe(op, ADAPTER.recipe(trace, op)))
        else:
            initial[op.label] = op.out
    return steps, initial


@pytest.mark.parametrize("name", _CONFIG_ROWS)
def test_session_replays_the_graph_and_pullback_matches_the_tape(name: str) -> None:
    """Forward through the session reproduces payloads; reverse pullback equals the tape."""

    model, example = _ROWS[name].builder()
    trace = tl.trace(model, example, backend="tf")
    steps, initial = _plan(trace)
    assert steps
    with ReplaySession(ADAPTER, trace, ReplayPlan(tuple(steps), initial)) as session:
        values = session.run()
    ops = ops_by_label(trace)
    for step in steps:
        assert ADAPTER.close(ops[step.label].out, values[step.label]), f"{name}:{step.label}"
    grads = _output_seeds(trace)
    assert grads, "no model-output op to seed"
    for step in reversed(steps):
        seed = grads.get(step.label)
        if seed is None:
            continue
        parents = [values.get(o.parent_label, initial.get(o.parent_label)) for o in step.inputs]
        grads = merge_grads(ADAPTER, grads, pullback_step(ADAPTER, trace, step, parents, [seed]))
    inputs = tf.convert_to_tensor(example)
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        flat = tf.nest.flatten(model(inputs))
        loss = tf.add_n([tf.reduce_sum(item * _loss_weights(item)) for item in flat])
    (input_op,) = [op for op in _unique_ops(trace) if bool(getattr(op, "is_input", False))]
    assert ADAPTER.close(tape.gradient(loss, inputs), grads[input_op.label]), name
