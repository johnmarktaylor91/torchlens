"""MLX replay adapter proofs on the MLX conformance roster.

``forward`` equals the saved payload on every replayable op; ``vjp`` composed
over each model equals ``mx.grad``; recipes survive a JSON round trip and
rebuild a capture list the unchanged MLX validator passes with the same
decisions; corrupt recipes are refused before anything executes.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

mlx = pytest.importorskip("mlx")
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

import torchlens as tl  # noqa: E402
from tests.conformance._rosters.mlx import CONFIG_BUILT  # noqa: E402
from torchlens.backends._replay_adapters import (  # noqa: E402
    OpRecipe,
    ReplayRecipeError,
    ReplaySession,
    ReplayStep,
    _mlx_ops as ops_mod,
    _mlx_recipes as recipes_mod,
    decode_recipe,
    encode_recipe,
    get_adapter,
    mlx as adapter,
    pullback_step,
)
from torchlens.backends._validation_shared import ops_by_label  # noqa: E402
from torchlens.backends.mlx._replay_args import _saved_payload  # noqa: E402
from torchlens.backends.mlx.validation import validate_mlx_captures_detailed  # noqa: E402

pytestmark = [pytest.mark.backend_mlx, pytest.mark.rare]

ROWS = {row.family: row for row in CONFIG_BUILT}
FAMILIES = sorted(ROWS)


def _trace(family: str) -> tuple[Any, Any, Any]:
    """Trace one roster row on MLX; return the trace, model and input."""

    model, example = ROWS[family].builder()
    return tl.trace(model, example, backend="mlx"), model, example


def _parents(trace: Any, capture: Any) -> list[Any]:
    """Saved parent payloads, one per occurrence."""

    ops = ops_by_label(trace)
    return [_saved_payload(trace, ops, label) for label, _path in ops_mod._occurrences(capture)]


def _decisions(outcome: Any) -> list[tuple[Any, ...]]:
    """Comparable view of validator decisions."""

    return sorted(
        (d["op_label"], d["func_name"], d["phase"], d["decision"], d["reason"])
        for d in outcome.decisions
    )


def test_adapter_is_registered_and_complete() -> None:
    """``get_adapter('mlx')`` returns this module with every contract function."""

    assert get_adapter("mlx") is adapter
    assert adapter.op_allowlist() == adapter.ALLOWLIST
    assert {"input", "constant", "core.matmul", "layer.Linear"} <= adapter.ALLOWLIST


@pytest.mark.parametrize("family", FAMILIES)
def test_forward_equals_saved_payload(family: str) -> None:
    """Every replayable op of the row re-runs to its saved payload."""

    trace, _model, _x = _trace(family)
    ops = ops_by_label(trace)
    checked = 0
    for op in trace._layer_list:
        assert adapter.can_replay(trace, op) is None, (family, op.func_name)
        if op.is_input:
            (value,) = adapter.forward(trace, op, [])
            assert adapter.close(_saved_payload(trace, ops, op._label_raw), value)
            continue
        capture = ops_mod._capture_for(trace, op)
        outputs = adapter.forward(trace, op, _parents(trace, capture))
        mx.eval(*outputs)
        for label, value in zip(capture.labels_raw, outputs, strict=True):
            expected = _saved_payload(trace, ops, label)
            assert adapter.close(expected, value), (family, capture.op_name, label)
        checked += 1
    assert checked


def _output_labels(trace: Any) -> list[str]:
    """Raw labels of the ops that produced the model outputs, in output order."""

    ops = ops_by_label(trace)
    return [ops[label]._label_raw for label in trace.output_layers]


def _flat_outputs(value: Any) -> list[Any]:
    """Array leaves of a model output."""

    if isinstance(value, mx.array):
        return [value]
    return [leaf for item in value for leaf in _flat_outputs(item)]


def _plan(trace: Any) -> tuple[list[ReplayStep], dict[str, Any]]:
    """Replay steps for every non-input capture and the input payloads."""

    ops = ops_by_label(trace)
    steps: list[ReplayStep] = []
    initial: dict[str, Any] = {}
    for op in trace._layer_list:
        if op.is_input:
            initial[op._label_raw] = _saved_payload(trace, ops, op._label_raw)
            continue
        rec = adapter.recipe(trace, op)
        assert rec is not None, op.func_name
        if rec.output_slots[0] == op._label_raw:
            steps.append(ReplayStep.from_recipe(op, rec))
    return steps, initial


@pytest.mark.parametrize("family", FAMILIES)
def test_session_replays_the_row_from_its_inputs(family: str) -> None:
    """Chained ``forward`` from the input alone reproduces every saved payload."""

    trace, _model, _x = _trace(family)
    steps, initial = _plan(trace)
    values = ReplaySession(adapter, trace, steps, initial=initial).run()
    ops = ops_by_label(trace)
    for op in trace._layer_list:
        expected = _saved_payload(trace, ops, op._label_raw)
        assert adapter.close(expected, values[op._label_raw]), (family, op.func_name)


def _reverse(trace: Any, seeds: dict[str, Any]) -> dict[str, Any]:
    """Compose the adapter's per-op ``vjp`` over the trace with ``pullback_step``."""

    steps, initial = _plan(trace)
    values = ReplaySession(adapter, trace, steps, initial=initial).run()
    grads = dict(seeds)
    for step in reversed(steps):
        cotangents = [grads.get(slot) for slot in step.output_slots]
        if all(c is None for c in cotangents):
            continue
        parents = [values[occ.parent_label] for occ in step.inputs]
        pullback_step(adapter, trace, step, parents, cotangents, grads)
    return grads


@pytest.mark.parametrize("family", FAMILIES)
def test_vjp_chain_equals_mx_grad(family: str) -> None:
    """The per-op derivatives composed over the trace equal ``mx.grad``."""

    trace, model, example = _trace(family)
    outputs = _flat_outputs(model(example))
    rng = np.random.default_rng(7)
    weights = [mx.array(rng.standard_normal(o.shape).astype(np.float32)) for o in outputs]

    def loss(x: Any) -> Any:
        produced = _flat_outputs(model(x))
        return sum(mx.sum(w * o) for w, o in zip(weights, produced, strict=True))

    expected = mx.grad(loss)(example)
    labels = _output_labels(trace)
    assert len(labels) == len(outputs), (family, labels)
    seeds: dict[str, Any] = {}
    for label, weight in zip(labels, weights, strict=True):
        seeds[label] = weight if label not in seeds else seeds[label] + weight
    grads = _reverse(trace, seeds)
    (input_op,) = [op for op in trace._layer_list if op.is_input]
    composed = grads[input_op._label_raw]
    mx.eval(expected, composed)
    assert adapter.close(expected, composed), (family, np.abs(np.array(expected - composed)).max())


def _recipes(trace: Any) -> dict[str, OpRecipe]:
    """Every op's recipe after an encode, JSON and decode round trip."""

    allow = adapter.op_allowlist()
    recipes: dict[str, OpRecipe] = {}
    for op in trace._layer_list:
        built = recipes_mod._recipe_or_reason(trace, op)
        assert isinstance(built, OpRecipe), (op.func_name, built)
        data = json.loads(json.dumps(encode_recipe(built)))
        decoded = decode_recipe(data, backend="mlx", allowlist=allow)
        assert decoded == built
        recipes[decoded.output_slots[0]] = decoded
    return recipes


@pytest.mark.parametrize("keep_inventory", [True, False])
@pytest.mark.parametrize("family", FAMILIES)
def test_recipe_round_trip_passes_unchanged_validator(family: str, keep_inventory: bool) -> None:
    """Captures rebuilt from recipes pass the validator with the same decisions."""

    trace, _model, _x = _trace(family)
    before = validate_mlx_captures_detailed(trace)
    assert before.failed_count == 0
    recipes = _recipes(trace)
    trace._replay_param_payloads = adapter.live_param_payloads(trace)
    trace._mlx_op_captures = []
    if not keep_inventory:
        trace._mlx_replay_inventory = None
    adapter.bind_recipes(trace, recipes)
    after = validate_mlx_captures_detailed(trace)
    assert after.failed_count == 0
    assert after.replayed_count == before.replayed_count
    assert _decisions(after) == _decisions(before)


def test_bound_captures_still_catch_a_corrupt_payload() -> None:
    """A rebound trace is not vacuous: a corrupted saved output fails validation."""

    trace, _model, _x = _trace("mlp")
    recipes = _recipes(trace)
    trace._replay_param_payloads = adapter.live_param_payloads(trace)
    trace._mlx_op_captures = []
    adapter.bind_recipes(trace, recipes)
    victim = next(op for op in trace._layer_list if not op.is_input and op.out is not None)
    with pytest.warns(UserWarning):
        victim.out = victim.out + 1.0
    assert validate_mlx_captures_detailed(trace).failed_count > 0


def test_missing_parameter_payload_is_refused() -> None:
    """Binding without a parameter payload refuses the recipe before replay."""

    trace, _model, _x = _trace("mlp")
    recipes = _recipes(trace)
    captures = list(trace._mlx_op_captures)
    trace._replay_param_payloads = {}
    with pytest.raises(ReplayRecipeError, match="no parameter payload"):
        adapter.bind_recipes(trace, recipes)
    assert list(trace._mlx_op_captures) == captures


def test_corrupt_recipes_are_refused() -> None:
    """Unknown identities and skeletons that disagree with inputs are refused."""

    trace, _model, _x = _trace("chain_branch")
    recipes = _recipes(trace)
    label, rec = next((k, r) for k, r in recipes.items() if r.inputs)
    data = encode_recipe(rec)
    data["op_identity"] = "core.not_an_mlx_op"
    with pytest.raises(ReplayRecipeError, match="allowlist"):
        decode_recipe(data, backend="mlx", allowlist=adapter.op_allowlist())
    dropped = OpRecipe(**{**rec.__dict__, "inputs": rec.inputs[:-1]})
    with pytest.raises(ReplayRecipeError, match="occurrences"):
        adapter.bind_recipes(trace, {**recipes, label: dropped})


class _Drop(nn.Module):
    """Dropout after a linear layer, for the random-op refusal."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.drop = nn.Dropout(0.5)

    def __call__(self, x: Any) -> Any:
        return self.drop(self.lin(x))


def test_random_and_unlisted_ops_name_their_reason() -> None:
    """Training-mode dropout and an unlisted callable return a ``can_replay`` reason."""

    model = _Drop()
    model.train()
    trace = tl.trace(model, mx.ones((2, 4)), backend="mlx")
    reasons = {op.func_name: adapter.can_replay(trace, op) for op in trace._layer_list}
    assert "random op" in str(reasons.get("dropout"))
    assert reasons.get("linear") is None
    capture = ops_mod._capture_for(trace, next(op for op in trace._layer_list if not op.is_input))
    fake = type(capture)(labels_raw=capture.labels_raw, op_name="mystery", func=lambda x: x)
    assert "unlisted op" in str(adapter.can_replay(trace, fake))
    model.eval()
    trace = tl.trace(model, mx.ones((2, 4)), backend="mlx")
    assert all(adapter.can_replay(trace, op) is None for op in trace._layer_list)
