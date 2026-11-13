"""TensorFlow replay recipes round trip into a capture list the unchanged validator passes.

Each roster model: every captured op gets a data-only recipe, which is encoded,
sent through JSON, decoded against the closed allowlist, and bound back onto the
trace after its runtime captures were removed. The unchanged TensorFlow
validator must then reach exactly the live verdict. A saved and reloaded trace
validates the same way from recipes plus its loaded payloads.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from tests.conformance._rosters import tf as tf_roster

tf = pytest.importorskip("tensorflow")

import torchlens as tl  # noqa: E402
from torchlens.backends._replay_adapters import (  # noqa: E402
    ReplayRefusal,
    decode_recipe,
    encode_recipe,
    get_adapter,
)
from torchlens.backends.tf.validation import (  # noqa: E402
    validate_tf_trace_detailed,
    validate_tf_trace_status,
)

pytestmark = [pytest.mark.tf_backend, pytest.mark.rare]

ADAPTER = get_adapter("tf")
_ROWS = {row.family: row for row in tf_roster.roster()}
_ECOSYSTEM = set(tf_roster.ECOSYSTEM_FAMILIES)


def _row_params() -> list[Any]:
    return [
        pytest.param(name, marks=[pytest.mark.slow] if name in _ECOSYSTEM else [], id=name)
        for name in _ROWS
    ]


def _trace(name: str) -> Any:
    model, example = _ROWS[name].builder()
    return tl.trace(model, example, backend="tf")


def _unique_ops(trace: Any) -> list[Any]:
    """Every recorded op once, in graph (execution) order."""

    return list(trace._layer_list)


def _recipes_through_json(trace: Any) -> dict[str, Any]:
    """Every captured op's recipe, encoded, sent through JSON and decoded."""

    decoded = {}
    for op in _unique_ops(trace):
        recipe = ADAPTER.recipe(trace, op)
        if recipe is None:
            continue
        wire = json.loads(json.dumps(encode_recipe(recipe)))
        decoded[op.label] = decode_recipe(wire, backend="tf", allowlist=ADAPTER.ALLOWLIST)
    return decoded


def _strip_runtime_captures(trace: Any) -> None:
    trace._tf_op_captures = ()
    trace._tf_init_op_labels = ()
    trace._tf_unresolved_producers = ()


def _verdict(result: Any) -> tuple[Any, ...]:
    return (result.counts, result.failures, dict(result.classes), result.decisions)


def _input_key(item: Any) -> tuple[Any, ...]:
    # A parent-linked input's value is the parent's payload; a live capture
    # without a backward pass releases its tensor, so only literals carry data.
    linked = item.producer_label_raw is not None or item.source_label_raw is not None
    array = None if linked else ADAPTER._literal_array(item.tensor)
    tensor = None if array is None else array.tolist()
    return (item.input_index, item.producer_label_raw, item.source_kind, item.source_label_raw,
            tensor)  # fmt: skip


def _capture_key(capture: Any) -> tuple[Any, ...]:
    attrs = {k: (v.name if hasattr(v, "as_datatype_enum") else v) for k, v in capture.attrs.items()}
    attrs = {k: (list(v) if isinstance(v, tuple) else v) for k, v in attrs.items()}
    return (capture.label_raw, capture.op_type, capture.output_index, attrs,
            tuple(_input_key(item) for item in capture.inputs))  # fmt: skip


def _literal_key(capture: Any) -> tuple[Any, ...]:
    """Capture key with resource and over-bound literal tensors dropped (not data)."""

    key = list(_capture_key(capture))
    key[4] = tuple(entry[:4] for entry in key[4])
    return tuple(key)


@pytest.mark.parametrize("name", _row_params())
def test_recipe_round_trip_rebuilds_a_capture_list_the_validator_passes(name: str) -> None:
    """Strip, bind from JSON-borne recipes, and get the live verdict back exactly."""

    trace = _trace(name)
    live = validate_tf_trace_detailed(trace)
    assert live.counts.failed_node_count == 0
    original = tuple(trace._tf_op_captures)
    original_init = tuple(trace._tf_init_op_labels)
    recipes = _recipes_through_json(trace)
    assert len(recipes) == len(original)
    _strip_runtime_captures(trace)
    ADAPTER.bind_recipes(trace, recipes)
    rebound = tuple(trace._tf_op_captures)
    assert [_literal_key(c) for c in rebound] == [_literal_key(c) for c in original]
    for before, after in zip(original, rebound, strict=True):
        if before.op_type in ADAPTER.KERNEL_OPS:
            assert _capture_key(after) == _capture_key(before), before.label_raw
    assert tuple(trace._tf_init_op_labels) == original_init
    assert _verdict(validate_tf_trace_detailed(trace)) == _verdict(live)


@pytest.mark.parametrize("name", ["cnn_module", "chain_branch_function"])
def test_saved_and_loaded_trace_validates_from_recipes(tmp_path: Path, name: str) -> None:
    """Validation after load: recipes plus loaded payloads give the live verdict and counts.

    ``cnn_module``'s live verdict is ``unverified`` (its variable reads are effect
    regions); ``chain_branch_function`` has none and passes.
    """

    trace = _trace(name)
    live = validate_tf_trace_detailed(trace)
    live_state = validate_tf_trace_status(trace, validate_metadata=False).state
    recipes = {k: encode_recipe(v) for k, v in _recipes_through_json(trace).items()}
    bundle = tmp_path / f"{name}.tlspec"
    trace.save(bundle, level="executable_with_callables")
    loaded = tl.load(str(bundle))
    before = validate_tf_trace_status(loaded, validate_metadata=False)
    # The bundle stores one replay recipe per op, so the loaded trace already
    # validates by replay; binding the same recipes again keeps the live verdict.
    assert before.state == live_state, before.message
    decoded = {
        label: decode_recipe(
            json.loads(json.dumps(data)), backend="tf", allowlist=ADAPTER.ALLOWLIST
        )
        for label, data in recipes.items()
    }
    ADAPTER.bind_recipes(loaded, decoded)
    after = validate_tf_trace_status(loaded, validate_metadata=False)
    assert after.state == live_state, after.message
    assert after.source == "loaded"
    assert loaded._tf_validation_result.counts == live.counts


def test_tampered_recipe_attr_changes_the_verdict() -> None:
    """The validator replays what the recipe says: a flipped MatMul attr fails replay."""

    trace = _trace("loop3_function")
    recipes = _recipes_through_json(trace)
    label = next(k for k, r in recipes.items() if r.op_identity == "MatMul")
    attrs = dict(recipes[label].attrs)
    attrs["tf_attrs"] = {**attrs["tf_attrs"], "transpose_b": True}
    recipes[label] = replace(recipes[label], attrs=attrs)
    _strip_runtime_captures(trace)
    ADAPTER.bind_recipes(trace, recipes)
    result = validate_tf_trace_detailed(trace)
    assert result.counts.failed_node_count >= 1
    assert any(item.startswith("replay_or_perturbation_failed:") for item in result.failures)


def test_unknown_and_inconsistent_recipes_are_refused_before_binding() -> None:
    """Closed allowlist on decode; identity and op type checks on bind; nothing rebuilt."""

    trace = _trace("mlp_module")
    recipes = _recipes_through_json(trace)
    label, sample = next((k, r) for k, r in recipes.items() if r.op_identity == "MatMul")
    wire = encode_recipe(sample)
    with pytest.raises(ReplayRefusal) as unknown:
        decode_recipe({**wire, "op_identity": "tf.Exec"}, backend="tf", allowlist=ADAPTER.ALLOWLIST)
    assert unknown.value.reason == "unknown_op"
    original = tuple(trace._tf_op_captures)
    bad_cases = (
        replace(sample, op_identity="tf.inert"),
        replace(sample, op_identity="Relu", attrs={**sample.attrs, "tf_op_type": "Relu"}),
        replace(sample, backend="jax"),
        replace(sample, attrs={k: v for k, v in sample.attrs.items() if k != "input_sources"}),
    )
    for bad in bad_cases:
        with pytest.raises(ReplayRefusal):
            ADAPTER.bind_recipes(trace, {**recipes, label: bad})
        assert tuple(trace._tf_op_captures) == original
    with pytest.raises(ReplayRefusal):
        ADAPTER.bind_recipes(trace, {"no_such_op": sample})


def test_recipes_are_data_only_and_name_their_parents_in_call_order() -> None:
    """Literals carry data, parents carry public labels, the resource read is inert."""

    trace = _trace("reshape_transpose_function")
    recipes = {op.label: ADAPTER.recipe(trace, op) for op in _unique_ops(trace)}
    reshape = next(r for r in recipes.values() if r is not None and r.op_identity == "Reshape")
    assert [occ.arg_path for occ in reshape.inputs] == [(0,)]
    ((path, literal),) = reshape.literals
    assert path == (1,) and literal["source_kind"] == "shape/axis"
    assert literal["array"]["data"]
    mlp = _trace("mlp_module")
    read = next(op for op in _unique_ops(mlp) if op.func_name == "ReadVariableOp")
    read_recipe = ADAPTER.recipe(mlp, read)
    assert read_recipe.op_identity == "tf.inert"
    assert read_recipe.literals[0][1]["array"] is None
    assert ADAPTER.can_replay(mlp, read).startswith("resource read")
