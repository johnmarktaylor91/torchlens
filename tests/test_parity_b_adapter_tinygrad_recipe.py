"""tinygrad replay recipes: data only, and a round trip rebuilds captures the validator passes."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pytest

import torchlens as tl
from torchlens.backends._replay_adapters import (
    ReplayRefusal,
    decode_recipe,
    encode_recipe,
    get_adapter,
    validate_recipe,
)

pytest.importorskip("tinygrad")
from tinygrad import Tensor  # noqa: E402

from tests.conformance._rosters import tinygrad as roster  # noqa: E402
from tests.test_parity_b_adapter_tinygrad_replay import live_validator_failure  # noqa: E402
from torchlens.backends.tinygrad import TinygradBackend  # noqa: E402

pytestmark = [pytest.mark.backend_tinygrad, pytest.mark.rare]

adapter = get_adapter("tinygrad")
ROWS = roster.config_built_rows() + roster.random_init_rows()


def _trace(row: Any) -> Any:
    model, value = row.builder()
    return tl.trace(model, value, backend="tinygrad")


def _round_tripped_recipes(trace: Any) -> dict[str, Any]:
    """Every op's recipe through encode, JSON text and decode against the allowlist."""

    recipes = {}
    for op in trace._layer_list:
        recipe = adapter.recipe(trace, op)
        if recipe is None:
            continue
        validate_recipe(recipe)
        text = json.dumps(encode_recipe(recipe))
        decoded = decode_recipe(json.loads(text), backend="tinygrad", allowlist=adapter.ALLOWLIST)
        assert decoded == recipe
        recipes[recipe.output_slots[0]] = decoded
    return recipes


@pytest.mark.parametrize("row", ROWS, ids=lambda row: row.family)
def test_recipe_round_trip_rebuilds_captures_the_validator_passes(row: Any) -> None:
    """Drop the live captures, bind decoded recipes: the unchanged validator gives the live verdict."""

    trace = _trace(row)
    live = {c.label_raw: c for c in trace.tinygrad_uop_captures}
    live_failure = live_validator_failure(trace)
    recipes = _round_tripped_recipes(trace)
    trace.tinygrad_uop_captures = ()
    adapter.bind_recipes(trace, recipes)
    rebuilt = {c.label_raw: c for c in trace.tinygrad_uop_captures}
    assert rebuilt.keys() == live.keys()
    for raw, capture in rebuilt.items():
        assert capture.parent_arg_positions == tuple(sorted(live[raw].parent_arg_positions))
    # The rebuilt captures give the live verdict: a pass, or the same rejected op
    # where the unchanged validator rejects one live on this device.
    assert live_validator_failure(trace) == live_failure


def test_rebuilt_captures_still_catch_a_corrupted_payload() -> None:
    """After a round trip the validator still fails a corrupted saved output."""

    trace = _trace(ROWS[0])
    recipes = _round_tripped_recipes(trace)
    trace.tinygrad_uop_captures = ()
    adapter.bind_recipes(trace, recipes)
    target = next(op for op in trace._layer_list if op.type == "add")
    target.out = (target.out + 99.0).realize()
    assert TinygradBackend().validate_trace(trace) is False


def test_recipes_are_data_only_and_name_their_inputs() -> None:
    """Recipes carry the UOp op, occurrences with source positions, and literal sources."""

    trace = tl.trace(lambda v: (v * v).reshape(3, 1), Tensor([1.0, -2.0, 3.0]), backend="tinygrad")
    by_type = {op.type: adapter.recipe(trace, op) for op in trace._layer_list}
    mul = by_type["mul"]
    assert mul.op_identity == "tinygrad.uop.MUL"
    assert [occ.arg_path for occ in mul.inputs] == [(0,), (1,)]
    assert mul.inputs[0].parent_label == mul.inputs[1].parent_label
    reshape = by_type["reshape"]
    assert reshape.op_identity == "tinygrad.uop.RESHAPE"
    assert [path for path, _tree in reshape.literals] == [(1,)]
    assert reshape.rng is None and reshape.framework_version


def test_bind_refuses_missing_foreign_and_corrupt_recipes() -> None:
    """A missing recipe, another backend's, an unlisted op or a bad source position is refused."""

    trace = _trace(ROWS[0])
    recipes = _round_tripped_recipes(trace)
    label, item = next((k, v) for k, v in recipes.items() if v.inputs)
    cases = [
        {k: v for k, v in recipes.items() if k != label},
        {**recipes, label: replace(item, backend="jax")},
        {**recipes, label: replace(item, op_identity="tinygrad.uop.THREEFRY")},
        {**recipes, label: replace(item, attrs={**item.attrs, "n_src": 0})},
    ]
    for bad in cases:
        with pytest.raises(ReplayRefusal):
            adapter.bind_recipes(trace, bad)
    with pytest.raises(ReplayRefusal):
        decode_recipe(
            {**encode_recipe(item), "op_identity": "tinygrad.uop.CUSTOM"},
            backend="tinygrad",
            allowlist=adapter.ALLOWLIST,
        )


def _saved_value_op(trace: Any, recipes: dict[str, Any]) -> Any:
    """An op whose recipe is a saved value (no parents), with a float payload."""

    return next(
        op
        for op in trace._layer_list
        if op.label in recipes
        and recipes[op.label].op_identity == adapter.SAVED_VALUE_IDENTITY
        and "float" in str(op.out.dtype)
    )


@pytest.mark.parametrize("row", ROWS[:3], ids=lambda row: row.family)
def test_rebound_saved_value_is_checked_against_the_recipe_not_the_payload(row: Any) -> None:
    """A payload tampered before binding fails validation: the snapshot is recipe data."""

    trace = _trace(row)
    recipes = _round_tripped_recipes(trace)
    target = _saved_value_op(trace, recipes)
    target.out = (target.out + 99.0).realize()
    trace.tinygrad_uop_captures = ()
    adapter.bind_recipes(trace, recipes)
    assert TinygradBackend().validate_trace(trace) is False


def test_large_saved_values_bind_by_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    """Over the size limit a saved value is a digest: the same payload binds, another is refused."""

    monkeypatch.setattr(adapter, "_MAX_VALUE_ITEMS", 0)
    trace = _trace(ROWS[0])
    recipes = _round_tripped_recipes(trace)
    target = _saved_value_op(trace, recipes)
    assert "sha256" in recipes[target.label].attrs["value"]
    trace.tinygrad_uop_captures = ()
    adapter.bind_recipes(trace, recipes)
    assert TinygradBackend().validate_trace(trace) is True
    target.out = (target.out + 99.0).realize()
    with pytest.raises(ReplayRefusal):
        adapter.bind_recipes(trace, recipes)
