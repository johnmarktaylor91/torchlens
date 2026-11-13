"""Paddle replay recipes are data only and round trip through the unchanged validator.

Proof (wave B, lane b-adapter-paddle): recipes from every roster row survive
``encode_recipe`` / JSON / ``decode_recipe``; ``bind_recipes`` rebuilds a
capture list equal to the live one that the unchanged Paddle validator passes;
a tampered, missing or corrupt recipe fails closed, naming the op.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

paddle = pytest.importorskip("paddle")

import torchlens as tl  # noqa: E402
from tests.conformance._rosters import paddle as paddle_roster  # noqa: E402
from torchlens._data_classes.param import Param  # noqa: E402
from torchlens.backends._replay_adapters import (  # noqa: E402
    ReplayRefusal,
    decode_recipe,
    encode_recipe,
    get_adapter,
)
from torchlens.backends.paddle.validation import _ops_by_label as ops_by_label  # noqa: E402
from torchlens.backends.paddle.validation_decisions import public_label  # noqa: E402
from torchlens.backends.registry import get_backend_spec  # noqa: E402

pytestmark = [pytest.mark.backend_paddle, pytest.mark.rare]

adapter: Any = get_adapter("paddle")
_ROWS = {row.family: row for row in paddle_roster.roster()}
_FAMILIES = sorted(_ROWS)
_SWAPS = {
    "add": "subtract",
    "tensor.__add__": "tensor.__sub__",
    "tensor.__mul__": "tensor.__add__",
    "functional.relu": "functional.sigmoid",
    "functional.gelu": "functional.relu",
    "matmul": "add",
}


def _trace(family: str) -> Any:
    """Capture one roster row."""

    model, example = _ROWS[family].builder()
    args = example if isinstance(example, tuple) else (example,)
    return tl.trace(model, args, backend="paddle")


def _validate(trace: Any) -> Any:
    """Run the unchanged Paddle validator through the backend protocol."""

    factory = get_backend_spec("paddle").protocol
    assert factory is not None
    return factory().validate(trace, None)


def _recipes(trace: Any) -> tuple[dict[str, Any], dict[str, str]]:
    """Round-tripped recipes keyed by first slot, and refusals by label."""

    recipes: dict[str, Any] = {}
    refused: dict[str, str] = {}
    for op in trace._layer_list:
        if op.is_input:
            continue
        item = adapter.recipe(trace, op)
        if item is None:
            refused[str(op._label_raw)] = str(adapter.recipe_refusal(trace, op))
            continue
        tree = json.loads(json.dumps(encode_recipe(item)))
        decoded = decode_recipe(tree, backend="paddle", allowlist=adapter.ALLOWLIST)
        assert decoded == item
        recipes[decoded.output_slots[0]] = decoded
    return recipes, refused


def _capture_view(capture: Any) -> tuple[Any, ...]:
    """The fields of a capture the validator reads, comparable across rebuilds."""

    return (
        capture.func,
        capture.op_name,
        capture.label_raw,
        capture.args_template,
        capture.kwargs_template,
        capture.tensor_inputs,
        tuple(tuple(path) for path in capture.output_leaf_paths),
        capture.producer_labels,
        capture.capture_gap_markers,
        tuple(sorted(map(repr, capture.param_inputs))),
    )


def _failed_labels(status: Any) -> list[str]:
    """Labels of every failed decision in a status."""

    return [str(item.get("op_label")) for item in status.decisions if item["decision"] == "failed"]


@pytest.mark.parametrize("family", _FAMILIES)
def test_recipe_round_trip_rebuilds_captures_the_validator_passes(family: str) -> None:
    """Strip the live captures, bind round-tripped recipes, validate unchanged."""

    trace = _trace(family)
    assert _validate(trace).state == "passed"
    recipes, refused = _recipes(trace)
    assert refused == {}, family
    live = {
        c.label_raw: _capture_view(c) for c in trace._paddle_op_captures if not c.alias_annotations
    }
    trace._paddle_op_captures = []
    trace._paddle_alias_annotations = []
    adapter.bind_recipes(trace, recipes)
    bound = {c.label_raw: _capture_view(c) for c in trace._paddle_op_captures}
    assert bound == live, family
    status = _validate(trace)
    assert status.state == "passed", f"{family}: {status.message}"


def _swap_target(recipes: dict[str, Any]) -> tuple[str, Any]:
    """The first recipe whose op identity has a same-arity swap."""

    return next((key, item) for key, item in recipes.items() if item.op_identity in _SWAPS)


@pytest.mark.parametrize("family", ["mod_mlp", "fn_chain_branch"])
def test_tampered_recipe_fails_validation_naming_the_op(family: str) -> None:
    """A recipe naming another allowlisted op replays wrong and fails at that op."""

    trace = _trace(family)
    recipes, _refused = _recipes(trace)
    key, item = _swap_target(recipes)
    recipes[key] = replace(item, op_identity=_SWAPS[item.op_identity])
    adapter.bind_recipes(trace, recipes)
    status = _validate(trace)
    assert status.state == "failed"
    assert public_label(key, ops_by_label(trace)) in _failed_labels(status), status.decisions


def test_missing_recipe_fails_validation_naming_the_op() -> None:
    """An op left without a recipe is a coverage failure, never a silent skip."""

    trace = _trace("mod_mlp")
    recipes, _refused = _recipes(trace)
    dropped = sorted(recipes)[-1]
    del recipes[dropped]
    adapter.bind_recipes(trace, recipes)
    status = _validate(trace)
    assert status.state == "failed"
    assert public_label(dropped, ops_by_label(trace)) in _failed_labels(status), status.decisions


def test_missing_parameter_value_fails_closed() -> None:
    """A parameter reference with no recorded value fails at that op."""

    trace = _trace("mod_mlp")
    recipes, _refused = _recipes(trace)
    key, item = next((k, r) for k, r in recipes.items() if r.param_refs)
    path, _address = item.param_refs[0]
    recipes[key] = replace(item, param_refs=((path, "no.such.param"), *item.param_refs[1:]))
    adapter.bind_recipes(trace, recipes)
    assert _validate(trace).state == "failed"


def test_unlisted_op_identity_and_corrupt_literals_are_refused() -> None:
    """Decoding refuses ops outside the closed allowlist; binding refuses bad tags."""

    trace = _trace("mod_mlp")
    recipes, _refused = _recipes(trace)
    key, item = next(iter(recipes.items()))
    tree = encode_recipe(replace(item, op_identity="os.system"))
    with pytest.raises(ReplayRefusal, match="unknown_op"):
        decode_recipe(tree, backend="paddle", allowlist=adapter.ALLOWLIST)
    with pytest.raises(ReplayRefusal):
        adapter.bind_recipes(trace, {key: replace(item, op_identity="os.system")})
    with pytest.raises(ReplayRefusal, match="slots"):
        adapter.bind_recipes(trace, {"not-the-first-slot": item})
    with pytest.raises(ReplayRefusal):
        adapter._decode_literal({"__paddle_dtype__": "not_a_dtype"})


def test_literal_codec_round_trips_paddle_values() -> None:
    """Slices, Ellipsis, NumPy scalars and dtypes, and Paddle dtypes round trip as data."""

    values = [
        (slice(None, 2, None), Ellipsis, 0),
        np.float32(1.5),
        np.dtype("int64"),
        paddle.float32,
        {"axis": [0, 1], "keepdim": True},
    ]
    for value in values:
        encoded = adapter._encode_literal(value)
        assert adapter._decode_literal(encoded) == value


def test_allowlist_is_the_captured_surface() -> None:
    """The allowlist names captured ops only: no denied, random or mutator names."""

    assert "functional.linear" in adapter.ALLOWLIST
    assert "conv2d" in adapter.ALLOWLIST
    assert "constant" in adapter.ALLOWLIST
    for denied in ("randn", "tensor.__setitem__", "tensor.numpy", "save", "tensor.add_"):
        assert denied not in adapter.ALLOWLIST


@pytest.mark.parametrize("family", ["mod_mlp", "mod_cnn", "mod_shared_twice"])
def test_bind_reads_parameters_from_loader_payloads(family: str, monkeypatch: Any) -> None:
    """With no live parameter (as after load), binding reads the loader's payloads."""

    trace = _trace(family)
    recipes, _refused = _recipes(trace)
    payloads = adapter.live_param_payloads(trace)
    assert payloads and all(isinstance(v, np.ndarray) for v in payloads.values()), family
    assert {a for r in recipes.values() for _p, a in r.param_refs} == set(payloads), family
    monkeypatch.setattr(Param, "value", property(lambda _self: None))
    trace._paddle_op_captures = []
    adapter.bind_recipes(trace, recipes)
    assert _validate(trace).state == "failed", family
    adapter.bind_recipes(trace, recipes, param_payloads=payloads)
    status = _validate(trace)
    assert status.state == "passed", f"{family}: {status.message}"
