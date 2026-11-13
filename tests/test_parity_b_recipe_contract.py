"""Replay recipe contract: data-only round trips and refusals before execution."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.support import replay_fake_adapter as fake
from torchlens.backends import _replay_adapters as ra
from torchlens.backends._replay_adapters._contract import RecipeLimits

PKG_DIR = Path(ra.__file__).parent
FRAMEWORKS = {"torch", "jax", "jaxlib", "tensorflow", "mlx", "paddle", "tinygrad", "flax", "keras"}


def _recipe(**overrides: Any) -> ra.OpRecipe:
    base: dict[str, Any] = {
        "backend": "jax",
        "op_identity": "dot_general",
        "inputs": (
            ra.InputOccurrence("x", (0,)),
            ra.InputOccurrence("x", (1,)),
            ra.InputOccurrence("w", ("rhs", 0, "kernel")),
        ),
        "literals": (((2,), float("-inf")), ((3,), (1, 2.5, None, True)), ((4,), 1 + 2j)),
        "attrs": {
            "dimension_numbers": (((1,), (0,)), ((), ())),
            "jaxpr": {"eqns": [{"prim": "add", "params": {"axes": (0,)}}], "consts": []},
            "precision": None,
            "nan_fill": float("nan"),
        },
        "output_slots": ("y_1", "y_2", "y_3"),
        "param_refs": (((1,), "layers.0.kernel"),),
        "constant_refs": (((5,), "constant_7"),),
        "rng": {"key_label": "key_1", "draws": 2},
        "framework_version": "0.4.38",
    }
    base.update(overrides)
    return ra.OpRecipe(**base)


def _same(left: ra.OpRecipe, right: ra.OpRecipe) -> bool:
    # NaN never equals itself, so compare canonical encodings.
    return json.dumps(ra.encode_recipe(left), sort_keys=True) == json.dumps(
        ra.encode_recipe(right), sort_keys=True
    )


def _decode(tree: Any, **kw: Any) -> ra.OpRecipe:
    kw.setdefault("backend", "jax")
    kw.setdefault("allowlist", {"dot_general"})
    return ra.decode_recipe(tree, **kw)


def test_round_trip_through_json_keeps_repeats_slots_and_nested_attrs() -> None:
    recipe = _recipe()
    tree = json.loads(json.dumps(ra.encode_recipe(recipe), allow_nan=False))
    back = _decode(tree)
    assert _same(back, recipe)
    assert back.parent_labels == ("x", "x", "w")
    assert back.inputs[2].arg_path == ("rhs", 0, "kernel")
    assert back.output_slots == ("y_1", "y_2", "y_3")
    assert back.attrs["dimension_numbers"] == (((1,), (0,)), ((), ()))
    assert isinstance(back.attrs["jaxpr"]["eqns"], list)
    assert back.attrs["jaxpr"]["eqns"][0]["params"]["axes"] == (0,)
    assert back.literals[0][1] == float("-inf")
    assert back.literals[2][1] == 1 + 2j
    assert np.isnan(back.attrs["nan_fill"])
    assert back.rng == {"key_label": "key_1", "draws": 2}


def test_encoding_is_versioned_and_json_safe() -> None:
    tree = ra.encode_recipe(_recipe())
    assert tree["format_version"] == ra.RECIPE_FORMAT_VERSION
    json.dumps(tree, allow_nan=False)


def test_fake_adapter_recipes_round_trip_and_rebind() -> None:
    trace = _fake_trace()
    recipes = {}
    for op in trace.ops.values():
        rec = fake.recipe(trace, op)
        assert rec is not None
        tree = json.loads(json.dumps(ra.encode_recipe(rec)))
        recipes[op.label] = ra.decode_recipe(tree, backend="fake", allowlist=fake.ALLOWLIST)
    fresh = fake.FakeTrace(ops={}, payloads=dict(trace.payloads))
    fake.bind_recipes(fresh, recipes)
    assert fresh.captures == recipes
    for op in fresh.ops.values():
        if op.op_identity == "input":
            continue
        outs = fake.forward(fresh, op, [fresh.payloads[p] for p in op.parents])
        for slot, out in zip(op.output_slots, outs, strict=True):
            assert fake.close(trace.payloads[slot], out)


def _fake_trace() -> fake.FakeTrace:
    ops = [
        fake.FakeOp("x", "input", (), ("x",)),
        fake.FakeOp("sq", "mul", ("x", "x"), ("sq",)),
        fake.FakeOp("lo", "split2", ("sq",), ("lo", "hi"), attrs={"index": 2}),
        fake.FakeOp("m", "scale", ("hi",), ("m",), literals=((("factor",), 3.0),)),
        fake.FakeOp("s", "sum", ("m",), ("s",), attrs={"axis": 0}),
    ]
    return fake.build_trace(ops, {"x": np.arange(5.0)})


def test_fake_adapter_fills_the_protocol() -> None:
    assert isinstance(fake, ra.ReplayAdapter)


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda t: {**t, "format_version": 2}, "format_version"),
        (lambda t: {**t, "format_version": True}, "format_version"),
        (lambda t: {k: v for k, v in t.items() if k != "format_version"}, "format_version"),
        (lambda t: {k: v for k, v in t.items() if k != "attrs"}, "shape"),
        (lambda t: {**t, "loader": "os.system"}, "shape"),
        (lambda t: {**t, "op_identity": "os.system"}, "unknown_op"),
        (lambda t: {**t, "backend": "tf"}, "backend"),
        (lambda t: {**t, "inputs": [["x"]]}, "shape"),
        (lambda t: {**t, "inputs": [["x", []]]}, "type"),
        (lambda t: {**t, "inputs": [["x", [-1]]]}, "type"),
        (lambda t: {**t, "inputs": [[3, [0]]]}, "type"),
        (lambda t: {**t, "attrs": {"$pickle": "gASV"}}, "type"),
        (lambda t: {**t, "attrs": {"$dict": {"a": {"$float": "1e999"}}}}, "type"),
        (lambda t: {**t, "attrs": {"$dict": {"a": float("nan")}}}, "type"),
        (lambda t: {**t, "attrs": [1, 2]}, "type"),
        (lambda t: {**t, "output_slots": []}, "slots"),
        (lambda t: {**t, "output_slots": ["a", "a"]}, "slots"),
        (lambda t: {**t, "rng": "key"}, "type"),
        (lambda t: [t], "shape"),
        (lambda t: None, "shape"),
    ],
)
def test_corrupt_recipes_are_refused(mutate: Any, reason: str) -> None:
    tree = mutate(ra.encode_recipe(_recipe()))
    with pytest.raises(ra.ReplayRefusal) as info:
        _decode(tree)
    assert info.value.reason == reason


def test_non_data_leaves_are_refused_by_the_bounds_walk() -> None:
    for leaf in (object(), print, b"raw", np.float32(1.0), np.zeros(2)):
        tree = {**ra.encode_recipe(_recipe()), "framework_version": leaf}
        with pytest.raises(ra.ReplayRefusal) as info:
            _decode(tree)
        assert info.value.reason == "type"
    tree = ra.encode_recipe(_recipe())
    tree["attrs"] = {"$dict": {1: 2}}
    with pytest.raises(ra.ReplayRefusal):
        _decode(tree)


@pytest.mark.parametrize(
    "bad_value", [print, object(), np.zeros(2), np.float64(1.0), {1: "x"}, {"a": {3}}]
)
def test_encoding_refuses_non_data_values(bad_value: Any) -> None:
    with pytest.raises(ra.ReplayRefusal) as info:
        ra.encode_recipe(_recipe(attrs={"bad": bad_value}))
    assert info.value.reason == "type"


def test_bounds_refuse_oversized_trees() -> None:
    tree = ra.encode_recipe(_recipe())
    cases = [
        (tree, RecipeLimits(max_inputs=2), "shape"),
        ({**tree, "framework_version": "v" * 80}, RecipeLimits(max_str_len=64), "bounds"),
        ({**tree, "attrs": {"$dict": {"n": 1 << 40}}}, RecipeLimits(max_int_bits=32), "bounds"),
        ({**tree, "attrs": _nest(20)}, RecipeLimits(max_depth=16), "bounds"),
        ({**tree, "literals": [[[0], list(range(300))]]}, RecipeLimits(max_nodes=300), "bounds"),
    ]
    for case, limits, reason in cases:
        _decode(tree, limits=RecipeLimits())  # the unmodified tree fits the defaults
        with pytest.raises(ra.ReplayRefusal) as info:
            _decode(case, limits=limits)
        assert info.value.reason == reason


def test_default_bounds_refuse_a_deep_tree_without_recursing() -> None:
    tree = {**ra.encode_recipe(_recipe()), "attrs": _nest(100_000)}
    with pytest.raises(ra.ReplayRefusal) as info:
        _decode(tree)
    assert info.value.reason == "bounds"


def _nest(depth: int) -> Any:
    node: Any = 0
    for _ in range(depth):
        node = [node]
    return {"$dict": {"deep": node}}


def test_refusal_happens_before_any_adapter_call() -> None:
    calls: list[str] = []

    class Spy:
        def __getattr__(self, name: str) -> Any:
            calls.append(name)
            raise AssertionError(name)

    tree = {**ra.encode_recipe(_recipe()), "op_identity": "unlisted"}
    with pytest.raises(ra.ReplayRefusal):
        recipe = _decode(tree)
        Spy().bind_recipes(None, {"y_1": recipe})
    assert calls == []


def test_validate_recipe_refuses_bad_shapes() -> None:
    with pytest.raises(ra.ReplayRefusal):
        ra.validate_recipe(_recipe(output_slots=()))
    with pytest.raises(ra.ReplayRefusal):
        ra.validate_recipe(_recipe(op_identity=""))
    with pytest.raises(ra.ReplayRefusal):
        ra.validate_recipe(_recipe(inputs=(("x", (0,)),)))


def test_lookup_is_closed_and_honest() -> None:
    for name in ("torch", "fake", "os", "../x", ""):
        assert not ra.has_adapter(name)
        with pytest.raises(ra.ReplayRefusal):
            ra.get_adapter(name)
    for name in sorted(ra.ADAPTER_BACKENDS):
        try:
            module = ra.get_adapter(name)
        except ra.ReplayRefusal:
            continue
        assert isinstance(module, ra.ReplayAdapter)


def test_package_imports_no_framework_at_module_level() -> None:
    """Every module in the package, adapters included, keeps framework imports
    inside functions."""

    for path in sorted(PKG_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in _module_level_imports(tree):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module]
            for name in names:
                if name and getattr(node, "level", 0) == 0:
                    assert name.split(".")[0] not in FRAMEWORKS, f"{path.name} imports {name}"


def _module_level_imports(tree: ast.Module) -> list[ast.Import | ast.ImportFrom]:
    found: list[ast.Import | ast.ImportFrom] = []
    stack: list[ast.AST] = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            found.append(node)
        elif isinstance(node, (ast.If, ast.Try, ast.With)):
            stack.extend(ast.iter_child_nodes(node))
        elif isinstance(node, ast.ExceptHandler):
            stack.extend(node.body)
    return found


def _module_like_fake(drop: str | None = None) -> Any:
    import types

    module = types.ModuleType("torchlens.backends._replay_adapters.paddle")
    for name in (*ra._ADAPTER_FUNCTIONS, "ALLOWLIST"):
        if name != drop:
            setattr(module, name, getattr(fake, name))
    return module


@pytest.mark.parametrize("drop", [None, "ALLOWLIST", "vjp", "bind_recipes"])
def test_lookup_requires_every_function_and_the_allowlist(
    monkeypatch: pytest.MonkeyPatch, drop: str | None
) -> None:
    import sys

    module = _module_like_fake(drop)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    if drop is None:
        assert ra.get_adapter("paddle") is module
        return
    with pytest.raises(ra.ReplayRefusal, match=drop):
        ra.get_adapter("paddle")


def test_lookup_refuses_a_mutable_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    module = _module_like_fake()
    module.ALLOWLIST = set(fake.ALLOWLIST)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(ra.ReplayRefusal, match="ALLOWLIST"):
        ra.get_adapter("paddle")
