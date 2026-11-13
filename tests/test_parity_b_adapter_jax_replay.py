"""Unit proofs for the private JAX replay adapter (contract, edge cases, refusals)."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from tests.test_parity_b_adapter_jax_support import (
    capture,
    engine_grads,
    equation_ops,
    func,
    oracle_grads,
    round_trip_validates,
    saved_parent_values,
)
from torchlens.backends._replay_adapters import ReplayRefusal, get_adapter

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


@pytest.fixture(scope="module")
def adapter():
    return get_adapter("jax")


def _x(*shape: int) -> jax.Array:
    return jnp.asarray(np.random.default_rng(1).standard_normal(shape), dtype=jnp.float32)


def _assert_forward_all(adapter, trace) -> None:
    for op in equation_ops(trace):
        assert adapter.can_replay(trace, op) is None, op.label
        (out,) = adapter.forward(trace, op, saved_parent_values(adapter, trace, op))
        assert adapter.close(op.out, out), op.label


def _assert_grads_match(adapter, trace) -> None:
    leaves, engine = engine_grads(adapter, trace)
    replayed = oracle_grads(adapter, trace, leaves)
    for label in leaves:
        assert adapter.close(replayed[label], engine[label]), label


def test_get_adapter_returns_complete_module(adapter) -> None:
    assert isinstance(adapter.ALLOWLIST, frozenset)
    assert "jax.primitive.add" in adapter.ALLOWLIST
    assert "jax.scan_read" in adapter.ALLOWLIST


def test_repeated_parent_gets_one_cotangent_per_occurrence(adapter) -> None:
    trace = capture(lambda v: v * v, _x(3))
    (mul,) = [op for op in equation_ops(trace) if func(op) == "mul"]
    occurrences = adapter.step(trace, mul).inputs
    assert len(occurrences) == 2
    assert occurrences[0].parent_label == occurrences[1].parent_label
    parents = saved_parent_values(adapter, trace, mul)
    grads = adapter.vjp(trace, mul, parents, (jnp.ones(3, jnp.float32),))
    assert len(grads) == 2
    assert adapter.close(parents[0], grads[0]) and adapter.close(parents[0], grads[1])
    _assert_grads_match(adapter, trace)


def test_multi_output_parent_selects_its_element(adapter) -> None:
    def split(v):
        a, b = jnp.split(v, 2)
        return a * 2.0 + jnp.sin(b)

    trace = capture(split, _x(4))
    _assert_forward_all(adapter, trace)
    _assert_grads_match(adapter, trace)
    round_trip_validates(adapter, trace)


def test_integer_parent_has_no_cotangent(adapter) -> None:
    def gather(v):
        idx = jnp.argmax(v)
        return v[idx] * 3.0

    trace = capture(gather, _x(5))
    _assert_forward_all(adapter, trace)
    (argmax,) = [op for op in equation_ops(trace) if func(op) == "argmax"]
    assert not jnp.issubdtype(argmax.out.dtype, jnp.floating)
    _assert_grads_match(adapter, trace)


def test_key_based_random_replays_exactly(adapter) -> None:
    def noisy(v):
        key = jax.random.key_data(jax.random.PRNGKey(0))
        return v + jax.random.normal(jax.random.wrap_key_data(key), v.shape)

    trace = capture(noisy, _x(3))
    _assert_forward_all(adapter, trace)
    round_trip_validates(adapter, trace)


def test_sources_are_graph_only_and_carry_no_recipe(adapter) -> None:
    trace = capture(lambda v: jnp.sin(v) * 2.0, _x(3))
    sources = [op for op in trace._layer_list if func(op) == "input"]
    assert sources
    for op in sources:
        assert adapter.can_replay(trace, op).startswith("graph-only entry")
        assert adapter.recipe(trace, op) is None
        with pytest.raises(ValueError):
            adapter.forward(trace, op, [])
    for op in equation_ops(trace):
        assert adapter.recipe(trace, op) is not None, op.label


def test_stateful_rng_uniform_is_refused(adapter) -> None:
    trace = capture(lambda v: jnp.tanh(v) + 1.0, _x(3))
    captures = list(trace.jax_equation_captures)
    captures[0] = dataclasses.replace(captures[0], primitive="rng_uniform")
    trace.jax_equation_captures = tuple(captures)
    ops = equation_ops(trace)
    reason = adapter.can_replay(trace, ops[0])
    assert reason is not None and "random" in reason
    assert adapter.recipe(trace, ops[0]) is None


def test_bind_refuses_missing_foreign_and_corrupt_recipes(adapter) -> None:
    from torchlens.backends._replay_adapters import decode_recipe, encode_recipe

    trace = capture(lambda v: jnp.exp(v) * 2.0, _x(3))
    recipes = {op.label: adapter.recipe(trace, op) for op in equation_ops(trace)}
    first = next(iter(recipes))
    with pytest.raises(ReplayRefusal):
        adapter.bind_recipes(trace, {k: v for k, v in recipes.items() if k != first})
    with pytest.raises(ReplayRefusal):
        adapter.bind_recipes(
            trace, {**recipes, first: dataclasses.replace(recipes[first], backend="tf")}
        )
    tree = json.loads(json.dumps(encode_recipe(recipes[first])))
    tree["op_identity"] = "jax.primitive.not_a_primitive"
    with pytest.raises(ReplayRefusal):
        decode_recipe(tree, backend="jax", allowlist=adapter.ALLOWLIST)


def test_unbound_trace_reports_a_reason(adapter) -> None:
    trace = capture(lambda v: jnp.cos(v), _x(2))
    op = equation_ops(trace)[0]
    trace.jax_equation_captures = ()
    reason = adapter.can_replay(trace, op)
    assert reason is not None and "capture" in reason


@pytest.mark.parametrize(
    "fn",
    [
        lambda v: jax.lax.fori_loop(0, 3, lambda i, c: c * 1.5 + 0.1, v),
        lambda v: jax.lax.scan(lambda c, x: (c + x, c * x), jnp.zeros(()), v)[1],
        lambda v: jax.lax.cond(v.sum() > 0, jnp.sin, jnp.cos, v),
        lambda v: jax.lax.while_loop(lambda c: c.sum() < 10.0, lambda c: c + 1.0, v),
    ],
    ids=["fori", "scan", "cond", "while"],
)
def test_control_flow_replays_and_round_trips(adapter, fn) -> None:
    trace = capture(fn, _x(3))
    _assert_forward_all(adapter, trace)
    round_trip_validates(adapter, trace)


def test_tuple_aware_helpers(adapter) -> None:
    pair = (jnp.ones(2), jnp.arange(2.0))
    zero = adapter.zeros_like(pair)
    assert adapter.close((jnp.zeros(2), jnp.zeros(2)), zero)
    assert adapter.close(pair, adapter.add(None, pair))
    assert adapter.close((jnp.ones(2), jnp.arange(2.0) * 2), adapter.add(pair, (None, pair[1])))
    host = adapter.to_host(pair)
    assert isinstance(host[0], np.ndarray)
    assert not adapter.close(pair, (jnp.ones(2),))


@jax.custom_vjp
def _custom_sin(a):
    return jnp.sin(a)


_custom_sin.defvjp(lambda a: (jnp.sin(a), a), lambda a, g: (g * jnp.cos(a),))


def test_opaque_region_is_refused_and_unbindable(adapter) -> None:
    trace = capture(lambda v: _custom_sin(v) + 1.0, _x(3))
    regions = [op for op in trace._layer_list if func(op).startswith("region")]
    assert regions, "custom_vjp no longer captures as an opaque region"
    for op in regions:
        reason = adapter.can_replay(trace, op)
        assert reason is not None and reason.startswith("opaque region"), reason
        assert adapter.recipe(trace, op) is None
    recipes = {op.label: adapter.recipe(trace, op) for op in equation_ops(trace)}
    trace.jax_region_captures = ()  # a loaded trace: region captures are not rebuilt
    with pytest.raises(ReplayRefusal):
        adapter.bind_recipes(trace, recipes)
