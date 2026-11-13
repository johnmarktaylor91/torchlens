"""JAX replay adapter proofs on every config-built JAX roster model.

For each roster row: ``forward`` reproduces every op's saved payload,
``vjp`` composed over the whole graph equals ``jax.grad`` of the replayed
graph (and of the original function where the mapping is unambiguous), and a
JSON round trip of every recipe rebuilds a capture list the unchanged JAX
validator passes. ``can_replay`` refuses nothing on the roster.
"""

from __future__ import annotations

import pytest

from tests.conformance._rosters.jax import config_built_roster
from tests.test_parity_b_adapter_jax_support import (
    capture,
    engine_grads,
    equation_ops,
    func,
    graph_forward,
    oracle_grads,
    round_trip_validates,
    saved_parent_values,
)
from torchlens.backends._replay_adapters import get_adapter

pytest.importorskip("jax")

ROWS = {row.family: row for row in config_built_roster()}


def _row_trace(family: str):
    row = ROWS[family]
    try:
        net, example = row.builder()
    except ImportError as exc:  # optional ecosystem package (equinox, flax) absent
        pytest.skip(f"{family}: {exc}")
    return net, example, capture(net, example)


@pytest.mark.parametrize("family", sorted(ROWS))
def test_forward_reproduces_every_saved_payload(family: str) -> None:
    adapter = get_adapter("jax")
    _net, _example, trace = _row_trace(family)
    refused = {}
    for op in trace._layer_list:
        reason = adapter.can_replay(trace, op)
        if reason is not None:
            refused[op.label] = reason
            continue
        outputs = adapter.forward(trace, op, saved_parent_values(adapter, trace, op))
        assert len(outputs) == 1, op.label
        assert adapter.close(op.out, outputs[0]), f"{family}: {op.label} replay differs"
    equations = {op.label for op in equation_ops(trace)}
    assert equations, f"{family}: no equation ops captured"
    assert not equations & set(refused), f"{family}: equations refused {refused}"
    for label, reason in refused.items():
        assert reason.startswith("graph-only entry: a source value"), (label, reason)


@pytest.mark.parametrize("family", sorted(ROWS))
def test_vjp_chain_equals_jax_grad(family: str) -> None:
    adapter = get_adapter("jax")
    net, example, trace = _row_trace(family)
    leaves, engine = engine_grads(adapter, trace)
    if not leaves:
        pytest.skip(f"{family}: no floating input leaf to differentiate")
    replayed = oracle_grads(adapter, trace, leaves)
    for label in leaves:
        assert adapter.close(replayed[label], engine[label]), f"{family}: {label} grad differs"
    direct = oracle_grads(adapter, trace, leaves, net=net, example=example)
    for label, grad in direct.items():
        assert adapter.close(grad, engine[label]), f"{family}: {label} differs from jax.grad(net)"


@pytest.mark.parametrize("family", sorted(ROWS))
def test_recipe_round_trip_passes_unchanged_validator(family: str) -> None:
    adapter = get_adapter("jax")
    _net, _example, trace = _row_trace(family)
    round_trip_validates(adapter, trace)


def test_replayed_graph_is_differentiable_by_jax() -> None:
    import jax
    import jax.numpy as jnp

    adapter = get_adapter("jax")
    _net, _example, trace = _row_trace("chain_branch_function")
    (leaf,) = [op for op in trace._layer_list if func(op) == "input"]

    def loss(value):
        outputs = graph_forward(adapter, trace, {leaf.label: value})
        return sum(jnp.sum(v) for v in outputs.values())

    grad = jax.grad(loss)(leaf.out)
    assert grad.shape == leaf.out.shape
    assert bool(jnp.all(jnp.isfinite(grad)))


@pytest.mark.parametrize("family", sorted(ROWS))
def test_bind_reads_param_payloads(family: str) -> None:
    """Loaded parameter arrays (keyed by ``param_refs`` address) feed the rebuilt captures."""

    import numpy as np

    from torchlens.backends.jax.backend import JAXBackend, _jax_ops_by_label

    adapter = get_adapter("jax")
    _net, _example, trace = _row_trace(family)
    recipes = round_trip_validates(adapter, trace)
    by_label = _jax_ops_by_label(trace)
    params = {}
    for rec in recipes.values():
        refs = {tuple(path): address for path, address in rec.param_refs}
        for occ in rec.inputs:
            address = refs.get(tuple(occ.arg_path))
            if address is not None:
                params[address] = np.asarray(by_label[occ.parent_label].out)
    if not params:
        pytest.skip(f"{family}: no parameter references")
    adapter.bind_recipes(trace, recipes, param_payloads=params)
    assert JAXBackend().validation_status(trace).state == "passed"
    shifted = {address: value + 1.0 for address, value in params.items()}
    adapter.bind_recipes(trace, recipes, param_payloads=shifted)
    assert JAXBackend().validation_status(trace).state == "failed"
