"""Helpers for the JAX replay adapter tests (graph forward, reverse engine, round trip)."""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np


def capture(net: Any, example: Any) -> Any:
    """Capture ``net`` on ``example`` with the JAX backend."""

    import torchlens as tl

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return tl.trace(net, example, backend="jax")


def func(op: Any) -> str:
    """The op's recorded function name (``input``, ``constant`` or the primitive)."""

    from torchlens._utils._record_group_table import record_member

    return str(record_member(op, "func_name", "") or "")


def equation_ops(trace: Any) -> tuple[Any, ...]:
    from torchlens.backends.jax.backend import _jax_equation_ops

    return _jax_equation_ops(trace)


def saved_parent_values(adapter: Any, trace: Any, op: Any) -> list[Any]:
    """Saved payload of each parent occurrence, as the validator reads them."""

    from torchlens.backends.jax.backend import _jax_ops_by_label

    by_label = _jax_ops_by_label(trace)
    return [by_label[occ.parent_label].out for occ in adapter.step(trace, op).inputs]


def graph_forward(adapter: Any, trace: Any, overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Replay the whole graph through ``forward``; return the output-op values."""

    values: dict[str, Any] = {}
    for op in trace._layer_list:
        step = adapter.step(trace, op)
        label = step.output_slots[0]
        if label in overrides:
            values[label] = overrides[label]
            continue
        (values[label],) = adapter.forward(
            trace, op, [values[occ.parent_label] for occ in step.inputs]
        )
    return {op.label: values[op.label] for op in trace.output_ops}


def _float_leaves(trace: Any) -> dict[str, Any]:
    import jax.numpy as jnp

    return {
        op.label: op.out
        for op in trace._layer_list
        if func(op) == "input"
        and not isinstance(op.out, tuple)
        and jnp.issubdtype(op.out.dtype, jnp.floating)
    }


def _weights(trace: Any) -> dict[str, Any]:
    """Fixed random cotangent per floating output op (seeded, deterministic)."""

    import jax.numpy as jnp

    rng = np.random.default_rng(0)
    weights = {}
    for op in trace.output_ops:
        value = op.out
        if not isinstance(value, tuple) and jnp.issubdtype(value.dtype, jnp.floating):
            weights[op.label] = jnp.asarray(rng.standard_normal(value.shape), dtype=value.dtype)
    return weights


def engine_grads(adapter: Any, trace: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reverse accumulation over ``adapter.vjp`` from the weighted outputs."""

    leaves = _float_leaves(trace)
    from torchlens.backends.jax.backend import _jax_ops_by_label

    by_label = _jax_ops_by_label(trace)
    cot: dict[str, Any] = dict(_weights(trace))
    for op in reversed(trace._layer_list):
        step = adapter.step(trace, op)
        label = step.output_slots[0]
        if label not in cot or not step.inputs:
            continue
        parents = [by_label[occ.parent_label].out for occ in step.inputs]
        grads = adapter.vjp(trace, op, parents, (cot[label],))
        assert len(grads) == len(step.inputs), label
        for occ, grad in zip(step.inputs, grads, strict=True):
            if grad is not None:
                cot[occ.parent_label] = adapter.add(cot.get(occ.parent_label), grad)
    return leaves, {label: cot.get(label, adapter.zeros_like(v)) for label, v in leaves.items()}


def oracle_grads(
    adapter: Any,
    trace: Any,
    leaves: Mapping[str, Any],
    *,
    net: Any = None,
    example: Any = None,
) -> dict[str, Any]:
    """``jax.grad`` of the weighted loss: over the replayed graph, or over ``net``.

    The ``net`` oracle runs only for a raw single-array function whose single
    floating input leaf and single floating output map one to one.
    """

    import jax
    import jax.numpy as jnp

    weights = _weights(trace)
    labels = tuple(leaves)
    if net is None:

        def loss(*values: Any) -> Any:
            outputs = graph_forward(adapter, trace, dict(zip(labels, values, strict=True)))
            return sum(jnp.sum(outputs[k] * w) for k, w in weights.items())

        grads = jax.grad(loss, argnums=tuple(range(len(labels))))(*leaves.values())
        return dict(zip(labels, grads, strict=True))
    if len(labels) != 1 or len(weights) != 1 or not callable(net) or isinstance(example, tuple):
        return {}
    if hasattr(net, "__dict__") and type(net).__module__.split(".")[0] in ("equinox", "flax"):
        return {}
    ((_, weight),) = weights.items()

    def direct(value: Any) -> Any:
        return jnp.sum(net(value) * weight)

    return {labels[0]: jax.grad(direct)(example)}


def round_trip_validates(adapter: Any, trace: Any) -> dict[str, Any]:
    """Strip captures, JSON round trip every recipe, bind, validate; return recipes."""

    from torchlens.backends._replay_adapters import decode_recipe, encode_recipe
    from torchlens.backends.jax.backend import JAXBackend

    original = tuple(trace.jax_equation_captures)
    recipes = {}
    for op in equation_ops(trace):
        found = adapter.recipe_or_reason(trace, op)
        assert not isinstance(found, str), f"{op.label}: no recipe ({found})"
        tree = json.loads(json.dumps(encode_recipe(found)))
        recipes[op.label] = decode_recipe(tree, backend="jax", allowlist=adapter.ALLOWLIST)
    trace.jax_equation_captures = ()
    assert JAXBackend().validation_status(trace).state == "failed"
    adapter.bind_recipes(trace, recipes)
    status = JAXBackend().validation_status(trace)
    assert status.state == "passed", status.message
    assert status.num_replayed_ops == len(original)
    for old, new in zip(original, trace.jax_equation_captures, strict=True):
        assert (old.index, old.kind, old.primitive, old.invars) == (
            new.index,
            new.kind,
            new.primitive,
            new.invars,
        )
        assert old.primitive_obj is new.primitive_obj
        assert set(old.params) == set(new.params)
    return recipes
