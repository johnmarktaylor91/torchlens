"""JAX interpreter early stop, retention and pullback hooks (lane b-jax-interp).

A halted run stores the frontier capture, then unwinds every nested frame before any
further primitive binds. Its captures equal the full run's captures op for op up to the
frontier, and each prefix capture replays to its saved output.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from torchlens.backends.jax import _interp_hooks  # noqa: E402
from torchlens.backends.jax._interp_hooks import InterpreterHooks  # noqa: E402
from torchlens.backends.jax.jaxpr import (  # noqa: E402
    JaxEquationCapture,
    interpret_closed_jaxpr_with_inlining,
    interpret_pullback_jaxpr,
    replay_equation,
)

pytestmark = [pytest.mark.backend_jax, pytest.mark.optional]


def _nested(x: Any) -> Any:
    """Exercise jit, cond, scan and a while loop, with a cos in the suffix."""

    def body(carry: Any, item: Any) -> tuple[Any, Any]:
        """One scan step with a sin inside."""

        updated = jnp.sin(carry) + item
        return updated, updated * 2.0

    carry, ys = jax.lax.scan(body, x, jnp.arange(3.0))
    branch = jax.lax.cond(jnp.sum(carry) > 0, lambda v: v * 3.0, lambda v: v - 1.0, carry)
    looped = jax.lax.while_loop(lambda s: s[0] < 2, lambda s: (s[0] + 1, s[1] + 1.0), (0, branch))
    jitted = jax.jit(lambda v: jnp.tanh(v) * 2.0)(looped[1])
    return jnp.cos(jitted) + jnp.sum(ys)


def _closed_and_args() -> tuple[Any, list[Any]]:
    """Return the closed jaxpr of ``_nested`` and its flat input."""

    x = jnp.linspace(0.1, 0.4, 3)
    return jax.make_jaxpr(_nested)(x), [x]


def _signature(capture: Any) -> tuple[Any, ...]:
    """Return a structural identity for one capture (primitive, path, values)."""

    values = tuple(np.asarray(value).tolist() for value in capture.output_values)
    return (capture.kind, capture.primitive, capture.source_path, values)


def _primitive_index(captures: Any, primitive: str, occurrence: int = 0) -> int:
    """Return the capture index of the ``occurrence``-th ``primitive`` capture."""

    matches = [c.index for c in captures if c.kind == "primitive" and c.primitive == primitive]
    return matches[occurrence]


@pytest.mark.parametrize(
    ("primitive", "occurrence"),
    [("sin", 1), ("mul", 0), ("tanh", 0), ("add", 0)],
)
def test_halted_prefix_equals_full_run_prefix(primitive: str, occurrence: int) -> None:
    """Halting inside scan, cond, while or jit keeps exactly the full run's prefix."""

    closed, args = _closed_and_args()
    full = interpret_closed_jaxpr_with_inlining(closed, args)
    stop = _primitive_index(full.captures, primitive, occurrence)
    halted = interpret_closed_jaxpr_with_inlining(
        closed, args, hooks=InterpreterHooks(halt=lambda capture: capture.index == stop)
    )

    assert halted.outputs == ()
    assert halted.halt_frontier is halted.captures[-1]
    assert halted.halt_frontier.index == stop
    assert len(halted.captures) == stop + 1
    assert [_signature(c) for c in halted.captures] == [
        _signature(c) for c in full.captures[: stop + 1]
    ]
    for capture in halted.equations:
        replayed = replay_equation(capture, capture.input_values)
        for got, saved in zip(replayed, capture.output_values, strict=True):
            np.testing.assert_allclose(np.asarray(got), np.asarray(saved), rtol=1e-6)


def test_suffix_primitive_never_binds(monkeypatch: pytest.MonkeyPatch) -> None:
    """After the frontier no primitive binds: the suffix ``cos`` is never called."""

    closed, args = _closed_and_args()
    full = interpret_closed_jaxpr_with_inlining(closed, args)
    stop = _primitive_index(full.captures, "tanh")
    binds: list[str] = []
    cos_p = jax.lax.cos_p
    original = cos_p.bind

    def counting_bind(*bind_args: Any, **params: Any) -> Any:
        """Count binds of ``cos``."""

        binds.append("cos")
        return original(*bind_args, **params)

    monkeypatch.setattr(cos_p, "bind", counting_bind)
    interpret_closed_jaxpr_with_inlining(
        closed, args, hooks=InterpreterHooks(halt=lambda capture: capture.index == stop)
    )
    assert binds == []
    interpret_closed_jaxpr_with_inlining(closed, args)
    assert binds == ["cos"]


def test_value_dependent_halt_stops_on_concrete_values() -> None:
    """A stop that reads the concrete output value stops at the first match."""

    closed, args = _closed_and_args()

    def first_negative(capture: Any) -> bool:
        """Stop at the first primitive whose output has a value above 1."""

        out = capture.output_values[0] if capture.output_values else None
        return capture.kind == "primitive" and bool(np.any(np.asarray(out) > 1.0))

    halted = interpret_closed_jaxpr_with_inlining(
        closed, args, hooks=InterpreterHooks(halt=first_negative)
    )
    assert halted.halt_frontier is not None
    assert first_negative(halted.halt_frontier)
    assert not any(first_negative(c) for c in halted.captures[:-1])


def test_unhooked_run_is_unchanged() -> None:
    """With no hooks the result has no released captures and no frontier."""

    closed, args = _closed_and_args()
    result = interpret_closed_jaxpr_with_inlining(closed, args)
    np.testing.assert_allclose(
        np.asarray(result.outputs[0]), np.asarray(_nested(args[0])), rtol=1e-6
    )
    assert result.released_capture_indices == frozenset()
    assert result.halt_frontier is None


class _FakeRetention:
    """Stand-in for ``jax/_retention.py``: keep only even capture indexes."""

    @staticmethod
    def retention_policy(session: Any) -> Any:
        """Return ``keep(index)``."""

        assert session == "session"
        return lambda index: index % 2 == 0


def test_retention_releases_payloads_and_keeps_outputs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Released primitive captures hold shape placeholders; outputs stay exact."""

    monkeypatch.setattr(_interp_hooks, "_retention_module", lambda: _FakeRetention)
    closed, args = _closed_and_args()
    full = interpret_closed_jaxpr_with_inlining(closed, args)
    sparse = interpret_closed_jaxpr_with_inlining(
        closed, args, hooks=InterpreterHooks(session="session")
    )

    np.testing.assert_allclose(np.asarray(sparse.outputs[0]), np.asarray(full.outputs[0]))
    assert len(sparse.captures) == len(full.captures)
    expected = {c.index for c in full.captures if c.kind == "primitive" and c.index % 2}
    assert sparse.released_capture_indices == expected
    for dense, light in zip(full.captures, sparse.captures, strict=True):
        if light.index in expected:
            assert all(isinstance(v, jax.ShapeDtypeStruct) for v in light.output_values)
            assert [v.shape for v in light.output_values] == [
                np.shape(v) for v in dense.output_values
            ]
        else:
            assert _signature(light) == _signature(dense)


def test_pullback_entry_binds_scatter_add() -> None:
    """The backward entry interprets a pullback holding ``scatter-add``."""

    def loss(x: Any) -> Any:
        """Gather then reduce, so the pullback scatters."""

        return jnp.sum(jnp.sin(x[jnp.array([0, 2, 2])]))

    x = jnp.linspace(0.1, 0.5, 4)
    _value, pullback = jax.vjp(loss, x)
    seed = jnp.ones(())
    closed = jax.make_jaxpr(pullback)(seed)
    primitives = {eqn.primitive.name for eqn in closed.jaxpr.eqns}
    result = interpret_pullback_jaxpr(closed, [seed])

    np.testing.assert_allclose(
        np.asarray(result.outputs[0]), np.asarray(jax.grad(loss)(x)), rtol=1e-6
    )
    assert all(isinstance(c, JaxEquationCapture) for c in result.captures)
    if primitives & _interp_hooks.COMBINER_JAXPR_PRIMITIVES:
        with pytest.raises(ValueError, match="nested"):
            interpret_closed_jaxpr_with_inlining(closed, [seed])
