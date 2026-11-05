"""JAX perturbation evidence for bool-output equations (comparisons and predicates).

A bool equation is a threshold op: the additive perturbation steps routinely fail to
cross it, so the oracle also tries a dtype battery and, for a comparison, the comparand
straddle, as the torch validator does. A bool equation whose output no substitute can move still fails.
"""

from __future__ import annotations

from typing import Any

import pytest

import torchlens as tl

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

pytestmark = [pytest.mark.backend_jax]


def _status(fn: Any, x: Any) -> Any:
    """Trace ``fn(None, x)`` on the JAX backend and return its validation status."""

    trace = tl.trace(fn, (None, x), backend="jax")
    trace.validate_forward_pass([], validate_metadata=False)
    return trace.validation_replay_status


def _failed_reasons(status: Any) -> set[str]:
    """Return the reasons of the failed decisions on a validation status."""

    return {str(row["reason"]) for row in status.decisions if row["decision"] == "failed"}


def test_comparison_against_constant_validates_when_steps_cannot_cross() -> None:
    """``mask != 0`` on an all-ones mask: additive steps stay nonzero, the straddle flips it."""

    def masked(_: None, mask: Any) -> Any:
        """Return the nonzero pattern of a float mask as float."""

        return (mask != 0.0).astype(jnp.float32)

    status = _status(masked, jnp.ones((2, 4), dtype=jnp.float32))

    assert status.state == "passed", status.message


def test_self_comparison_nan_test_validates() -> None:
    """``isnan`` lowers to ``x != x``; only a non-finite substitute moves it."""

    def nan_mask(_: None, x: Any) -> Any:
        """Return where ``x`` is NaN, as float."""

        return jnp.isnan(x * 2.0).astype(jnp.float32)

    status = _status(nan_mask, jnp.arange(6.0, dtype=jnp.float32).reshape(2, 3))

    assert status.state == "passed", status.message


def test_integer_self_comparison_still_fails() -> None:
    """``i != i`` on integers is constant False: the battery has no NaN, so it still fails."""

    def int_self_compare(_: None, i: Any) -> Any:
        """Return ``(i + 1) != (i + 1)`` as float, which is False for every integer."""

        shifted = i + 1
        return (shifted != shifted).astype(jnp.float32)

    status = _status(int_self_compare, jnp.arange(6, dtype=jnp.int32))

    assert status.state == "failed"
    assert "perturbation_insensitive" in _failed_reasons(status)
