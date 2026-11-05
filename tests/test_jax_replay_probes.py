"""JAX replay of multi-output parents and the non-uniform perturbation probes.

A ``split`` materializes as one op holding the tuple of its outputs; each child must
replay with the element it consumed. An ``argmax`` or a dominated ``max`` cannot be
moved by uniform shifts, so the oracle also tries the torch validator's single-entry
lift and extrema crossing value. A child no candidate moves still fails.
"""

from __future__ import annotations

from typing import Any

import pytest

import torchlens as tl

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

pytestmark = [pytest.mark.backend_jax]


def _trace(fn: Any, x: Any) -> Any:
    """Trace ``fn(None, x)`` on the JAX backend."""

    return tl.trace(fn, (None, x), backend="jax")


def _failed_reasons(status: Any) -> set[str]:
    """Return the reasons of the failed decisions on a validation status."""

    return {str(row["reason"]) for row in status.decisions if row["decision"] == "failed"}


def _halves(_: None, x: Any) -> Any:
    """Split ``x`` in two along axis 1 and recombine the flattened halves."""

    left, right = jnp.split(x, 2, axis=1)
    return left.reshape(-1) * 2.0 + right.reshape(-1)


def test_children_of_a_multi_output_split_replay_their_own_element() -> None:
    """Each ``reshape`` child of ``split`` replays with its element, metadata invariants included."""

    trace = _trace(_halves, jnp.arange(8.0, dtype=jnp.float32).reshape(2, 4))
    assert any(op.func_name == "split" for op in trace._layer_list)

    trace.validate_forward_pass([], validate_metadata=True)
    status = trace.validation_replay_status

    assert status.state == "passed", status.message


def test_corrupted_multi_output_payload_still_fails() -> None:
    """A saved ``split`` payload whose element drifted fails replay instead of passing."""

    trace = _trace(_halves, jnp.arange(8.0, dtype=jnp.float32).reshape(2, 4))
    split_op = next(op for op in trace._layer_list if op.func_name == "split")
    saved = split_op.out
    assert isinstance(saved, tuple) and len(saved) == 2
    split_op._internal_set("out", (saved[0] + 1.0, saved[1]))

    trace.validate_forward_pass([], validate_metadata=False)
    status = trace.validation_replay_status

    assert status.state == "failed"
    assert "replay_mismatch" in _failed_reasons(status)


def test_argmax_validates_through_the_single_entry_lift() -> None:
    """Uniform shifts never move an ``argmax``; lifting one entry does."""

    def winner(_: None, x: Any) -> Any:
        """Return the per-row argmax of ``2 * x`` as float."""

        return jnp.argmax(x * 2.0, axis=-1).astype(jnp.float32)

    trace = _trace(winner, jnp.arange(6.0, dtype=jnp.float32).reshape(2, 3))
    trace.validate_forward_pass([], validate_metadata=False)
    status = trace.validation_replay_status

    assert status.state == "passed", status.message


def test_dominated_max_operand_validates_through_the_crossing_value() -> None:
    """``max(|x|, 100)`` with small ``x``: only a value past 100 moves the output."""

    def floored(_: None, x: Any) -> Any:
        """Return ``max(|x|, 100)``."""

        return jnp.maximum(jnp.abs(x), 100.0)

    trace = _trace(floored, jnp.arange(6.0, dtype=jnp.float32).reshape(2, 3) / 10.0)
    trace.validate_forward_pass([], validate_metadata=False)
    status = trace.validation_replay_status

    assert status.state == "passed", status.message


def test_argmax_over_a_single_entry_axis_still_fails() -> None:
    """An ``argmax`` over an axis of size one is 0 for every input: no probe moves it."""

    def trivial_winner(_: None, x: Any) -> Any:
        """Return the argmax over a size-one axis, as float."""

        return jnp.argmax(x * 2.0, axis=-1).astype(jnp.float32)

    trace = _trace(trivial_winner, jnp.arange(3.0, dtype=jnp.float32).reshape(3, 1))
    trace.validate_forward_pass([], validate_metadata=False)
    status = trace.validation_replay_status

    assert status.state == "failed"
    assert "perturbation_insensitive" in _failed_reasons(status)


def test_float_child_that_ignores_its_parent_still_fails() -> None:
    """``x * 0.0`` stays zero under every candidate, the lift included."""

    def vanishing(_: None, x: Any) -> Any:
        """Return zero for every input."""

        return x * 0.0

    trace = _trace(vanishing, jnp.arange(6.0, dtype=jnp.float32).reshape(2, 3))
    trace.validate_forward_pass([], validate_metadata=False)
    status = trace.validation_replay_status

    assert status.state == "failed"
    assert "perturbation_insensitive" in _failed_reasons(status)
