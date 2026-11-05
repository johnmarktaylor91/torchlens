"""The JAX backend's zero-annihilator proof (torch parity, ``multiplicative_zero_annihilator``).

A ``mul`` or ``dot_general`` operand whose saved co-operand is exactly zero cannot move
the output (``x * 0 == 0``), so no perturbation candidate moves it. The proof credits
that edge only on exact saved zeros with an exactly zero saved output. Every
unintended case of the same shape still fails ``perturbation_insensitive``, and the
proof never replaces the replay.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

import torchlens as tl

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

pytestmark = [pytest.mark.backend_jax]

_REASON = "multiplicative_zero_annihilator"


def _validated_status(fn: Any, args: tuple[Any, ...]) -> Any:
    """Trace ``fn`` on the JAX backend, run replay validation and return the status."""

    trace = tl.trace(fn, args, backend="jax")
    trace.validate_forward_pass([], validate_metadata=False)
    return trace.validation_replay_status


def _reasons(status: Any, decision: str) -> set[str]:
    """Return the reasons of the decisions of one kind on a validation status."""

    return {str(row["reason"]) for row in status.decisions if row["decision"] == decision}


@pytest.fixture
def no_candidates(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Make every perturbation candidate stream empty, so each data edge looks dead.

    This plants the bug the proof must not hide: an edge no candidate moves. Only
    the proof's own conditions can then save an equation.
    """

    from torchlens.backends.jax import _trace_graph_helpers

    def _empty(*_args: Any, **_kwargs: Any) -> Iterator[Any]:
        """Yield no candidates."""

        return iter(())

    monkeypatch.setattr(_trace_graph_helpers, "_candidate_stream", _empty)
    yield


def _times_zero(_: None, x: Any) -> Any:
    """Return ``x * 0.0``."""

    return x * 0.0


def _times_two(_: None, x: Any) -> Any:
    """Return ``x * 2.0``."""

    return x * 2.0


def _square(_: None, x: Any) -> Any:
    """Return ``x * x``."""

    return x * x


def _plus_zero(_: None, x: Any) -> Any:
    """Return ``x + 0.0``."""

    return x + 0.0


def _matmul(params: dict[str, Any], x: Any) -> Any:
    """Return ``x @ params["w"]``."""

    return jnp.dot(x, params["w"])


def test_mul_by_a_zero_literal_is_credited() -> None:
    """``x * 0.0`` validates with the annihilator exemption instead of failing."""

    status = _validated_status(_times_zero, (None, jnp.arange(6.0, dtype=jnp.float32)))

    assert status.state == "passed", status.message
    assert _REASON in _reasons(status, "exempted")
    assert status.exempted_reason_counts.get(_REASON) == 1


def test_dot_general_with_both_operands_zero_is_credited() -> None:
    """``zeros @ zeros``: each operand is annihilated by the other's saved zeros."""

    params = {"w": jnp.zeros((3, 2), dtype=jnp.float32)}
    status = _validated_status(_matmul, (params, jnp.zeros((2, 3), dtype=jnp.float32)))

    assert status.state == "passed", status.message
    assert _REASON in _reasons(status, "exempted")


@pytest.mark.usefixtures("no_candidates")
def test_dead_mul_edge_with_a_nonzero_co_operand_still_fails() -> None:
    """Planted dead edge on ``x * 2.0``: a nonzero co-operand proves nothing."""

    status = _validated_status(_times_two, (None, jnp.arange(6.0, dtype=jnp.float32)))

    assert status.state == "failed"
    assert "perturbation_insensitive" in _reasons(status, "failed")
    assert _REASON not in _reasons(status, "exempted")


@pytest.mark.usefixtures("no_candidates")
def test_dead_dot_general_edge_with_one_nonzero_weight_still_fails() -> None:
    """Planted dead edge on ``zeros @ w``: one nonzero entry in ``w`` refuses the proof."""

    weight = jnp.zeros((3, 2), dtype=jnp.float32).at[1, 0].set(1.0)
    status = _validated_status(_matmul, ({"w": weight}, jnp.zeros((2, 3), dtype=jnp.float32)))

    assert status.state == "failed"
    assert "perturbation_insensitive" in _reasons(status, "failed")


@pytest.mark.usefixtures("no_candidates")
def test_dead_edge_filling_both_mul_slots_still_fails() -> None:
    """Planted dead edge on ``x * x`` with ``x`` zero: a parent in both slots is never credited."""

    status = _validated_status(_square, (None, jnp.zeros((4,), dtype=jnp.float32)))

    assert status.state == "failed"
    assert "perturbation_insensitive" in _reasons(status, "failed")


@pytest.mark.usefixtures("no_candidates")
def test_dead_edge_on_a_non_bilinear_primitive_still_fails() -> None:
    """Planted dead edge on ``x + 0.0`` with ``x`` zero: only ``mul``/``dot_general`` qualify."""

    status = _validated_status(_plus_zero, (None, jnp.zeros((4,), dtype=jnp.float32)))

    assert status.state == "failed"
    assert "perturbation_insensitive" in _reasons(status, "failed")


@pytest.mark.usefixtures("no_candidates")
def test_dead_mul_edge_with_a_non_finite_output_still_fails() -> None:
    """Planted dead edge on ``x * 0.0`` with an inf in ``x``: ``inf * 0`` is NaN, not zero."""

    x = jnp.asarray([jnp.inf, 1.0, 2.0], dtype=jnp.float32)
    status = _validated_status(_times_zero, (None, x))

    assert status.state == "failed"
    assert "perturbation_insensitive" in _reasons(status, "failed")


def test_proof_never_replaces_the_replay() -> None:
    """A corrupted saved output under ``x * 0.0`` fails replay; the proof is never reached."""

    trace = tl.trace(_times_zero, (None, jnp.arange(6.0, dtype=jnp.float32)), backend="jax")
    mul_op = next(op for op in trace._layer_list if op.func_name == "mul")
    mul_op._internal_set("out", jnp.ones_like(mul_op.out))

    trace.validate_forward_pass([], validate_metadata=False)
    status = trace.validation_replay_status

    assert status.state == "failed"
    assert "replay_mismatch" in _reasons(status, "failed")
    assert _REASON not in _reasons(status, "exempted")


def test_annihilator_reason_is_a_known_validation_reason() -> None:
    """The JAX exemption reuses torch's registered code; it adds no new reason."""

    from torchlens._validation_status import is_known_validation_reason, unknown_reason_codes

    status = _validated_status(_times_zero, (None, jnp.arange(6.0, dtype=jnp.float32)))

    assert is_known_validation_reason(_REASON)
    assert _REASON not in unknown_reason_codes(status)
