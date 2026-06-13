"""JAX backend replay-validation spine tests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any, cast

import pytest

from torchlens.backends.jax.jaxpr import (
    ALL_JAX_EQUATION_KINDS,
    JAX_REPLAY_HANDLERS,
    JaxEquationCapture,
    replay_equation,
)
from torchlens.ir.events import JaxEquationKind


def _capture_for_kind(kind: object) -> JaxEquationCapture:
    """Return a minimal replay capture for dispatch-table tests.

    Parameters
    ----------
    kind
        Replay kind to attach to the capture.

    Returns
    -------
    JaxEquationCapture
        Capture carrying the requested replay kind.
    """

    return JaxEquationCapture(
        index=0,
        kind=cast(JaxEquationKind, kind),
        primitive="dispatch_test",
        primitive_obj=None,
        input_values=(),
        output_values=(),
        params=cast(Mapping[str, Any], {}),
        source_path=("root", "0:dispatch_test"),
        invars=(),
        outvars=(),
        input_avals=(),
        output_avals=(),
        inlined=False,
    )


def test_jax_replay_kind_table_covers_foundation_kinds() -> None:
    """Replay dispatch should register every JAX interpreter foundation kind."""

    assert ALL_JAX_EQUATION_KINDS == (
        "primitive",
        "scan_read",
        "scan_stack",
        "cond_decision",
        "while_decision",
    )
    assert set(JAX_REPLAY_HANDLERS) == set(ALL_JAX_EQUATION_KINDS)


def test_jax_scan_replay_kinds_are_deterministic() -> None:
    """Scan helper replay kinds should reconstruct deterministic slice and stack values."""

    jnp = pytest.importorskip("jax.numpy")
    scan_input = jnp.arange(4, dtype=jnp.float32)
    read_capture = replace(
        _capture_for_kind("scan_read"),
        input_values=(scan_input,),
        params={"index": 2},
    )
    stack_capture = replace(
        _capture_for_kind("scan_stack"),
        input_values=(jnp.asarray(1.0), jnp.asarray(3.0)),
        params={"axis": 0},
    )

    assert replay_equation(read_capture)[0].item() == 2.0
    assert replay_equation(stack_capture)[0].tolist() == [1.0, 3.0]


def test_jax_decision_replay_kinds_validate_recorded_control_points() -> None:
    """Decision helper replay kinds should validate branch and stop-point semantics."""

    jnp = pytest.importorskip("jax.numpy")
    cond_capture = replace(
        _capture_for_kind("cond_decision"),
        input_values=(jnp.asarray(1, dtype=jnp.int32),),
        output_values=(jnp.asarray(1, dtype=jnp.int32),),
        params={"branch_index": 1, "num_branches": 2},
    )
    assert replay_equation(cond_capture)[0].item() == 1
    with pytest.raises(ValueError, match="different branch"):
        replay_equation(cond_capture, (jnp.asarray(0, dtype=jnp.int32),))

    jax = pytest.importorskip("jax")

    def cond_fn(limit: Any, value: Any) -> Any:
        """Return whether ``value`` is below ``limit``."""

        return value < limit

    def body_fn(step: Any, value: Any) -> Any:
        """Return the next loop carry."""

        return value + step

    cond_jaxpr = jax.make_jaxpr(cond_fn)(jnp.asarray(3), jnp.asarray(0))
    body_jaxpr = jax.make_jaxpr(body_fn)(jnp.asarray(1), jnp.asarray(0))
    while_capture = replace(
        _capture_for_kind("while_decision"),
        input_values=(jnp.asarray(3), jnp.asarray(1), jnp.asarray(0)),
        output_values=(jnp.asarray(3, dtype=jnp.int32), jnp.asarray(False)),
        params={
            "cond_jaxpr": cond_jaxpr,
            "body_jaxpr": body_jaxpr,
            "cond_nconsts": 1,
            "body_nconsts": 1,
            "iteration_count": 3,
            "final_predicate": False,
            "jax_max_control_flow_unroll": 8,
        },
    )
    iterations, final_predicate = replay_equation(while_capture)
    assert iterations.item() == 3
    assert final_predicate.item() is False
    with pytest.raises(ValueError, match="different point"):
        replay_equation(
            replace(
                while_capture,
                params={**while_capture.params, "iteration_count": 2},
            )
        )


def test_jax_replay_unknown_kind_raises_actionable_error() -> None:
    """Unknown replay kinds should fail through the dispatch table."""

    with pytest.raises(NotImplementedError, match="not registered.*primitive"):
        replay_equation(_capture_for_kind("future_kind"))
