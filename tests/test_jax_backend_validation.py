"""JAX backend replay-validation spine tests."""

from __future__ import annotations

from collections.abc import Mapping
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


def test_jax_replay_unimplemented_kind_raises_bcf_message() -> None:
    """Control-flow replay kinds should raise the planned B-CF handoff error."""

    with pytest.raises(NotImplementedError, match="kind scan_read lands in B-CF"):
        replay_equation(_capture_for_kind("scan_read"))


def test_jax_replay_unknown_kind_raises_actionable_error() -> None:
    """Unknown replay kinds should fail through the dispatch table."""

    with pytest.raises(NotImplementedError, match="not registered.*primitive"):
        replay_equation(_capture_for_kind("future_kind"))
