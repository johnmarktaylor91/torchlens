"""Probe tinygrad interception and realization hook behavior.

Run with:
  DEV=PYTHON nice -n 19 ionice -c3 /tmp/jaxtg/tinygrad-venv/bin/python \
    .research/spikes/tinygrad/probe_interception_realization.py
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from tinygrad import Tensor
from tinygrad.uop.ops import Ops, UOp

import tinygrad.tensor as tensor_mod


def uop_signature(tensor: Tensor) -> tuple[tuple[str, tuple[str, ...], str, str], ...]:
    """Return a stable-enough structural signature for a tiny UOp graph.

    Parameters
    ----------
    tensor
        Tensor whose current UOp lineage should be summarized.

    Returns
    -------
    tuple[tuple[str, tuple[str, ...], str, str], ...]
        Toposorted UOp operation names, source operation names, dtypes, and args.
    """
    return tuple(
        (uop.op.name, tuple(src.op.name for src in uop.src), str(uop.dtype), repr(uop.arg))
        for uop in tensor.uop.toposort()
    )


def linear_ops(linear: UOp) -> tuple[str, ...]:
    """Return operation names from a tinygrad LINEAR schedule.

    Parameters
    ----------
    linear
        LINEAR UOp passed to tinygrad's realization runner.

    Returns
    -------
    tuple[str, ...]
        Direct child operation names for the schedule.
    """
    assert linear.op is Ops.LINEAR
    return tuple(call.op.name for call in linear.src)


def build_expression() -> Tensor:
    """Build a toy lazy expression with Tensor API and UOp graph evidence.

    Returns
    -------
    Tensor
        A two-element tensor expression that requires realization for payload reads.
    """
    return (Tensor([1.0, 2.0]) + 1.0).relu()


def patch_tensor_apply(events: list[str]) -> Callable[..., Tensor]:
    """Wrap ``Tensor._apply_uop`` to inventory Tensor-API event visibility.

    Parameters
    ----------
    events
        Mutable list receiving created UOp names.

    Returns
    -------
    Callable[..., Tensor]
        Original method for restoration.
    """
    original = Tensor._apply_uop

    def wrapped(self: Tensor, fxn: Callable[..., UOp], *x: Tensor, **kwargs: Any) -> Tensor:
        """Record the Tensor event and delegate unchanged."""
        ret = original(self, fxn, *x, **kwargs)
        events.append(ret.uop.op.name)
        return ret

    Tensor._apply_uop = wrapped  # type: ignore[method-assign]
    return original


def patch_run_linear(events: list[tuple[str, ...]]) -> Callable[..., Any]:
    """Wrap the realization runner imported by ``tinygrad.tensor``.

    Parameters
    ----------
    events
        Mutable list receiving direct LINEAR child op names.

    Returns
    -------
    Callable[..., Any]
        Original runner for restoration.
    """
    original = tensor_mod.run_linear

    def wrapped(linear: UOp, *args: Any, **kwargs: Any) -> Any:
        """Record the schedule-level hook and delegate unchanged."""
        events.append(linear_ops(linear))
        return original(linear, *args, **kwargs)

    tensor_mod.run_linear = wrapped
    return original


def assert_interception_inventory() -> None:
    """Assert Tensor API, UOp graph, and scheduler/realize hooks are all observable."""
    tensor_events: list[str] = []
    original_apply = patch_tensor_apply(tensor_events)
    try:
        expr = build_expression()
    finally:
        Tensor._apply_uop = original_apply  # type: ignore[method-assign]

    assert "ADD" in tensor_events, tensor_events
    assert "WHERE" in tensor_events, tensor_events

    graph_ops = tuple(name for name, _src, _dtype, _arg in uop_signature(expr))
    assert graph_ops[-1] == "WHERE", graph_ops
    assert "ADD" in graph_ops, graph_ops
    assert "CMPLT" in graph_ops, graph_ops

    schedule_events: list[tuple[str, ...]] = []
    original_run_linear = patch_run_linear(schedule_events)
    try:
        assert expr.tolist() == [2.0, 3.0]
    finally:
        tensor_mod.run_linear = original_run_linear

    assert len(schedule_events) == 1, schedule_events
    assert schedule_events[0], schedule_events


def assert_realization_non_interference() -> None:
    """Assert observational hooks and payload reads do not rewrite source lineage."""
    baseline = build_expression()
    baseline_before = uop_signature(baseline)
    assert baseline.tolist() == [2.0, 3.0]
    assert uop_signature(baseline) == baseline_before

    observed = build_expression()
    observed_before = uop_signature(observed)
    schedule_events: list[tuple[str, ...]] = []
    original_run_linear = patch_run_linear(schedule_events)
    try:
        assert observed.tolist() == [2.0, 3.0]
    finally:
        tensor_mod.run_linear = original_run_linear

    assert uop_signature(observed) == observed_before
    assert len(schedule_events) == 1, schedule_events


def main() -> None:
    """Run all interception and realization probes."""
    assert_interception_inventory()
    assert_realization_non_interference()
    print("PASS probe_interception_realization")


if __name__ == "__main__":
    main()
