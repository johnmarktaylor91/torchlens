"""Probe TinyJit captured execution hooks and payload read behavior.

Run with:
  DEV=PYTHON nice -n 19 ionice -c3 /tmp/jaxtg/tinygrad-venv/bin/python \
    .research/spikes/tinygrad/probe_tinyjit_payload_round3.py
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from tinygrad import Tensor
from tinygrad.engine.jit import TinyJit
from tinygrad.uop.ops import UOp

import tinygrad.engine.jit as jit_mod
import tinygrad.tensor as tensor_mod


def values(tensor: Tensor) -> list[float]:
    """Read tensor values as a Python list.

    Parameters
    ----------
    tensor
        Tensor to realize through tinygrad's host-readable payload API.

    Returns
    -------
    list[float]
        Realized scalar values.
    """
    return list(tensor.tolist())


def uop_ops(uop: UOp) -> tuple[str, ...]:
    """Return operation names from a UOp lineage.

    Parameters
    ----------
    uop
        UOp root to summarize.

    Returns
    -------
    tuple[str, ...]
        Topologically sorted operation names.
    """
    return tuple(node.op.name for node in uop.toposort())


def tensor_ops(tensor: Tensor) -> tuple[str, ...]:
    """Return operation names from a tensor's current UOp lineage.

    Parameters
    ----------
    tensor
        Tensor to inspect.

    Returns
    -------
    tuple[str, ...]
        Topologically sorted operation names.
    """
    return uop_ops(tensor.uop)


def patch_tensor_run_linear(events: list[tuple[str, int, bool]]) -> Callable[..., Any]:
    """Wrap ordinary tensor realization to separate it from captured JIT execution.

    Parameters
    ----------
    events
        Mutable list receiving root op, child count, and JIT flag.

    Returns
    -------
    Callable[..., Any]
        Original runner for restoration.
    """
    original = tensor_mod.run_linear

    def wrapped(linear: UOp, *args: Any, **kwargs: Any) -> Any:
        """Record ordinary tensor realization and delegate unchanged."""
        events.append((linear.op.name, len(linear.src), bool(kwargs.get("jit", False))))
        return original(linear, *args, **kwargs)

    tensor_mod.run_linear = wrapped
    return original


def patch_jit_run_linear(events: list[tuple[str, int, bool]]) -> Callable[..., Any]:
    """Wrap the realization runner imported by TinyJit captured execution.

    Parameters
    ----------
    events
        Mutable list receiving root op, child count, and JIT flag.

    Returns
    -------
    Callable[..., Any]
        Original runner for restoration.
    """
    original = jit_mod.run_linear

    def wrapped(linear: UOp, *args: Any, **kwargs: Any) -> Any:
        """Record captured JIT execution and delegate unchanged."""
        events.append((linear.op.name, len(linear.src), bool(kwargs.get("jit", False))))
        return original(linear, *args, **kwargs)

    jit_mod.run_linear = wrapped
    return original


def build_jitted_add_one() -> TinyJit[Tensor]:
    """Build a tiny JIT function with a realized tensor output.

    Returns
    -------
    TinyJit[Tensor]
        JIT wrapper around a one-kernel add operation.
    """

    @TinyJit
    def add_one(x: Tensor) -> Tensor:
        """Add one and realize so TinyJit captures a LINEAR schedule."""
        return (x + 1.0).realize()

    return add_one


def realized_input(data: list[float]) -> Tensor:
    """Create a realized input tensor suitable for TinyJit.

    Parameters
    ----------
    data
        Input values.

    Returns
    -------
    Tensor
        Realized tensor backed by a buffer.
    """
    return Tensor(data).realize()


def assert_captured_jit_hook_and_payload_reads() -> None:
    """Assert captured TinyJit execution is visible and payload reads are isolated."""
    tensor_events: list[tuple[str, int, bool]] = []
    jit_events: list[tuple[str, int, bool]] = []
    original_tensor_run = patch_tensor_run_linear(tensor_events)
    original_jit_run = patch_jit_run_linear(jit_events)
    add_one = build_jitted_add_one()
    try:
        first = add_one(realized_input([1.0, 2.0]))
        assert add_one.cnt == 1
        assert add_one.captured is None
        assert values(first) == [2.0, 3.0]

        second = add_one(realized_input([1.0, 2.0]))
        assert add_one.cnt == 2
        assert add_one.captured is not None
        assert values(second) == [2.0, 3.0]

        captured_linear = add_one.captured.linear
        captured_before = uop_ops(captured_linear)
        ret_before = tensor_ops(second)

        third = add_one(realized_input([3.0, 4.0]))
        assert third is second
        assert add_one.captured.ret is third
        assert values(third) == [4.0, 5.0]
        assert uop_ops(captured_linear) == captured_before
        assert tensor_ops(third) == ret_before

        assert values(third) == [4.0, 5.0]
        assert uop_ops(captured_linear) == captured_before
        assert tensor_ops(third) == ret_before
    finally:
        tensor_mod.run_linear = original_tensor_run
        jit_mod.run_linear = original_jit_run

    assert tensor_events, tensor_events
    assert all(event[2] is False for event in tensor_events), tensor_events
    assert jit_events == [("LINEAR", 1, True), ("LINEAR", 1, True)], jit_events


def assert_captured_jit_free_intermediates_keeps_execution_valid() -> None:
    """Assert freeing captured intermediates does not invalidate future executions."""
    add_one = build_jitted_add_one()
    assert values(add_one(realized_input([1.0, 2.0]))) == [2.0, 3.0]
    assert values(add_one(realized_input([1.0, 2.0]))) == [2.0, 3.0]
    assert add_one.captured is not None
    before = uop_ops(add_one.captured.linear)
    add_one.captured.free_intermediates()
    assert values(add_one(realized_input([7.0, 8.0]))) == [8.0, 9.0]
    assert uop_ops(add_one.captured.linear) == before


def main() -> None:
    """Run all round-3 TinyJit payload probes."""
    assert_captured_jit_hook_and_payload_reads()
    assert_captured_jit_free_intermediates_keeps_execution_valid()
    print("PASS probe_tinyjit_payload_round3")


if __name__ == "__main__":
    main()
