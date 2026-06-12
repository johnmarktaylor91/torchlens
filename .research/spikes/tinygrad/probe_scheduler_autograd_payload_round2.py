"""Probe tinygrad scheduler, JIT, autograd, aliasing, and payload edge cases.

Run with:
  DEV=PYTHON nice -n 19 ionice -c3 /tmp/jaxtg/tinygrad-venv/bin/python \
    .research/spikes/tinygrad/probe_scheduler_autograd_payload_round2.py
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from tinygrad import Tensor
from tinygrad.engine.jit import TinyJit
from tinygrad.uop.ops import UOp

import tinygrad.tensor as tensor_mod


def values(tensor: Tensor) -> list[float]:
    """Read tensor values as a Python list.

    Parameters
    ----------
    tensor
        Tensor to read through tinygrad's sanctioned host payload API.

    Returns
    -------
    list[float]
        Realized scalar values.
    """
    return list(tensor.tolist())


def graph_ops(tensor: Tensor) -> tuple[str, ...]:
    """Return topologically sorted UOp names for a tensor.

    Parameters
    ----------
    tensor
        Tensor whose current lineage should be summarized.

    Returns
    -------
    tuple[str, ...]
        Operation names in topological order.
    """
    return tuple(uop.op.name for uop in tensor.uop.toposort())


def build_lazy_expression() -> Tensor:
    """Build a tiny lazy expression that has semantic UOp lineage.

    Returns
    -------
    Tensor
        Two-element expression with ADD, CMPLT, and WHERE lineage.
    """
    return (Tensor([1.0, 2.0]) + 1.0).relu()


def patch_linear_with_vars(events: list[tuple[str, tuple[str, ...]]]) -> Callable[..., Any]:
    """Wrap ``Tensor.linear_with_vars`` to observe explicit realization planning.

    Parameters
    ----------
    events
        Mutable list receiving the LINEAR root and child operation names.

    Returns
    -------
    Callable[..., Any]
        Original method for restoration.
    """
    original = Tensor.linear_with_vars

    def wrapped(self: Tensor, *lst: Tensor) -> tuple[UOp, dict[str, int]]:
        """Record the scheduler-planning boundary and delegate unchanged."""
        linear, var_vals = original(self, *lst)
        events.append((linear.op.name, tuple(child.op.name for child in linear.src)))
        return linear, var_vals

    Tensor.linear_with_vars = wrapped  # type: ignore[method-assign]
    return original


def patch_run_linear(events: list[tuple[bool, int]]) -> Callable[..., Any]:
    """Wrap ``run_linear`` to observe realized scheduler execution.

    Parameters
    ----------
    events
        Mutable list receiving the ``jit`` flag and direct LINEAR child count.

    Returns
    -------
    Callable[..., Any]
        Original runner for restoration.
    """
    original = tensor_mod.run_linear

    def wrapped(linear: UOp, *args: Any, **kwargs: Any) -> Any:
        """Record the realization boundary and delegate unchanged."""
        events.append((bool(kwargs.get("jit", False)), len(linear.src)))
        return original(linear, *args, **kwargs)

    tensor_mod.run_linear = wrapped
    return original


def assert_explicit_realize_rewrites_lineage() -> None:
    """Assert explicit ``realize`` collapses semantic lineage to buffer identity."""
    expr = build_lazy_expression()
    before = graph_ops(expr)
    assert before[-1] == "WHERE", before
    assert "ADD" in before, before

    linear_events: list[tuple[str, tuple[str, ...]]] = []
    run_events: list[tuple[bool, int]] = []
    original_linear = patch_linear_with_vars(linear_events)
    original_run = patch_run_linear(run_events)
    try:
        assert expr.realize() is expr
    finally:
        Tensor.linear_with_vars = original_linear  # type: ignore[method-assign]
        tensor_mod.run_linear = original_run

    after = graph_ops(expr)
    assert after[-1] == "BUFFER", after
    assert "WHERE" not in after, after
    assert linear_events and linear_events[0][0] == "LINEAR", linear_events
    assert run_events == [(False, 1)], run_events
    assert values(expr) == [2.0, 3.0]


def assert_payload_read_does_not_rewrite_lineage() -> None:
    """Assert ``tolist`` obtains payloads without rebasing the source UOp graph."""
    expr = build_lazy_expression()
    before = graph_ops(expr)
    assert values(expr) == [2.0, 3.0]
    assert graph_ops(expr) == before


def assert_tinyjit_run_linear_shape() -> None:
    """Assert TinyJit captures after the warmup call and bypasses normal run hooks."""
    run_events: list[tuple[bool, int]] = []
    original_run = patch_run_linear(run_events)

    @TinyJit
    def add_one(x: Tensor) -> Tensor:
        """Add one and realize so TinyJit has a schedule to capture."""
        return (x + 1.0).realize()

    try:
        assert values(add_one(Tensor([1.0, 2.0]))) == [2.0, 3.0]
        assert add_one.captured is None
        assert values(add_one(Tensor([1.0, 2.0]))) == [2.0, 3.0]
        assert add_one.captured is not None
        assert values(add_one(Tensor([1.0, 2.0]))) == [2.0, 3.0]
    finally:
        tensor_mod.run_linear = original_run

    assert run_events == [(False, 1), (False, 0)], run_events


def assert_view_alias_signature_can_stay_stale() -> None:
    """Assert alias values can change while a held view's UOp signature is unchanged."""
    base = Tensor([1.0, 2.0, 3.0])
    view = base[1:]
    before_view_ops = graph_ops(view)
    assert values(view) == [2.0, 3.0]
    assert view.assign(Tensor([9.0, 8.0])) is view
    assert values(base) == [1.0, 9.0, 8.0]
    assert values(view) == [9.0, 8.0]
    assert graph_ops(view) == before_view_ops


def assert_explicit_gradient_lifecycle() -> None:
    """Assert explicit gradients work for scalar backward and non-scalar gradient."""
    x = Tensor([1.0, 2.0])
    y = (x * 3.0).sum()
    assert y.backward(gradient=Tensor(2.0)) is y
    assert x.grad is not None
    assert values(x.grad) == [6.0, 6.0]

    target = x * 3.0
    grads = target.gradient(x, gradient=Tensor([1.0, 10.0]))
    assert len(grads) == 1
    assert values(grads[0]) == [3.0, 30.0]
    assert values(x.grad) == [6.0, 6.0]


def assert_grad_payload_snapshot_isolated_after_realize() -> None:
    """Assert realized gradient payload copies stay stable after primal mutation."""
    x = Tensor([1.0, 2.0])
    grad = (x * x).sum().gradient(x)[0]
    snapshot = grad.clone().realize()
    assert values(snapshot) == [2.0, 4.0]
    assert x.assign(Tensor([10.0, 20.0])) is x
    assert values(snapshot) == [2.0, 4.0]


def assert_clone_payload_isolated_after_source_mutation() -> None:
    """Assert a realized clone keeps payload values after source tensor mutation."""
    source = build_lazy_expression()
    snapshot = source.clone().realize()
    assert values(snapshot) == [2.0, 3.0]
    assert source.assign(Tensor([7.0, 8.0])) is source
    assert values(source) == [7.0, 8.0]
    assert values(snapshot) == [2.0, 3.0]


def main() -> None:
    """Run all round-2 tinygrad probes."""
    assert_explicit_realize_rewrites_lineage()
    assert_payload_read_does_not_rewrite_lineage()
    assert_tinyjit_run_linear_shape()
    assert_view_alias_signature_can_stay_stale()
    assert_explicit_gradient_lifecycle()
    assert_grad_payload_snapshot_isolated_after_realize()
    assert_clone_payload_isolated_after_source_mutation()
    print("PASS probe_scheduler_autograd_payload_round2")


if __name__ == "__main__":
    main()
