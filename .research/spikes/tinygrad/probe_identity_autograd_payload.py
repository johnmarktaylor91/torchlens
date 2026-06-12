"""Probe tinygrad identity, mutation, autograd lifecycle, JIT, GC, and payload reads.

Run with:
  DEV=PYTHON nice -n 19 ionice -c3 /tmp/jaxtg/tinygrad-venv/bin/python \
    .research/spikes/tinygrad/probe_identity_autograd_payload.py
"""

from __future__ import annotations

import gc
import weakref

from tinygrad import Tensor
from tinygrad.engine.jit import TinyJit
from tinygrad.tensor import all_tensors


def values(tensor: Tensor) -> list[float]:
    """Read tensor values as a Python list.

    Parameters
    ----------
    tensor
        Tensor to realize into a tiny host payload.

    Returns
    -------
    list[float]
        Realized scalar values.
    """
    return list(tensor.tolist())


def op_name(tensor: Tensor) -> str:
    """Return the current root UOp name for a tensor.

    Parameters
    ----------
    tensor
        Tensor to inspect.

    Returns
    -------
    str
        Root UOp operation name.
    """
    return tensor.uop.op.name


def graph_signature(tensor: Tensor) -> tuple[str, ...]:
    """Return topologically sorted operation names for a tensor graph.

    Parameters
    ----------
    tensor
        Tensor to inspect.

    Returns
    -------
    tuple[str, ...]
        Toposorted operation names.
    """
    return tuple(uop.op.name for uop in tensor.uop.toposort())


def assert_assign_replaces_uop() -> None:
    """Assert assign mutates the Tensor object's UOp identity."""
    target = Tensor([1.0, 2.0])
    original = target.uop
    assert target.assign(Tensor([3.0, 4.0])) is target
    assert target.uop is not original
    assert op_name(target) == "AFTER"
    assert values(target) == [3.0, 4.0]


def assert_replace_replaces_uop() -> None:
    """Assert replace swaps Tensor UOp identity directly."""
    target = Tensor([1.0, 2.0])
    original = target.uop
    assert target.replace(Tensor([5.0, 6.0])) is target
    assert target.uop is not original
    assert op_name(target) == "BUFFER"
    assert values(target) == [5.0, 6.0]


def assert_view_assign_rewrites_base() -> None:
    """Assert view assignment rewrites base tensor lineage and preserves object identity."""
    base = Tensor([1.0, 2.0, 3.0])
    view = base[1:]
    original_base = base.uop
    original_view = view.uop
    assert view.assign(Tensor([9.0, 8.0])) is view
    assert base.uop is not original_base
    assert view.uop is not original_view
    assert op_name(base) == "AFTER"
    assert values(base) == [1.0, 9.0, 8.0]


def assert_setitem_rewrites_uop() -> None:
    """Assert setitem mutates Tensor lineage through assign/replace paths."""
    target = Tensor([1.0, 2.0, 3.0])
    original = target.uop
    target[1] = 7.0
    assert target.uop is not original
    assert op_name(target) == "AFTER"
    assert values(target) == [1.0, 7.0, 3.0]


def assert_repeated_backward_accumulates() -> None:
    """Assert repeated backward accumulates into existing gradient state."""
    x = Tensor([2.0, 3.0])
    y = (x * x).sum()
    y.backward()
    first_grad = x.grad
    assert first_grad is not None
    assert values(first_grad) == [4.0, 6.0]
    y.backward()
    assert x.grad is first_grad
    assert values(x.grad) == [8.0, 12.0]


def assert_tinyjit_lifecycle() -> None:
    """Assert TinyJit ignores, captures, then executes a toy realized function."""

    @TinyJit
    def add_one(x: Tensor) -> Tensor:
        """Add one and realize so TinyJit has a schedule to capture."""
        return (x + 1.0).realize()

    x = Tensor([1.0, 2.0])
    assert values(add_one(x)) == [2.0, 3.0]
    assert add_one.cnt == 1
    assert add_one.captured is None
    assert values(add_one(x)) == [2.0, 3.0]
    assert add_one.cnt == 2
    assert add_one.captured is not None
    assert values(add_one(x)) == [2.0, 3.0]
    assert add_one.cnt == 3
    assert add_one.captured is not None


def assert_gc_removes_tensor_registry_entry() -> None:
    """Assert tinygrad's weak tensor registry drops dead tensors."""
    tensor = Tensor([1.0])
    ref = weakref.ref(tensor)
    before = len(all_tensors)
    del tensor
    gc.collect()
    assert ref() is None
    assert len(all_tensors) < before


def assert_autograd_lifecycle() -> None:
    """Assert gradient and backward lifecycle on scalar and non-scalar roots."""
    x = Tensor([1.0, 2.0, 3.0])
    y = (x * 2.0).sum()
    assert x.grad is None
    grads = y.gradient(x)
    assert len(grads) == 1
    assert values(grads[0]) == [2.0, 2.0, 2.0]
    assert x.grad is None
    assert y.backward() is y
    assert x.grad is not None
    assert values(x.grad) == [2.0, 2.0, 2.0]
    try:
        (x * 2.0).backward()
    except AssertionError as exc:
        assert "scalar tensor" in str(exc)
    else:
        raise AssertionError("non-scalar backward without gradient unexpectedly passed")


def assert_payload_copy_non_interference() -> None:
    """Assert host payload reads and clones do not mutate the source UOp graph."""
    source = (Tensor([1.0, 2.0]) + 1.0).relu()
    before = graph_signature(source)
    assert values(source) == [2.0, 3.0]
    assert graph_signature(source) == before
    clone = source.clone()
    assert values(clone) == [2.0, 3.0]
    assert graph_signature(source) == before


def main() -> None:
    """Run all identity, autograd, and payload probes."""
    assert_assign_replaces_uop()
    assert_replace_replaces_uop()
    assert_view_assign_rewrites_base()
    assert_setitem_rewrites_uop()
    assert_repeated_backward_accumulates()
    assert_tinyjit_lifecycle()
    assert_gc_removes_tensor_registry_entry()
    assert_autograd_lifecycle()
    assert_payload_copy_non_interference()
    print("PASS probe_identity_autograd_payload")


if __name__ == "__main__":
    main()
