"""UOp-graph walks the tinygrad capture records ops from.

Extracted from ``backend`` so that module stays under its size ceiling.
:func:`_data_flow_uops` picks the UOps a capture records: those whose value
reaches an output as data.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import Any, cast


def _unique_uops(outputs: Sequence[Any]) -> tuple[Any, ...]:
    """Return UOps reachable from outputs in topological order.

    Parameters
    ----------
    outputs
        tinygrad output tensors.

    Returns
    -------
    tuple[Any, ...]
        Unique UOps in first-seen topological order.
    """

    seen: set[int] = set()
    ordered: list[Any] = []
    for output in outputs:
        for uop in cast(Any, output).uop.toposort():
            if id(uop) in seen:
                continue
            seen.add(id(uop))
            ordered.append(uop)
    return tuple(ordered)


def _is_materializable_uop(uop: Any) -> bool:
    """Return whether a UOp can be saved as a tinygrad Tensor payload.

    Parameters
    ----------
    uop
        Candidate tinygrad UOp.

    Returns
    -------
    bool
        True when tinygrad can expose shape and host payload for ``uop``.
    """

    try:
        from tinygrad import Tensor

        tensor = Tensor(uop)
        tuple(tensor.shape)
        if getattr(tensor.dtype.base, "fmt", None) is None:
            return False
        tensor.tolist()
    except Exception:
        return False
    return True


def _uop_name(uop: Any) -> str:
    """Return a stable tinygrad UOp name.

    Parameters
    ----------
    uop
        tinygrad UOp.

    Returns
    -------
    str
        Operation name without the ``Ops.`` prefix.
    """

    op = getattr(uop, "op", None)
    return str(getattr(op, "name", op)).removeprefix("Ops.")


def _data_flow_uops(
    uops: Sequence[Any], outputs: Sequence[Any], source_ids: Collection[int]
) -> tuple[Any, ...]:
    """Return the UOps whose values reach an output as data, in ``uops`` order.

    The walk starts at the outputs and follows ``src`` edges through
    materializable UOps only, stopping at the capture's sources (``source_ids``,
    the input UOps). A non-materializable UOp ends the walk: it carries no
    tensor value, so what feeds it is not data of the forward. That leaves out
    the scalar ``CONST`` elements of a movement op's shape vector and the
    storage ``BUFFER`` behind an input, which a plain topological sort reaches
    but which no recorded op consumes. The same rule decides which ops torch
    keeps: a value no output depends on and no input produces is an orphan.

    Parameters
    ----------
    uops
        UOps reachable from the outputs, in topological order (:func:`_unique_uops`).
    outputs
        tinygrad output tensors.
    source_ids
        ``id()`` of every source UOp (the capture's inputs).

    Returns
    -------
    tuple[Any, ...]
        The data-flow UOps, sources excluded, each materializable.
    """

    materializable: dict[int, bool] = {}
    kept: set[int] = set()
    stack = [cast(Any, output).uop for output in outputs]
    while stack:
        uop = stack.pop()
        key = id(uop)
        if key in kept or key in source_ids:
            continue
        if key not in materializable:
            materializable[key] = _is_materializable_uop(uop)
        if not materializable[key]:
            continue
        kept.add(key)
        stack.extend(getattr(uop, "src", ()) or ())
    return tuple(uop for uop in uops if id(uop) in kept)
