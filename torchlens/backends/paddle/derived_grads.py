"""Derived-gradient leaf helpers for the Paddle backend.

Leaf discovery over inputs and parameters, the tensor-state snapshots the
replay restores, the record builder and the float checks they share.
``PaddleBackend`` imports them; this module only names the backend for typing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..._data_classes.derived_grad import DerivedGradRecord
from ..._ir.refs import DtypeRef

if TYPE_CHECKING:
    from ..._data_classes.trace import Trace
    from .backend import PaddleBackend


@dataclass(frozen=True)
class PaddleGradLeaf:
    """One Paddle tensor leaf differentiated by the derived-gradient replay.

    Parameters
    ----------
    path
        Public derived-gradient record path.
    source
        Source group, either ``"inputs"`` or ``"params"``.
    argnum
        Positional model input argnum for inputs, or ``-1`` for params.
    input_argnum
        Input-relative argument index for input gradients.
    tensor
        Live Paddle tensor to differentiate.
    """

    path: str
    source: str
    argnum: int
    input_argnum: int | None
    tensor: Any


@dataclass(frozen=True)
class PaddleTensorState:
    """Snapshot of mutable Paddle tensor gradient state.

    Parameters
    ----------
    tensor
        Tensor whose state was snapshotted.
    stop_gradient
        Original ``stop_gradient`` value.
    grad
        Original ``.grad`` payload.
    """

    tensor: Any
    stop_gradient: bool
    grad: Any


def _normalize_paddle_input_grad_argnums(
    *,
    backend: PaddleBackend,
    args: Sequence[Any],
    input_grad_argnums: Sequence[int] | None,
) -> tuple[int, ...]:
    """Normalize Paddle derived-gradient input argnums.

    Parameters
    ----------
    backend
        Paddle backend used for tensor checks.
    args
        Positional model inputs.
    input_grad_argnums
        User-requested argnums, or ``None`` for all float tensor inputs.

    Returns
    -------
    tuple[int, ...]
        Validated positional input argnums.
    """

    if input_grad_argnums is None:
        return tuple(
            index
            for index, value in enumerate(args)
            if any(
                _is_float_paddle_tensor(tensor)
                for _path, tensor in backend._iter_tensors_with_paths(value)
            )
        )
    normalized = tuple(int(index) for index in input_grad_argnums)
    for index in normalized:
        if index < 0 or index >= len(args):
            raise ValueError(f"Paddle input_grad_argnums contains out-of-range argnum {index}.")
    return normalized


def _paddle_input_grad_leaves(
    backend: PaddleBackend,
    args: Sequence[Any],
    input_argnums: Sequence[int],
) -> tuple[PaddleGradLeaf, ...]:
    """Return differentiable Paddle input leaves.

    Parameters
    ----------
    backend
        Paddle backend used for tensor traversal.
    args
        Positional model inputs.
    input_argnums
        Positional argnums to differentiate.

    Returns
    -------
    tuple[PaddleGradLeaf, ...]
        Floating tensor input leaves.
    """

    leaves: list[PaddleGradLeaf] = []
    for argnum in input_argnums:
        for path, tensor in backend._iter_tensors_with_paths(args[argnum]):
            if not _is_float_paddle_tensor(tensor):
                continue
            suffix = ".".join(str(part) for part in path)
            record_path = f"inputs.{argnum}" if suffix == "" else f"inputs.{argnum}.{suffix}"
            leaves.append(
                PaddleGradLeaf(
                    path=record_path,
                    source="inputs",
                    argnum=argnum,
                    input_argnum=argnum,
                    tensor=tensor,
                )
            )
    return tuple(leaves)


def _paddle_param_grad_leaves(trace: Trace) -> tuple[PaddleGradLeaf, ...]:
    """Return differentiable Paddle parameter leaves from ``trace.params``.

    Parameters
    ----------
    trace
        Finalized Paddle trace.

    Returns
    -------
    tuple[PaddleGradLeaf, ...]
        Parameter leaves with stable ``params.<address>`` paths.
    """

    leaves: list[PaddleGradLeaf] = []
    for address, param in trace.params.items():
        tensor = getattr(param, "_param_ref", None)
        if tensor is None or not _is_float_paddle_tensor(tensor):
            continue
        leaves.append(
            PaddleGradLeaf(
                path=f"params.{address}",
                source="params",
                argnum=-1,
                input_argnum=None,
                tensor=tensor,
            )
        )
    return tuple(leaves)


def _snapshot_paddle_tensor_states(tensors: Sequence[Any]) -> tuple[PaddleTensorState, ...]:
    """Snapshot Paddle tensor ``stop_gradient`` and ``.grad`` state.

    Parameters
    ----------
    tensors
        Tensors whose mutable gradient state will be restored.

    Returns
    -------
    tuple[PaddleTensorState, ...]
        Unique tensor snapshots keyed by object identity.
    """

    snapshots: list[PaddleTensorState] = []
    seen: set[int] = set()
    for tensor in tensors:
        if id(tensor) in seen:
            continue
        seen.add(id(tensor))
        snapshots.append(
            PaddleTensorState(
                tensor=tensor,
                stop_gradient=bool(getattr(tensor, "stop_gradient", True)),
                grad=getattr(tensor, "grad", None),
            )
        )
    return tuple(snapshots)


def _restore_paddle_tensor_states(snapshots: Sequence[PaddleTensorState]) -> None:
    """Restore Paddle tensor gradient state from snapshots.

    Parameters
    ----------
    snapshots
        Snapshots returned by ``_snapshot_paddle_tensor_states``.

    Returns
    -------
    None
        Tensors are restored in place.
    """

    for snapshot in snapshots:
        snapshot.tensor.stop_gradient = snapshot.stop_gradient
        snapshot.tensor.grad = snapshot.grad


def _records_for_paddle_leaf_grads(
    *,
    leaves: Sequence[PaddleGradLeaf],
    grads: Sequence[Any | None],
    provenance: Mapping[str, Any],
) -> dict[str, DerivedGradRecord]:
    """Build derived-gradient records for Paddle leaves.

    Parameters
    ----------
    leaves
        Differentiated leaves.
    grads
        Gradients returned by ``paddle.grad``.
    provenance
        Shared provenance metadata.

    Returns
    -------
    dict[str, DerivedGradRecord]
        Records keyed by stable leaf path. ``None`` gradients are skipped.
    """

    records: dict[str, DerivedGradRecord] = {}
    for leaf, grad in zip(leaves, grads, strict=True):
        if grad is None:
            continue
        records[leaf.path] = DerivedGradRecord(
            path=leaf.path,
            source=leaf.source,
            argnum=leaf.argnum,
            input_argnum=leaf.input_argnum,
            aval=f"Tensor(shape={tuple(getattr(grad, 'shape', ()))}, dtype={getattr(grad, 'dtype', None)})",
            dtype_ref=DtypeRef(backend="paddle", name=str(getattr(grad, "dtype", ""))),
            grad=grad,
            provenance=provenance,
        )
    return records


def _is_float_paddle_tensor(value: Any) -> bool:
    """Return whether ``value`` is a floating Paddle tensor.

    Parameters
    ----------
    value
        Candidate value.

    Returns
    -------
    bool
        True when ``value`` looks like a floating Paddle tensor.
    """

    return (
        hasattr(value, "dtype")
        and hasattr(value, "shape")
        and _is_float_dtype_text(str(getattr(value, "dtype", "")))
    )


def _is_float_dtype_text(dtype: str) -> bool:
    """Return whether a dtype string names a floating dtype.

    Parameters
    ----------
    dtype
        Backend dtype string.

    Returns
    -------
    bool
        True for float and bfloat dtypes.
    """

    lowered = dtype.lower()
    return "float" in lowered or "bfloat" in lowered
