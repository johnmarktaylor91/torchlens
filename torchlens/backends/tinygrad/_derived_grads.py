"""Derived-gradient helpers for the tinygrad backend.

Bracketed backward on tinygrad leaves: normalize the differentiated input
argnums, walk the positional inputs to their tensor leaves, snapshot any
pre-existing ``.grad`` state, build the leaf and exact intermediate
derived-gradient records after the backward, and restore the leaves. The
backend module keeps the capture; these helpers only read it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from ..._data_classes.derived_grad import DerivedGradRecord, IntermediateDerivedGradRecord
from ..._ir.refs import DtypeRef

if TYPE_CHECKING:
    from .backend import TinygradBackend, TinygradIntermediateCandidate

__tl_layer__ = "L2"


def _tree_leaves_with_paths(value: object, prefix: str = "") -> list[tuple[str, Any]]:
    """Return leaves from a simple Python container with dotted paths.

    Parameters
    ----------
    value
        Container or leaf.
    prefix
        Current dotted path prefix.

    Returns
    -------
    list[tuple[str, Any]]
        Flat ``(path, leaf)`` pairs.
    """

    if isinstance(value, dict):
        return [
            item
            for key, child in value.items()
            for item in _tree_leaves_with_paths(child, f"{prefix}.{key}" if prefix else str(key))
        ]
    if isinstance(value, tuple | list):
        return [
            item
            for index, child in enumerate(value)
            for item in _tree_leaves_with_paths(
                child, f"{prefix}.{index}" if prefix else str(index)
            )
        ]
    return [(prefix, value)]


def call_input_leaves(
    args: Sequence[Any], kwargs: Mapping[str, Any] | None
) -> list[tuple[str, Any]]:
    """Leaves of a call's positional, then keyword, arguments with dotted paths.

    A positional leaf's path starts with its argument index, a keyword leaf's
    with its keyword name.
    """

    return _tree_leaves_with_paths(tuple(args)) + _tree_leaves_with_paths(dict(kwargs or {}))


def _normalize_tinygrad_input_grad_argnums(
    value: Sequence[int] | None, num_inputs: int
) -> tuple[int, ...]:
    """Normalize tinygrad input gradient argnums.

    Parameters
    ----------
    value
        User-supplied positional input argnums, or ``None`` for all inputs.
    num_inputs
        Number of positional inputs passed to the tinygrad callable.

    Returns
    -------
    tuple[int, ...]
        Unique non-negative positional input indexes.
    """

    if value is None:
        return tuple(range(num_inputs))
    normalized = tuple(sorted({index if index >= 0 else num_inputs + index for index in value}))
    invalid = [index for index in normalized if index < 0 or index >= num_inputs]
    if invalid:
        raise ValueError(
            f"input_grad_argnums indexes out of range for {num_inputs} inputs: {invalid}."
        )
    return normalized


def _differentiated_input_leaves(
    backend: TinygradBackend,
    args: Sequence[Any],
    input_argnums: Sequence[int],
) -> tuple[tuple[str, int, Any], ...]:
    """Return differentiated tinygrad input tensor leaves.

    Parameters
    ----------
    backend
        tinygrad backend instance used for tensor checks.
    args
        Positional callable arguments.
    input_argnums
        Positional input indexes to differentiate.

    Returns
    -------
    tuple[tuple[str, int, Any], ...]
        ``(path, argnum, tensor)`` records for tensor leaves.
    """

    leaves: list[tuple[str, int, Any]] = []
    for argnum in input_argnums:
        for local_path, leaf in _tree_leaves_with_paths(args[argnum]):
            if not backend.is_tensor(leaf):
                continue
            suffix = f".{local_path}" if local_path else ""
            leaves.append((f"inputs.{argnum}{suffix}", argnum, leaf))
    return tuple(leaves)


def _snapshot_tinygrad_grads(
    backend: TinygradBackend,
    leaves: Sequence[tuple[str, int, Any]],
) -> dict[str, tuple[Any, Any | None, Any | None]]:
    """Snapshot pre-existing tinygrad leaf grads before backward.

    Parameters
    ----------
    backend
        tinygrad backend instance used for realized copies.
    leaves
        Differentiated input leaves.

    Returns
    -------
    dict[str, tuple[Any, Any | None, Any | None]]
        Mapping from derived-grad path to ``(leaf, original_grad_object, value_snapshot)``.
    """

    snapshots: dict[str, tuple[Any, Any | None, Any | None]] = {}
    for path, _argnum, tensor in leaves:
        original_grad = getattr(tensor, "grad", None)
        grad_snapshot = None if original_grad is None else backend._realized_copy(original_grad)
        snapshots[path] = (tensor, original_grad, grad_snapshot)
    return snapshots


def _records_for_tinygrad_leaf_grads(
    *,
    backend: TinygradBackend,
    leaves: Sequence[tuple[str, int, Any]],
    snapshots: Mapping[str, tuple[Any, Any | None, Any | None]],
    provenance: Mapping[str, Any],
) -> dict[str, DerivedGradRecord]:
    """Build derived-gradient records from bracketed tinygrad leaf grads.

    Parameters
    ----------
    backend
        tinygrad backend instance used for realized copies.
    leaves
        Differentiated input leaves.
    snapshots
        Pre-backward gradient snapshots.
    provenance
        Shared provenance metadata.

    Returns
    -------
    dict[str, DerivedGradRecord]
        Records keyed by stable input leaf path.
    """

    records: dict[str, DerivedGradRecord] = {}
    for path, argnum, tensor in leaves:
        post_grad = getattr(tensor, "grad", None)
        if post_grad is None:
            raise ValueError(f"tinygrad backward did not populate .grad for leaf {path!r}.")
        _tensor, _original_grad, grad_snapshot = snapshots[path]
        increment = post_grad if grad_snapshot is None else post_grad - grad_snapshot
        grad = backend._realized_copy(increment)
        records[path] = DerivedGradRecord(
            path=path,
            source="inputs",
            argnum=argnum,
            input_argnum=argnum,
            aval=f"Tensor(shape={tuple(getattr(grad, 'shape', ()))}, dtype={getattr(grad, 'dtype', None)})",
            dtype_ref=DtypeRef(backend="tinygrad", name=str(getattr(grad, "dtype", ""))),
            grad=grad,
            provenance=provenance,
        )
    return records


def _records_for_tinygrad_intermediate_grads(
    *,
    backend: TinygradBackend,
    trace_signatures: Mapping[tuple[Any, ...], list[Any]],
    live_candidates: Mapping[tuple[Any, ...], list[TinygradIntermediateCandidate]],
    snapshots: Mapping[str, tuple[Any, Any | None, Any | None]],
    provenance: Mapping[str, Any],
) -> dict[str, IntermediateDerivedGradRecord]:
    """Build exact tinygrad intermediate-derived gradient records.

    Parameters
    ----------
    backend
        tinygrad backend instance used for realized grad copies.
    trace_signatures
        Saved trace ops grouped by conservative signature.
    live_candidates
        Live no-realize tensors grouped by conservative signature.
    snapshots
        User input grad snapshots used only to avoid attaching records for the
        differentiated input leaves themselves.
    provenance
        Shared provenance metadata for exact attached records.

    Returns
    -------
    dict[str, IntermediateDerivedGradRecord]
        Records keyed by pass-qualified op label. Ambiguous or missing matches
        are skipped.
    """

    records: dict[str, IntermediateDerivedGradRecord] = {}
    skipped_input_ids = {id(tensor) for tensor, _original, _snapshot in snapshots.values()}
    for signature, ops in trace_signatures.items():
        candidates = live_candidates.get(signature, ())
        if len(ops) != 1 or len(candidates) != 1:
            continue
        op = ops[0]
        candidate = candidates[0]
        if id(candidate.tensor) in skipped_input_ids:
            continue
        grad = getattr(candidate.tensor, "grad", None)
        if grad is None:
            continue
        copied_grad = backend._realized_copy(grad)
        records[op.label] = IntermediateDerivedGradRecord(
            op_label=op.label,
            layer_label=op.layer_label,
            aval=candidate.aval,
            dtype_ref=candidate.dtype_ref,
            grad=copied_grad,
            provenance=dict(provenance),
        )
    return records


def _restore_tinygrad_grads(snapshots: Mapping[str, tuple[Any, Any | None, Any | None]]) -> None:
    """Restore pre-existing tinygrad leaf grad state after bracketed backward.

    Parameters
    ----------
    snapshots
        Mapping from derived-grad path to ``(leaf, original_grad_object, value_snapshot)``.

    Returns
    -------
    None
        Leaf ``.grad`` values are restored in place where possible.
    """

    for path, (tensor, original_grad, grad_snapshot) in snapshots.items():
        del path
        if original_grad is None:
            tensor.grad = None
            continue
        original_grad.assign(grad_snapshot)
        tensor.grad = original_grad
