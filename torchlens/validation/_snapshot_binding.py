"""Prove that a parent's per-child version snapshot is explained by the recorded graph.

Capture stores ``parent.out_versions_by_child[child]`` whenever the value in the
argument slot a parent is bound to differs from that parent's saved out
(``_get_parent_output_version_snapshot``). Argument logging and forward replay then
take that snapshot as the parent's value for the child. The snapshot is the slot's
content, so it agrees with whatever binding capture made: a call whose arguments
were bound to the wrong parents (two operands swapped, a base and its view
swapped, a ``cat`` list reordered, a write's consumer bound past the write) records
snapshots that carry the swapped content and replays exactly. This module is the
tripwire for that class: a snapshot that differs from its parent's saved out must
be explained by a recorded in-place write.

A differing snapshot is explained when

- the parent is itself an in-place op: a reader of the base or of another alias of
  the view it wrote through is bound to the write, and its value is that alias, not
  the written view; the snapshot has the write's dtype and the shape of an op on the
  write's receiver chain, or, for ``set_(storage, ...)``, exactly the bytes of the
  storage it bound the receiver to (content is not compared); or capture marked every
  slot bound to the write as holding an alias the write relabeled onto it (a sibling view
  taken off the receiver chain, ``backends/torch/_storage_exports.py``), and the snapshot
  has the write's dtype; or
- the parent is not an in-place op, the snapshot has the parent's shape and dtype
  (an in-place write never changes either), and an in-place op the child ALSO
  consumes wrote through the parent's lineage: the call reads a base together
  with the view written into it, the base bound to its own lineage before the write; or
- the parent is not an in-place op, the snapshot has the parent's shape and dtype, and
  capture marked this snapshot as a foreign write: the slot's live tensor sat on a
  storage the forward had handed to the host zero-copy (``data_ptr``, a storage handle,
  a NumPy view, DLPack), so code outside torch dispatch may have written it. The mark
  names the exact argument slots the parent was bound to
  (``backends/torch/_storage_exports.py``), and capture sets it only when each slot
  held the parent's own tensor on that exported storage; a parent bound to any other
  slots, or a snapshot without the mark, is not explained by it. A loaded trace carries
  no marks, and marks are not invalidated by later edits of a parent's snapshots; or
- the parent is not an in-place op, the snapshot has the parent's shape and dtype, and
  the parent's tensor went into a mutating operator call capture could not record and
  disclosed (``backends/torch/_torch_ops_calls.py``): that disclosure is a
  ``source_provenance`` gap, so the run fails there with the cause named.

Anything else is a binding the recorded graph cannot produce, and argument logging
fails. The check only adds failures; it never validates what failed before.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import torch

from .._utils._record_group_table import record_member
from .._utils.tensor_utils import tensor_nanequal
from ._replay_storage_context import receiver_lineage

if TYPE_CHECKING:
    from .._data_classes.op import Op


def child_version_snapshot(parent_layer: Op, target_layer: Op) -> Any:
    """Return the snapshot ``parent_layer`` recorded for ``target_layer``, else ``None``.

    The pass-qualified child label is tried first, then the bare layer label, as
    replay and argument logging read it.
    """

    versions = getattr(parent_layer, "out_versions_by_child", None) or {}
    for key in (getattr(target_layer, "label", None), target_layer.layer_label):
        if key is not None and key in versions:
            return versions[key]
    return None


def snapshot_binding_explained(
    target_layer: Op,
    parent_layer: Op,
    saved_out: Callable[[Op], Any],
    resolve: Callable[[str], Any],
    marks: Any = None,
) -> bool:
    """Return whether ``parent_layer``'s snapshot for ``target_layer`` follows from the graph.

    Parameters
    ----------
    target_layer:
        Child op whose argument binding is checked.
    parent_layer:
        Parent op bound to one or more of the child's argument slots.
    saved_out:
        Returns an op's retained saved out, or ``None`` when capture omitted it.
    resolve:
        Resolves a recorded parent label to its op.
    marks:
        The trace's capture marks (its kept foreign journal), or ``None`` when there
        are none. ``written_slots`` (foreign writes) and ``alias_slots`` (alias
        relabels) map ``(parent raw label, child raw label)`` to the parent's bound
        slots at capture; ``unrecorded_writes`` holds raw labels of tensors passed to
        a disclosed unrecordable mutating call.

    Returns
    -------
    bool
        ``True`` when there is no snapshot, no saved out to compare it with (other
        checks own missing evidence), the snapshot equals the saved out exactly, or a
        recorded in-place write explains the difference; ``False`` otherwise.
    """

    snapshot = child_version_snapshot(parent_layer, target_layer)
    saved = saved_out(parent_layer)
    if not isinstance(snapshot, torch.Tensor) or not isinstance(saved, torch.Tensor):
        return True
    same_geometry = tuple(snapshot.shape) == tuple(saved.shape) and snapshot.dtype == saved.dtype
    if same_geometry and tensor_nanequal(snapshot, saved, allow_tolerance=False):
        return True
    if record_member(parent_layer, "is_inplace", False):
        if _write_alias_geometry(snapshot, parent_layer, saved_out, resolve):
            return True
        return snapshot.dtype == saved.dtype and _slots_marked(
            target_layer, parent_layer, resolve, getattr(marks, "alias_slots", None) or {}
        )
    if not same_geometry:
        return False
    if _consumed_write_reaches(target_layer, parent_layer, resolve):
        return True
    if record_member(parent_layer, "_label_raw", None) in (
        getattr(marks, "unrecorded_writes", None) or ()
    ):
        return True
    return _slots_marked(
        target_layer, parent_layer, resolve, getattr(marks, "written_slots", None) or {}
    )


def _write_alias_geometry(
    snapshot: torch.Tensor,
    parent_layer: Op,
    saved_out: Callable[[Op], Any],
    resolve: Callable[[str], Any],
) -> bool:
    """Return whether an in-place parent's snapshot has the geometry of an alias it wrote.

    A reader bound to an in-place op reads the written view or an alias on the write's
    receiver chain (the view it wrote through, back to its base), so the snapshot has
    the dtype of the write and the shape of one op on that chain. Content is not
    compared: two same-shape in-place parents swapped, or a base and a same-shape view
    of it swapped, still pass (named blind spots in the tripwire tests).
    """

    written = saved_out(parent_layer)
    if not isinstance(written, torch.Tensor) or snapshot.dtype != written.dtype:
        return False
    if _covers_set_storage(snapshot, parent_layer):
        return True
    shape = tuple(snapshot.shape)
    for label in receiver_lineage(parent_layer, resolve):
        alias = saved_out(resolve(label))
        if isinstance(alias, torch.Tensor) and tuple(alias.shape) == shape:
            return True
    return False


def _covers_set_storage(snapshot: torch.Tensor, parent_layer: Op) -> bool:
    """Return whether ``snapshot`` spans exactly the storage a ``set_(storage, ...)`` parent bound.

    ``set_(storage, offset, size, stride)`` rebinds its receiver onto that storage, so a
    later reader of the storage's owner is bound to the write and reads the whole storage,
    not the receiver's view of it. Capture saves the storage argument by value at the
    call; only a snapshot of exactly that many bytes (the dtype is checked by the caller)
    is explained, and only for the storage overload of ``set_``.
    """

    if record_member(parent_layer, "func_name", None) != "set_":
        return False
    saved_args: tuple[Any, ...] = tuple(record_member(parent_layer, "saved_args", None) or ())
    storage = saved_args[1] if len(saved_args) > 1 else None
    return isinstance(storage, torch.UntypedStorage) and (
        snapshot.numel() * snapshot.element_size() == storage.nbytes()
    )


def _slots_marked(
    target_layer: Op,
    parent_layer: Op,
    resolve: Callable[[str], Any],
    marks: Mapping[tuple[str, str], frozenset[tuple[str, Any]]],
) -> bool:
    """Return whether capture marked this snapshot (foreign write or alias relabel) for its current slots."""

    key = (
        record_member(parent_layer, "_label_raw", None),
        record_member(target_layer, "_label_raw", None),
    )
    marked = marks.get(key)
    return bool(marked) and marked == _bound_slots(target_layer, parent_layer, resolve)


def _bound_slots(
    target_layer: Op, parent_layer: Op, resolve: Callable[[str], Any]
) -> frozenset[tuple[str, Any]]:
    """Return the ``(args|kwargs, position)`` slots of ``target_layer`` bound to ``parent_layer``."""

    positions = record_member(target_layer, "parent_arg_positions", None) or {}
    return frozenset(
        (kind, position)
        for kind in ("args", "kwargs")
        for position, label in (positions.get(kind) or {}).items()
        if resolve(label) is parent_layer
    )


def _consumed_write_reaches(
    target_layer: Op, parent_layer: Op, resolve: Callable[[str], Any]
) -> bool:
    """Return whether an in-place parent of ``target_layer`` wrote through ``parent_layer``.

    The write is the in-place op's receiver chain (the view it wrote through, back to
    its base); ``parent_layer`` must sit on it, and the child must consume the write.
    """

    for label in target_layer.parents:
        writer = resolve(label)
        if writer is parent_layer or not record_member(writer, "is_inplace", False):
            continue
        if parent_layer.label in receiver_lineage(writer, resolve):
            return True
    return False
