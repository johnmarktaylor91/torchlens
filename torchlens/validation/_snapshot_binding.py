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
  the written view (shape and content may differ); or
- the parent is not an in-place op, the snapshot has the parent's shape and dtype
  (an in-place write never changes either), and an in-place op the child ALSO
  consumes wrote through the parent's lineage: the call reads a base together
  with the view written into it, the base bound to its own lineage before the write.

Anything else is a binding the recorded graph cannot produce, and argument logging
fails. The check only adds failures; it never validates what failed before.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from .._utils._record_group_table import record_member
from .._utils.tensor_utils import tensor_nanequal
from ._replay_storage_context import receiver_lineage

if TYPE_CHECKING:
    from .._data_classes.op import Op
    from .._data_classes.trace import Trace


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


def snapshot_binding_explained(trace: Trace, target_layer: Op, parent_layer: Op) -> bool:
    """Return whether ``parent_layer``'s snapshot for ``target_layer`` follows from the graph.

    Parameters
    ----------
    trace:
        Trace being validated.
    target_layer:
        Child op whose argument binding is checked.
    parent_layer:
        Parent op bound to one or more of the child's argument slots.

    Returns
    -------
    bool
        ``True`` when there is no snapshot, no saved out to compare it with (other
        checks own missing evidence), the snapshot equals the saved out exactly, or a
        recorded in-place write explains the difference; ``False`` otherwise.
    """

    from ._core import _saved_out_payload  # deferred: cycle validation

    snapshot = child_version_snapshot(parent_layer, target_layer)
    saved = _saved_out_payload(parent_layer)
    if not isinstance(snapshot, torch.Tensor) or not isinstance(saved, torch.Tensor):
        return True
    same_geometry = tuple(snapshot.shape) == tuple(saved.shape) and snapshot.dtype == saved.dtype
    if same_geometry and tensor_nanequal(snapshot, saved, allow_tolerance=False):
        return True
    if record_member(parent_layer, "is_inplace", False):
        return True
    return same_geometry and _consumed_write_reaches(trace, target_layer, parent_layer)


def _consumed_write_reaches(trace: Trace, target_layer: Op, parent_layer: Op) -> bool:
    """Return whether an in-place parent of ``target_layer`` wrote through ``parent_layer``.

    The write is the in-place op's receiver chain (the view it wrote through, back to
    its base); ``parent_layer`` must sit on it, and the child must consume the write.
    """

    from ._core import _op_for_validation_label  # deferred: cycle validation

    def resolve(label: str) -> Any:
        """Resolve a recorded parent label to its op."""

        return _op_for_validation_label(trace, label)

    for label in target_layer.parents:
        writer = resolve(label)
        if writer is parent_layer or not record_member(writer, "is_inplace", False):
            continue
        if parent_layer.label in receiver_lineage(writer, resolve):
            return True
    return False
