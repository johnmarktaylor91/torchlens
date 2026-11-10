"""Layer mirror reads: the representative op behind a Layer and its mirrored values."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from .._trace_core.relation_views import LAYER_FROZENSET_VIEW_FIELDS, LAYER_TUPLE_VIEW_FIELDS
from .._utils._record_group_table import LAYER_MOVED_MEMBERS, op_spelling
from ._layer_spec import _LAYER_MIRROR_SPEC

if TYPE_CHECKING:
    from .layer import Layer
    from .op import Op


def _layer_rep_op(layer: Layer) -> Op | None:
    """Return the representative (first-pass) op backing one Layer's mirrors.

    Reads raw ``__dict__`` storage so descriptor bodies never re-enter the
    attribute protocol. The op accessor names its ops by store row, sorted by
    pass, so the first pair is the first pass: the op the former ``__init__``
    copied from. The row is read through the trace core's view table.
    """

    ops = layer.__dict__.get("ops")
    if ops is None:
        return None
    state = ops.__dict__
    try:
        row = state["_keys"][1]
        if row >= 0:
            # A typed local, not ``cast``: no call on the per-read path.
            op: Op | None = state["_core_ref"]()._views.ops[row]
            if op is not None:
                return op
    except (KeyError, IndexError, AttributeError, TypeError):
        pass
    # Slow path: no pass yet, an unkeyed first pass, a dropped view, or a
    # saved-shape accessor copy (reading ``_keys`` revives it).
    return cast("Op", ops._op_at(0)) if getattr(ops, "_keys", None) else None


def _layer_mirror_read(layer: Layer, name: str) -> Any:
    """Return the mirror value for ``name`` (no shadow consulted).

    Raises ``AttributeError`` when the layer has no representative op yet or
    the op's source cell is unset — the caller (descriptor or state
    enumeration) translates that into the historical missing-field behavior.
    """

    rep = _layer_rep_op(layer)
    if rep is None:
        raise AttributeError(name)
    source, normalize = _LAYER_MIRROR_SPEC[name]
    value = getattr(rep, source)
    return normalize(value) if normalize is not None else value


#: Layer relation-view fields whose Op spelling moved into a concern group
#: (``module_call_stack`` -> ``module_context.call_stack``), with that Op path.
LAYER_VIEW_OP_PATHS: dict[str, tuple[str, str]] = {
    name: (group, leaf)
    for name in (*LAYER_TUPLE_VIEW_FIELDS, *LAYER_FROZENSET_VIEW_FIELDS)
    if name not in LAYER_MOVED_MEMBERS
    for group, _, leaf in [op_spelling(name).partition(".")]
    if leaf
}


def layer_view_read(layer: Layer, name: str) -> Any:
    """Read an unstored Layer view field through its single pass's Op group path.

    ``Layer.module_call_stack`` stays a Layer member while ``Op.module_call_stack``
    refuses (moved to ``module_context.call_stack``): the Layer read follows the
    Op table path instead of the flat Op spelling, so the Op refusal never fires
    on a Layer read. A multi-pass layer refuses as ambiguous, like every per-pass
    field.
    """

    if layer.num_passes > 1:
        return layer._single_pass_or_error(name)
    group, leaf = LAYER_VIEW_OP_PATHS[name]
    return getattr(getattr(layer.ops[0], group), leaf)
