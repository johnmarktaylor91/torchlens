"""Predicates deciding whether a pruned orphan op's nondeterministic value mattered."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .._utils._record_group_table import record_member

if TYPE_CHECKING:
    from .._data_classes.trace import Trace


def _child_edge_totally_overwrites(self: Trace, walked_label: str, child_label: str) -> bool:
    """Return whether a child op TOTALLY overwrites the walked tensor's bytes (r53 hon_2).

    The sanitized pruned-orphan walk stops at a total-writer edge whose
    DESTINATION is the walked tensor (``copy_``/``zero_``/``fill_``/RNG-fill
    receiver, or an ``out=`` destination): the post-write value is independent
    of the walked nondeterministic bytes, so a pruned ``empty -> fill_(1) ->
    branch`` chain must NOT be flagged (empty-plus-fill is THE idiomatic
    scratch init). The walk CONTINUES when the walked tensor is the VALUE
    SOURCE of the writer (``dst.copy_(walked)``). Unknown or partial in-place
    writers never stop the walk (fail closed).
    """

    from .._utils._uninit_alloc import qualname_is_uninit_total_writer

    op = self._raw_graph_ws.raw_layer_dict.get(child_label)
    if op is None:
        return False
    func_id = record_member(op, "func_id", None)
    namespace = getattr(func_id, "namespace", None)
    qualname = getattr(func_id, "qualname", None)
    arg_locs = record_member(op, "parent_arg_positions", None)
    args_locs = arg_locs.get("args", {}) if isinstance(arg_locs, dict) else {}
    kwargs_locs = arg_locs.get("kwargs", {}) if isinstance(arg_locs, dict) else {}
    if (
        namespace in ("torch", "torch.Tensor", "torch.nn.functional")
        and kwargs_locs.get("out") == walked_label
    ):
        return True
    return qualname_is_uninit_total_writer(namespace, qualname) and args_locs.get(0) == walked_label


def _rng_orphan_drove_control_or_output(
    self: Trace, start_label: str, escape_sources: frozenset[str]
) -> bool:
    """Return whether an orphaned nondeterministic op's value steered control or output.

    Walks the forward (child) chain from ``start_label``. The nondeterministic
    result (a seeded torch-RNG draw or an uninitialized-memory family product,
    r53 hon_2) "influenced control" when the op itself or any descendant is a
    recorded tensor->host escape source (its value was read by
    ``bool()``/``.item()`` to steer pure-Python control flow); it "influenced
    output" when a descendant is an output node (defensive: an output-reaching
    op is not normally orphaned). The walk is SANITIZED: it never crosses a
    total-writer edge whose destination is the walked tensor
    (``_child_edge_totally_overwrites``), so an empty-then-fully-written
    scratch chain stays clean. A genuinely-dead draw whose result feeds nothing
    reaches neither and returns ``False``, so it stays VERIFIED.
    """

    seen: set[str] = set()
    stack = [start_label]
    while stack:
        label = stack.pop()
        if label in seen:
            continue
        seen.add(label)
        op = self._raw_graph_ws.raw_layer_dict.get(label)
        if op is None:
            continue
        if label in escape_sources or getattr(op, "is_output", False):
            return True
        for child_label in getattr(op, "children", ()) or ():
            if child_label in seen:
                continue
            if _child_edge_totally_overwrites(self, label, child_label):
                continue
            stack.append(child_label)
    return False


def _orphan_is_uninit_alloc_source(self: Trace, op: Any) -> bool:
    """Return whether an orphaned op is an uninitialized-memory value source (r53 hon_2).

    Keys on the shared closed family table (``utils/rng.py``): the ``empty``
    factory family taints unless its product has zero elements; a resize
    spelling taints only when it GREW its receiver beyond the pre-call element
    count (shrink/same-size preserves the prefix), failing closed to tainted
    when either shape is unreadable. The whole family is clean when the
    capture-time ambient snapshot proves deterministic fill.
    """

    from .._utils._uninit_alloc import (
        deterministic_fill_governs,
        qualname_is_uninit_growth_resize,
        qualname_is_uninitialized_alloc,
    )

    func_id = record_member(op, "func_id", None)
    namespace = getattr(func_id, "namespace", None)
    qualname = getattr(func_id, "qualname", None)
    is_factory = qualname_is_uninitialized_alloc(namespace, qualname)
    is_resize = qualname_is_uninit_growth_resize(namespace, qualname)
    if not is_factory and not is_resize:
        return False
    snapshot = self._runnable.capture_ambient
    if isinstance(snapshot, dict) and deterministic_fill_governs(
        snapshot.get("deterministic_algorithms"),
        snapshot.get("fill_uninitialized_memory"),
    ):
        return False

    def _numel(shape: Any) -> int | None:
        """Element count for a fully-static integer shape tuple, else ``None``."""

        if not isinstance(shape, tuple):
            return None
        numel = 1
        for dim in shape:
            if not isinstance(dim, int):
                return None
            numel *= dim
        return numel

    out_numel = _numel(getattr(op, "shape", None))
    if is_factory:
        return out_numel is None or out_numel > 0
    arg_locs = record_member(op, "parent_arg_positions", None)
    args_locs = arg_locs.get("args", {}) if isinstance(arg_locs, dict) else {}
    receiver = self._raw_graph_ws.raw_layer_dict.get(args_locs.get(0, ""))
    pre_numel = _numel(getattr(receiver, "shape", None)) if receiver is not None else None
    return pre_numel is None or out_numel is None or out_numel > pre_numel
