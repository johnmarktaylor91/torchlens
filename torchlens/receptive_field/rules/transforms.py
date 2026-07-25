"""Shape, ordering, slicing, concatenation, and padding RF rules."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .._rules import ReceptiveFieldRuleContext, _RuleResult, register_rf_rule
from ._utils import int_tuple


def _axis_map_result(
    mapping: Mapping[int, int], *, note: str, selected_parent_axes: Sequence[int] = ()
) -> _RuleResult:
    """Build an axis-map result with optional fixed-index parent axes.

    Parameters
    ----------
    mapping:
        Output-axis to parent-axis correspondence.
    note:
        Provenance note for the resulting rule.
    selected_parent_axes:
        Parent axes removed by scalar indexing. These cannot be represented as
        output-indexed grids and are conservatively bounded by their full extent.

    Returns
    -------
    _RuleResult
        Exact surviving-axis mapping with honest selected-axis metadata.
    """

    return _RuleResult(
        "axis_map",
        {
            "out_to_parent_axis": dict(mapping),
            "selected_parent_axes": tuple(selected_parent_axes),
        },
        note,
    )


def _shape_axis_map(
    parent_shape: Sequence[int], output_shape: Sequence[int]
) -> dict[int, int] | None:
    """Match axes across reshapes that only insert or remove singleton dimensions.

    Parameters
    ----------
    parent_shape:
        Shape before the structural transform.
    output_shape:
        Shape after the structural transform.

    Returns
    -------
    dict[int, int] | None
        Output-to-parent map, or ``None`` when non-singleton axes were mixed.
    """

    parent_non_singleton = [axis for axis, extent in enumerate(parent_shape) if extent != 1]
    output_non_singleton = [axis for axis, extent in enumerate(output_shape) if extent != 1]
    if [parent_shape[axis] for axis in parent_non_singleton] != [
        output_shape[axis] for axis in output_non_singleton
    ]:
        return None
    mapping = dict(zip(output_non_singleton, parent_non_singleton, strict=True))
    unused_parent = [axis for axis, extent in enumerate(parent_shape) if extent == 1]
    for output_axis, extent in enumerate(output_shape):
        if extent == 1 and unused_parent:
            mapping[output_axis] = unused_parent.pop(0)
    return mapping


@register_rf_rule("flatten", "reshape", "view", "squeeze", "unsqueeze")
def reshape(context: ReceptiveFieldRuleContext) -> _RuleResult:
    """Remap singleton-only reshapes exactly and refuse mixed-axis identities."""

    if not context.in_shapes:
        return context.unknown("reshape is missing its input shape")
    parent_shape = context.in_shapes[0]
    mapping = _shape_axis_map(parent_shape, context.out_shape)
    if mapping is None:
        return context.dissolve(note="reshape mixes non-singleton axes")
    mapped_parent_axes = set(mapping.values())
    removed_singletons = tuple(
        axis
        for axis, extent in enumerate(parent_shape)
        if axis not in mapped_parent_axes and extent == 1
    )
    return _axis_map_result(
        mapping,
        selected_parent_axes=removed_singletons,
        note="reshape only inserts or removes singleton axes",
    )


@register_rf_rule("permute", "transpose", "movedim", "swapaxes")
def permute(context: ReceptiveFieldRuleContext) -> _RuleResult:
    """Use an exact captured output-to-parent axis mapping when dimensions are known."""

    rank = len(context.in_shapes[0]) if context.in_shapes else len(context.out_shape)
    name = str(context.op.func_name).replace("_", "").lower()
    positional = tuple(context.op.non_tensor_pos_args)
    if name == "permute":
        dims = positional
        if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
            dims = tuple(dims[0])
        normalized = int_tuple(dims)
        if normalized is None or len(normalized) != rank:
            return context.unknown("permutation dimensions were not captured")
        order = tuple(axis % rank for axis in normalized)
    elif name == "transpose":
        dim0 = context.cfg("dim0", context.arg("dim0", None))
        dim1 = context.cfg("dim1", context.arg("dim1", None))
        if not isinstance(dim0, int) or not isinstance(dim1, int):
            return context.unknown("transpose dim0/dim1 were not captured")
        order_list = list(range(rank))
        first, second = dim0 % rank, dim1 % rank
        order_list[first], order_list[second] = order_list[second], order_list[first]
        order = tuple(order_list)
    elif name == "swapaxes":
        if len(positional) < 2 or not all(isinstance(item, int) for item in positional[:2]):
            return context.unknown("swapaxes dimensions were not captured")
        order_list = list(range(rank))
        first, second = int(positional[0]) % rank, int(positional[1]) % rank
        order_list[first], order_list[second] = order_list[second], order_list[first]
        order = tuple(order_list)
    else:
        if len(positional) < 2:
            return context.unknown("movedim source/destination dimensions were not captured")
        sources = int_tuple(positional[0])
        destinations = int_tuple(positional[1])
        if sources is None or destinations is None or len(sources) != len(destinations):
            return context.unknown("movedim dimensions were malformed")
        normalized_sources = tuple(axis % rank for axis in sources)
        normalized_destinations = tuple(axis % rank for axis in destinations)
        remaining = [axis for axis in range(rank) if axis not in normalized_sources]
        for destination, source in sorted(zip(normalized_destinations, normalized_sources)):
            remaining.insert(destination, source)
        order = tuple(remaining)
    if sorted(order) != list(range(rank)):
        return context.unknown("permutation dimensions did not form a complete axis order")
    return context.axis_map(
        {output_axis: parent_axis for output_axis, parent_axis in enumerate(order)},
        note="structural permutation preserves axis coordinates exactly",
    )


@register_rf_rule("cat", "concat", "stack")
def concatenate(context: ReceptiveFieldRuleContext) -> _RuleResult:
    """Route each concatenated parent through the engine's ordinary branch union."""

    if not context.in_shapes:
        return context.unknown("concatenation is missing parent shapes")
    raw_dim = context.cfg("dim", context.arg("dim", 0))
    if not isinstance(raw_dim, int):
        return context.unknown("concatenation dimension was not captured")
    is_stack = str(context.op.func_name).replace("_", "").lower() == "stack"
    rank = len(context.out_shape)
    axis = raw_dim % rank
    return _RuleResult(
        "passthrough",
        {"concatenate_axis": axis, "stack": is_stack},
        "concatenated parents are slice-routed and unioned by the engine",
    )


@register_rf_rule(
    "pad",
    "constant_pad_nd",
    "reflection_pad1d",
    "reflection_pad2d",
    "reflection_pad3d",
    "replication_pad1d",
    "replication_pad2d",
    "replication_pad3d",
)
def pad(context: ReceptiveFieldRuleContext) -> _RuleResult:
    """Map padded coordinates to exact constant, reflected, or replicated sources."""

    if not context.in_shapes:
        return context.unknown("padding is missing its input shape")
    parent_shape = context.in_shapes[0]
    raw_padding = context.cfg("padding", context.arg("pad", None))
    padding = int_tuple(raw_padding)
    if padding is None or len(padding) % 2 or len(padding) > 2 * len(parent_shape):
        return context.unknown("padding widths were not captured as integer pairs")
    mode = str(context.cfg("mode", context.arg("mode", "constant"))).lower()
    padded_rank = len(padding) // 2
    left_by_axis = [0] * len(parent_shape)
    for local_axis in range(padded_rank):
        parent_axis = len(parent_shape) - 1 - local_axis
        left_by_axis[parent_axis] = padding[2 * local_axis]

    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    first_padded_axis = len(parent_shape) - padded_rank
    for axis, extent in enumerate(parent_shape[first_padded_axis:], start=first_padded_axis):
        left = left_by_axis[axis]
        if mode == "constant" or axis < len(parent_shape) - padded_rank:
            edges.append(((1, -left), (1, -left)))
        else:
            edges.append(((0, 0), (0, extent - 1)))

    def map_padding_indices(local_axis: int, output_set: Any) -> tuple[Sequence[int], bool]:
        """Map one output index set through the captured padding mode exactly.

        Parameters
        ----------
        local_axis:
            Axis index within the full-rank edge tuple.
        output_set:
            Engine-owned bounded output index set.

        Returns
        -------
        tuple[collections.abc.Sequence[int], bool]
            Mapped source indices and exactness.
        """

        parent_axis = first_padded_axis + local_axis
        extent = parent_shape[parent_axis]
        left = left_by_axis[parent_axis]
        mapped: list[int] = []
        for output_index in output_set.values():
            source = int(output_index) - left
            if mode == "constant":
                if 0 <= source < extent:
                    mapped.append(source)
            elif mode in {"replicate", "replication"}:
                mapped.append(min(max(source, 0), extent - 1))
            elif mode in {"reflect", "reflection"}:
                if extent <= 1:
                    return (), False
                period = 2 * (extent - 1)
                reflected = source % period
                mapped.append(reflected if reflected < extent else period - reflected)
            else:
                return tuple(range(extent)), False
        return mapped, mode in {"constant", "replicate", "replication", "reflect", "reflection"}

    return _RuleResult(
        "window_edges",
        {
            "per_axis_edges": edges,
            "exact": mode == "constant",
            "preserve_non_window_axes": True,
        },
        f"{mode} padding maps output coordinates to captured input borders",
        map_index_set=map_padding_indices,
    )


@register_rf_rule("getitem")
def getitem(context: ReceptiveFieldRuleContext) -> _RuleResult:
    """Map basic integer and slice indexing without tainting descendants."""

    if not context.in_shapes or not context.op.non_tensor_pos_args:
        return context.unknown("getitem index metadata was not captured")
    parent_shape = context.in_shapes[0]
    raw_key = context.op.non_tensor_pos_args[0]
    key = raw_key if isinstance(raw_key, tuple) else (raw_key,)
    expanded: list[object] = []
    ellipsis_count = sum(item is Ellipsis for item in key)
    if ellipsis_count > 1:
        return context.unknown("getitem contains multiple ellipses")
    consumed = sum(item is not None and item is not Ellipsis for item in key)
    for item in key:
        if item is Ellipsis:
            expanded.extend([slice(None)] * (len(parent_shape) - consumed))
        else:
            expanded.append(item)
    expanded.extend([slice(None)] * (len(parent_shape) - consumed))
    output_to_parent: dict[int, int] = {}
    selected: list[int] = []
    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    parent_axis = 0
    output_axis = 0
    same_rank = True
    for item in expanded:
        if item is None:
            same_rank = False
            output_axis += 1
            continue
        if parent_axis >= len(parent_shape):
            return context.unknown("getitem index rank exceeds the parent rank")
        if isinstance(item, int):
            same_rank = False
            selected.append(parent_axis)
            parent_axis += 1
            continue
        if not isinstance(item, slice):
            return context.data_dependent("getitem uses tensor or data-dependent indexing")
        start, _stop, step = item.indices(parent_shape[parent_axis])
        output_to_parent[output_axis] = parent_axis
        edges.append(((step, start), (step, start)))
        parent_axis += 1
        output_axis += 1
    if same_rank and len(edges) == len(parent_shape):
        first_changed = next(
            (axis for axis, edge in enumerate(edges) if edge != ((1, 0), (1, 0))),
            len(edges),
        )
        if first_changed == len(edges):
            return context.passthrough(note="full slices preserve every coordinate")
        return _RuleResult(
            "window_edges",
            {
                "per_axis_edges": edges[first_changed:],
                "exact": True,
                "preserve_non_window_axes": True,
            },
            "basic slicing is an exact affine map",
        )
    return _axis_map_result(
        output_to_parent,
        selected_parent_axes=selected,
        note="getitem preserves surviving axes; fixed-index axes use a conservative extent bound",
    )
