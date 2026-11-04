"""Attention and normalization-over-axis receptive-field rules."""

from __future__ import annotations

from .._causal import causal_offset_from_weights
from .._rules import ReceptiveFieldRuleContext, _RuleResult, register_rf_rule


@register_rf_rule("softmax", "log_softmax")
def softmax(context: ReceptiveFieldRuleContext) -> _RuleResult:
    """Make the normalized dimension a full dependency, or a causal prefix when masked.

    A causally masked softmax has exactly zero weight on every key after the
    row's own position, so the row's Jacobian with respect to those logits is
    zero: query ``i`` depends on keys ``0..i + offset`` only. The structure is
    read from the saved weights' zero pattern, which must be the same prefix
    pattern on every batch row and head.
    """

    rank = len(context.in_shapes[0]) if context.in_shapes else len(context.out_shape)
    dim = context.arg("dim", context.cfg("dim", -1))
    if not isinstance(dim, int):
        return context.unknown("softmax dimension was not captured")
    key_axis = dim % rank
    if key_axis >= 1:
        offset = causal_offset_from_weights(context.op, key_axis)
        if offset is not None:
            return _RuleResult(
                "causal_window",
                {
                    "mode": "self",
                    "query_axis": key_axis - 1,
                    "key_axis": key_axis,
                    "offset": offset,
                    "exact": True,
                    "full_axes": {},
                },
                f"causal softmax: row i depends on keys 0..i{offset:+d} (saved weights' zero pattern)",
            )
    return context.full(
        axes=key_axis,
        exact=False,
        note="softmax family awaits a focused exactness golden",
    )


def _widened_axes(
    query_shape: tuple[int, ...], shape: tuple[int, ...], gqa: bool
) -> tuple[int, ...]:
    """Batch or head axes a key/value operand broadcasts or groups over."""

    widened: list[int] = []
    if len(shape) != len(query_shape):
        return ()
    for axis in range(max(len(shape) - 2, 0)):
        query_extent = int(query_shape[axis])
        parent_extent = int(shape[axis])
        is_head_axis = axis == len(shape) - 3
        if (
            parent_extent == 1
            and query_extent != 1
            or gqa
            and is_head_axis
            and parent_extent < query_extent
        ):
            widened.append(axis)
    return tuple(widened)


def _role_relation(
    context: ReceptiveFieldRuleContext, full_axes: set[int], sequence: str, note: str
) -> _RuleResult:
    """Build one operand's relation from its merged full axes and sequence-axis mode."""

    if sequence == "window":
        return _RuleResult(
            "causal_window",
            {
                "mode": "cross",
                "query_axis": -2,
                "key_axis": -2,
                "offset": 0,
                "exact": True,
                "full_axes": dict.fromkeys(sorted(full_axes), False),
            },
            note,
        )
    axes = set(full_axes) | ({-2} if sequence == "full" else set())
    return context.full(axes=tuple(sorted(axes)), exact=False, note=note)


def _per_operand_attention(
    context: ReceptiveFieldRuleContext, causal: bool, note: str
) -> _RuleResult | None:
    """Per-operand fused-attention relations, or ``None`` when shapes are not usable."""

    parents = tuple(context.op.parents)
    shapes = tuple(context.op.input_shapes)
    if len(parents) < 3 or len(shapes) < 3:
        return None
    raw_query, raw_key, raw_value = shapes[:3]
    if raw_query is None or raw_key is None or raw_value is None:
        return None
    query_shape = tuple(int(extent) for extent in raw_query)
    key_shape = tuple(int(extent) for extent in raw_key)
    value_shape = tuple(int(extent) for extent in raw_value)
    rank = len(query_shape)
    if rank < 2 or len(key_shape) != rank or len(value_shape) != rank:
        return None
    gqa = context.arg("enable_gqa", False) is True
    sequence_mode = "window" if causal else "full"
    roles = (
        (parents[0], {rank - 1}, "identity"),
        (parents[1], {rank - 1, *_widened_axes(query_shape, key_shape, gqa)}, sequence_mode),
        (parents[2], set(_widened_axes(query_shape, value_shape, gqa)), sequence_mode),
    )
    merged: dict[str, tuple[set[int], str]] = {}
    order = {"identity": 0, "window": 1, "full": 2}
    for label, full_axes, sequence in roles:
        axes, mode = merged.get(label, (set(), "identity"))
        merged[label] = (axes | full_axes, max(mode, sequence, key=order.__getitem__))
    by_parent = {
        label: _role_relation(context, axes, mode, note) for label, (axes, mode) in merged.items()
    }
    return _RuleResult(
        "per_parent",
        {"by_parent": by_parent, "default": context.full(axes=(-2, -1), exact=False, note=note)},
        None,
    )


@register_rf_rule("scaled_dot_product_attention")
def scaled_dot_product_attention(context: ReceptiveFieldRuleContext) -> _RuleResult:
    """Relate each fused-attention operand; ``is_causal`` narrows the key/value sequence axis.

    The query keeps its sequence position and consumes its head dimension.
    Keys consume their head dimension and, with ``is_causal=True``, only the
    prefix ``0..i`` of their sequence axis reaches output row ``i``; values
    likewise. Broadcast or grouped (GQA) batch and head axes widen to their
    whole extent. An explicit ``attn_mask`` keeps the containing envelope.
    """

    mask = context.arg("attn_mask", None)
    causal = context.arg("is_causal", False) is True
    note = (
        "causal attention: output row i reads keys and values 0..i"
        if causal and mask is None
        else "attention sequence dependence uses a containing mask-aware envelope"
        if mask is not None or causal
        else "attention sequence dependence uses a containing role-merged envelope"
    )
    if mask is None:
        per_operand = _per_operand_attention(context, causal, note)
        if per_operand is not None:
            return per_operand
    base_axes = (-2, -1)
    axes_by_parent: dict[str, tuple[int, ...]] = {"default": base_axes}
    input_shapes = context.op.input_shapes
    if len(input_shapes) >= 3 and input_shapes[0] is not None:
        query_shape = tuple(int(e) for e in input_shapes[0])
        gqa = context.arg("enable_gqa", False) is True
        for parent_index in (1, 2):
            parent_shape = input_shapes[parent_index]
            if parent_shape is None or parent_index >= len(context.op.parents):
                continue
            shape = tuple(int(e) for e in parent_shape)
            widened = base_axes + _widened_axes(query_shape, shape, gqa)
            axes_by_parent[context.op.parents[parent_index]] = tuple(widened)
    return context.full(axes=axes_by_parent, exact=False, note=note)
