"""Attention and normalization-over-axis receptive-field rules."""

from __future__ import annotations

from .._rules import ReceptiveFieldRuleContext, _RuleResult, register_rf_rule


@register_rf_rule("softmax", "log_softmax")
def softmax(context: ReceptiveFieldRuleContext) -> _RuleResult:
    """Make the normalized dimension an exact full dependency."""

    rank = len(context.in_shapes[0]) if context.in_shapes else len(context.out_shape)
    dim = context.arg("dim", context.cfg("dim", -1))
    if not isinstance(dim, int):
        return context.unknown("softmax dimension was not captured")
    return context.full(
        axes=(dim % rank),
        exact=False,
        note="softmax family awaits a focused exactness golden",
    )


@register_rf_rule("scaled_dot_product_attention")
def scaled_dot_product_attention(context: ReceptiveFieldRuleContext) -> _RuleResult:
    """Globalize attention axes and widen broadcast or grouped key/value axes."""

    mask = context.arg("attn_mask", None)
    causal = context.arg("is_causal", False)
    masked = mask is not None or causal is True
    base_axes = (-2, -1)
    axes_by_parent: dict[str, tuple[int, ...]] = {"default": base_axes}
    input_shapes = context.op.input_shapes
    if len(input_shapes) >= 3 and input_shapes[0] is not None:
        query_shape = tuple(input_shapes[0])
        enable_gqa = context.arg("enable_gqa", False) is True
        for parent_index in (1, 2):
            parent_shape = input_shapes[parent_index]
            if parent_shape is None or parent_index >= len(context.op.parents):
                continue
            shape = tuple(parent_shape)
            widened = list(base_axes)
            if len(shape) == len(query_shape):
                for axis in range(max(len(shape) - 2, 0)):
                    query_extent = int(query_shape[axis])
                    parent_extent = int(shape[axis])
                    is_head_axis = axis == len(shape) - 3
                    if parent_extent == 1 and query_extent != 1:
                        widened.append(axis)
                    elif enable_gqa and is_head_axis and parent_extent < query_extent:
                        widened.append(axis)
            axes_by_parent[context.op.parents[parent_index]] = tuple(widened)
    return context.full(
        axes=axes_by_parent,
        exact=False,
        note=(
            "attention sequence dependence uses a containing mask-aware envelope"
            if masked
            else "attention sequence dependence uses a containing role-merged envelope"
        ),
    )
