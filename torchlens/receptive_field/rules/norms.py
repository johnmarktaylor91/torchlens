"""Mode-aware normalization receptive-field rules."""

from __future__ import annotations

from .._rules import ReceptiveFieldRuleContext, _RuleResult, register_rf_rule


def _bool_arg(context: ReceptiveFieldRuleContext, name: str, position: int) -> bool | None:
    """Read a required raw boolean argument without guessing an absent switch."""

    value = context.op.non_tensor_kwargs.get(name)
    if value is None and position < len(context.op.non_tensor_pos_args):
        value = context.op.non_tensor_pos_args[position]
    return value if isinstance(value, bool) else None


@register_rf_rule("batch_norm")
def batch_norm(context: ReceptiveFieldRuleContext) -> _RuleResult:
    """Represent captured batch-stat coupling as full batch and spatial geometry."""

    training = _bool_arg(context, "training", 0)
    if training is None:
        return context.unknown("batch_norm statistics switch was not captured")
    if not training:
        return context.passthrough()
    rank = len(context.in_shapes[0]) if context.in_shapes else len(context.out_shape)
    return context.full(axes=(0, *range(2, rank)), note="batch-stat normalization couples batch")


@register_rf_rule("instance_norm")
def instance_norm(context: ReceptiveFieldRuleContext) -> _RuleResult:
    """Distinguish current-stat instance normalization from running-stat evaluation."""

    use_input_stats = _bool_arg(context, "use_input_stats", 0)
    if use_input_stats is None:
        return context.unknown("instance_norm statistics switch was not captured")
    if not use_input_stats:
        return context.full(
            exact=False,
            note="running-stat instance_norm awaits a focused exactness golden",
        )
    rank = len(context.in_shapes[0]) if context.in_shapes else len(context.out_shape)
    return context.full(axes=tuple(range(2, rank)))


@register_rf_rule("group_norm")
def group_norm(context: ReceptiveFieldRuleContext) -> _RuleResult:
    """Mark channel and spatial group statistics as exact whole dependencies."""

    rank = len(context.in_shapes[0]) if context.in_shapes else len(context.out_shape)
    return context.full(axes=tuple(range(1, rank)))


@register_rf_rule("layer_norm")
def layer_norm(context: ReceptiveFieldRuleContext) -> _RuleResult:
    """Mark the captured normalized suffix axes as exact whole dependencies."""

    shape = context.arg("normalized_shape", context.cfg("normalized_shape", None))
    if shape is None:
        return context.unknown("layer_norm normalized_shape was not captured")
    width = len(shape) if isinstance(shape, (tuple, list)) else 1
    rank = len(context.in_shapes[0]) if context.in_shapes else len(context.out_shape)
    return context.full(
        axes=tuple(range(rank - width, rank)),
        exact=False,
        note="layer_norm family awaits a focused exactness golden",
    )
