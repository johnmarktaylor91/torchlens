"""Shape, ordering, slicing, concatenation, and padding RF rules."""

from __future__ import annotations

from .._rules import ReceptiveFieldRuleContext, _RuleResult, register_rf_rule


@register_rf_rule("flatten", "reshape", "view", "squeeze", "unsqueeze")
def reshape(context: ReceptiveFieldRuleContext) -> _RuleResult:
    """Dissolve grid identity conservatively when captured ranks or axes change."""

    if not context.in_shapes:
        return context.unknown("reshape is missing its input shape")
    if len(context.in_shapes[0]) == len(context.out_shape):
        return context.full(exact=False, note="reshape family awaits a focused exactness golden")
    return context.dissolve(note="reshape changed rank and may mix tracked axes")


@register_rf_rule("permute", "transpose", "movedim", "swapaxes")
def permute(context: ReceptiveFieldRuleContext) -> _RuleResult:
    """Use an exact captured output-to-parent axis mapping when dimensions are known."""

    rank = len(context.in_shapes[0]) if context.in_shapes else len(context.out_shape)
    dims = context.arg("dims", None)
    if not isinstance(dims, (tuple, list)) or len(dims) != rank:
        return context.unknown("permutation dimensions were not captured")
    return context.full(
        exact=False,
        note="permutation family awaits a focused exactness golden",
    )


@register_rf_rule("cat", "concat", "stack")
def concatenate(context: ReceptiveFieldRuleContext) -> _RuleResult:
    """Use a sound whole-input envelope for branch-switching concatenation geometry."""

    return context.full(
        exact=False, note="concatenation segment routing is conservatively enveloped"
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
    """Use an upper bound for padding until pad mode and offsets are fully captured."""

    return context.full(exact=False, note="padding mode/offset is conservatively enveloped")
