"""Dense mixing and cumulative-operation receptive-field rules."""

from __future__ import annotations

from .._rules import ReceptiveFieldRuleContext, _RuleResult, register_rf_rule


@register_rf_rule("linear", "addmm")
def linear(context: ReceptiveFieldRuleContext) -> _RuleResult:
    """Mark a dense operand's final feature axis as an exact full dependency."""

    return context.full(
        axes=(-1,), exact=False, note="linear family awaits a focused exactness golden"
    )


@register_rf_rule("matmul", "bmm", "einsum")
def matrix_multiply(context: ReceptiveFieldRuleContext) -> _RuleResult:
    """Use a sound full-input envelope for multi-operand contracted products."""

    return context.full(exact=False, note="contracted axes are conservatively enveloped")


@register_rf_rule("cumsum", "cummax")
def cumulative(context: ReceptiveFieldRuleContext) -> _RuleResult:
    """Bound cumulative dependence on the captured dimension only."""

    rank = len(context.in_shapes[0]) if context.in_shapes else len(context.out_shape)
    dim = context.arg("dim", context.cfg("dim", None))
    if not isinstance(dim, int) or rank == 0:
        return context.unknown("cumulative dimension was not captured")
    return context.full(
        axes=(dim % rank,),
        exact=False,
        note="cumulative prefix uses a containing selected-axis envelope",
    )
