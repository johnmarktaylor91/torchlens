"""Fail-closed rules for operations with nonlocal sequence semantics."""

from __future__ import annotations

from .._rules import ReceptiveFieldRuleContext, _RuleResult, register_rf_rule


@register_rf_rule("roll")
def roll(context: ReceptiveFieldRuleContext) -> _RuleResult:
    """Use a whole-axis upper bound because circular wrapping is not pointwise."""

    rank = len(context.in_shapes[0]) if context.in_shapes else len(context.out_shape)
    dims = context.arg("dims", context.arg("dimension", None))
    if dims is None:
        axes = tuple(range(rank))
    elif isinstance(dims, (tuple, list)):
        axes = tuple(int(axis) % rank for axis in dims)
    else:
        axes = (int(dims) % rank,)
    return context.full(axes=axes, exact=False, note="roll is circular and not pointwise")
