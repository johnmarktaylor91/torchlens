"""Fail-closed rules for operations with nonlocal sequence semantics."""

from __future__ import annotations

from .._rules import ReceptiveFieldRuleContext, _RuleResult, register_rf_rule
from ._utils import int_tuple


@register_rf_rule("roll")
def roll(context: ReceptiveFieldRuleContext) -> _RuleResult:
    """Use a whole-axis upper bound because circular wrapping is not pointwise."""

    rank = len(context.in_shapes[0]) if context.in_shapes else len(context.out_shape)
    dims = context.arg("dims", context.arg("dimension", None))
    if dims is None:
        axes = tuple(range(rank))
    else:
        dimensions = int_tuple(dims)
        if dimensions is None:
            return context.unknown("roll dimensions were not captured as integers")
        axes = tuple(axis % rank for axis in dimensions)
    return context.full(axes=axes, exact=False, note="roll is circular and not pointwise")
