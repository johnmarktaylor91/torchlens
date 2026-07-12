"""Elementwise, broadcasting, reduction, and embedding RF rules."""

from __future__ import annotations

from .._rules import ReceptiveFieldRuleContext, _RuleResult, register_rf_rule


@register_rf_rule(
    "relu",
    "gelu",
    "silu",
    "sigmoid",
    "tanh",
    "leaky_relu",
    "elu",
    "selu",
    "dropout",
    "dropout2d",
    "dropout3d",
    "clone",
    "detach",
    "to",
    "contiguous",
    "add",
    "sub",
    "mul",
    "div",
    "maximum",
    "minimum",
    "pow",
    "abs",
    "neg",
    "where",
)
def elementwise(context: ReceptiveFieldRuleContext) -> _RuleResult:
    """Return a sound whole-input envelope pending individual family goldens."""

    return context.full(exact=False, note="elementwise family awaits a focused exactness golden")


@register_rf_rule("mean", "sum", "amax", "amin", "var", "std")
def reduction(context: ReceptiveFieldRuleContext) -> _RuleResult:
    """Mark reduced parent axes as exact whole-extent dependencies."""

    rank = len(context.in_shapes[0]) if context.in_shapes else len(context.out_shape)
    raw = context.arg("dim", context.cfg("dim", None))
    if raw is None:
        axes = tuple(range(rank))
    elif isinstance(raw, (tuple, list)):
        axes = tuple(int(axis) % rank for axis in raw)
    else:
        axes = (int(raw) % rank,)
    return context.full(
        axes=axes, exact=False, note="reduction family awaits a focused exactness golden"
    )


@register_rf_rule("embedding")
def embedding(context: ReceptiveFieldRuleContext) -> _RuleResult:
    """Pass index-tensor coordinates through an embedding lookup exactly."""

    return context.full(exact=False, note="embedding family awaits a focused exactness golden")
