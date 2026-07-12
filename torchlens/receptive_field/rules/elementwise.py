"""Elementwise, broadcasting, reduction, and embedding RF rules."""

from __future__ import annotations

from .._rules import ReceptiveFieldRuleContext, _RuleResult, register_rf_rule
from ._utils import int_tuple


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
    "abs",
    "neg",
)
def pointwise(context: ReceptiveFieldRuleContext) -> _RuleResult:
    """Shape-preserving pointwise ops leave the receptive field unchanged (exact).

    Each output element depends only on the input element at the same position, so
    the receptive field passes through these ops untouched.
    """

    return context.passthrough(
        note="pointwise: each output element depends only on the same input element"
    )


@register_rf_rule(
    "add",
    "sub",
    "mul",
    "div",
    "maximum",
    "minimum",
    "pow",
    "where",
)
def elementwise_binary(context: ReceptiveFieldRuleContext) -> _RuleResult:
    """Binary/n-ary elementwise: parents merge at the engine level.

    Broadcast-axis exactness awaits a focused family golden; a sound whole-input
    envelope is returned meanwhile.
    """

    return context.full(exact=False, note="elementwise binary/broadcast awaits a focused golden")


@register_rf_rule("mean", "sum", "amax", "amin", "var", "std")
def reduction(context: ReceptiveFieldRuleContext) -> _RuleResult:
    """Mark reduced parent axes as exact whole-extent dependencies."""

    rank = len(context.in_shapes[0]) if context.in_shapes else len(context.out_shape)
    raw = context.arg("dim", context.cfg("dim", None))
    if raw is None:
        axes = tuple(range(rank))
    else:
        dimensions = int_tuple(raw)
        if dimensions is None:
            return context.unknown("reduction dimensions were not captured as integers")
        axes = tuple(axis % rank for axis in dimensions)
    return context.full(
        axes=axes, exact=False, note="reduction family awaits a focused exactness golden"
    )


@register_rf_rule("embedding")
def embedding(context: ReceptiveFieldRuleContext) -> _RuleResult:
    """Pass index-tensor coordinates through an embedding lookup exactly."""

    return context.full(exact=False, note="embedding family awaits a focused exactness golden")
