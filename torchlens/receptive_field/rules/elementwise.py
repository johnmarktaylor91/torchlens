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
    "relu6",
    "hardswish",
    "hardsigmoid",
    "hardtanh",
    "mish",
    "softplus",
    "dropout",
    "dropout2d",
    "dropout3d",
    "clone",
    "detach",
    "to",
    "contiguous",
    "abs",
    "neg",
    "clamp",
    "clip",
    "exp",
    "log",
    "log1p",
    "sqrt",
    "rsqrt",
    "sin",
    "cos",
    "floor",
    "ceil",
    "round",
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
    "iadd",
    "sub",
    "isub",
    "mul",
    "imul",
    "div",
    "idiv",
    "truediv",
    "itruediv",
    "maximum",
    "minimum",
    "pow",
    "where",
)
def elementwise_binary(context: ReceptiveFieldRuleContext) -> _RuleResult:
    """Binary/n-ary elementwise: parents merge at the engine level.

    Each parent uses a trailing-aligned identity map. Extent-one broadcast axes
    become slope-zero maps in the engine, and the parent branches are then
    unioned by the ordinary merge machinery.
    """

    return _RuleResult(
        "passthrough",
        {"axis_alignment": "trailing"},
        "elementwise parents contribute same-position or broadcast coordinates",
    )


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
    return _RuleResult(
        "full",
        {
            "axes": axes,
            "surviving_parent_axes": tuple(axis for axis in range(rank) if axis not in axes),
        },
        "reduced axes depend on their complete captured extent",
    )


@register_rf_rule("embedding")
def embedding(context: ReceptiveFieldRuleContext) -> _RuleResult:
    """Map every index-tensor axis to the matching embedding output axis exactly."""

    if not context.in_shapes:
        return context.unknown("embedding is missing its index-tensor shape")
    input_rank = len(context.in_shapes[0])
    if len(context.out_shape) != input_rank + 1:
        return context.unknown("embedding output rank does not append one feature axis")
    return context.axis_map(
        {axis: axis for axis in range(input_rank)},
        note="embedding gathers one feature vector at each index-tensor position",
    )


@register_rf_rule("embedding_bag")
def embedding_bag(context: ReceptiveFieldRuleContext) -> _RuleResult:
    """Use a containing whole-index bound for bag-dependent embedding gathers."""

    if not context.in_shapes:
        return context.unknown("embedding_bag is missing its index-tensor shape")
    input_rank = len(context.in_shapes[0])
    return _RuleResult(
        "full",
        {"axes": tuple(range(input_rank)), "exact": False, "axis_alignment": "trailing"},
        "embedding_bag uses a containing bound over captured index positions",
    )
