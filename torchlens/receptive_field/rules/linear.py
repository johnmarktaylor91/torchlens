"""Dense mixing and cumulative-operation receptive-field rules."""

from __future__ import annotations

from .._causal import causal_weights_source
from .._rules import ReceptiveFieldRuleContext, _RuleResult, register_rf_rule


@register_rf_rule("linear", "addmm")
def linear(context: ReceptiveFieldRuleContext) -> _RuleResult:
    """Mark a dense operand's final feature axis as an exact full dependency."""

    return context.full(
        axes=(-1,), exact=False, note="linear family awaits a focused exactness golden"
    )


def _batched_operand_shapes(
    context: ReceptiveFieldRuleContext,
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """Return the two operand shapes of a same-rank batched product, else ``None``."""

    parents = tuple(context.op.parents)
    shapes = tuple(context.op.input_shapes)
    if len(parents) != 2 or len(shapes) != 2 or parents[0] == parents[1]:
        return None
    left_shape, right_shape = shapes
    if left_shape is None or right_shape is None:
        return None
    left = tuple(int(extent) for extent in left_shape)
    right = tuple(int(extent) for extent in right_shape)
    out = context.out_shape
    rank = len(out)
    if rank < 2 or len(left) != rank or len(right) != rank:
        return None
    if left[:-2] != out[:-2] or right[:-2] != out[:-2]:
        return None
    if left[-2] != out[-2] or right[-1] != out[-1] or left[-1] != right[-2]:
        return None
    return left, right


@register_rf_rule("matmul", "bmm")
def matrix_multiply(context: ReceptiveFieldRuleContext) -> _RuleResult:
    """Map a batched product per operand: rows and columns pass through, the contraction is full.

    ``out[..., i, j] = sum_k left[..., i, k] * right[..., k, j]``: the left
    operand keeps its row axis and consumes its last axis; the right operand
    keeps its column axis and consumes its second-to-last axis. When the left
    operand is a causally masked softmax (attention weights), the consumed key
    axis of the right operand narrows to the causal prefix of the output row.
    Mixed-rank or broadcast products keep the conservative whole-input envelope.
    """

    shapes = _batched_operand_shapes(context)
    if shapes is None:
        return context.full(exact=False, note="contracted axes are conservatively enveloped")
    rank = len(context.out_shape)
    left_label, right_label = context.op.parents
    left = context.axis_map(
        {axis: axis for axis in range(rank - 1)},
        note="matrix product left operand: rows pass through, the contracted axis is consumed",
    )
    offset = causal_weights_source(context.op)
    if offset is None:
        right = context.axis_map(
            {**{axis: axis for axis in range(rank - 2)}, rank - 1: rank - 1},
            note="matrix product right operand: columns pass through, the contracted axis is consumed",
        )
    else:
        right = _RuleResult(
            "causal_window",
            {
                "mode": "cross",
                "query_axis": rank - 2,
                "key_axis": rank - 2,
                "offset": offset,
                "exact": True,
                "full_axes": {},
            },
            f"causal attention product: row i reads value rows 0..i{offset:+d}",
        )
    return _RuleResult(
        "per_parent",
        {"by_parent": {left_label: left, right_label: right}, "default": context.full(exact=False)},
        None,
    )


@register_rf_rule("einsum")
def einsum(context: ReceptiveFieldRuleContext) -> _RuleResult:
    """Use a sound full-input envelope for a general contracted product."""

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
