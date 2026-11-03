"""Dense mixing and cumulative-operation receptive-field rules."""

from __future__ import annotations

from ..._capture.arg_positions import _normalize_func_name
from .._rules import ReceptiveFieldRuleContext, _RuleResult, register_rf_rule
from ._utils import sole_data_parent, whole_axes_by_parent

# Captured argument roles of each dense family's data operand (``input`` for
# ``F.linear``, ``mat1`` for ``addmm``); every other tensor operand is a weight
# or bias.
_DENSE_DATA_ROLES = {
    "linear": frozenset({"args:0", "kwargs:input"}),
    "addmm": frozenset({"args:1", "kwargs:mat1"}),
}


@register_rf_rule("linear", "addmm")
def linear(context: ReceptiveFieldRuleContext) -> _RuleResult:
    """Mark the data operand's final feature axis as a whole-extent dependency.

    ``y[..., o] = sum_i W[o, i] x[..., i] (+ b[o])``: every output element reads
    the complete feature axis of the data operand at the same leading position,
    and nothing else of it. With the weight a parameter (the data operand is
    the only traced parent) that is the exact potential support, under the same
    generic-weight convention as a dense convolution's channel axis. A traced
    (input-derived) weight or bias operand is bounded by its whole extent and
    the result is an upper bound.
    """

    family = _normalize_func_name(str(context.op.func_name or ""))
    roles = _DENSE_DATA_ROLES.get(family, _DENSE_DATA_ROLES["linear"])
    if sole_data_parent(context, roles):
        return context.full(
            axes=(-1,),
            exact=True,
            note="dense layer reads the whole feature axis at each leading position",
        )
    return context.full(
        axes=whole_axes_by_parent(context, roles, (-1,)),
        exact=False,
        note="traced dense operands are bounded by their whole extent",
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
