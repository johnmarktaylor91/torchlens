"""Mode-aware normalization receptive-field rules."""

from __future__ import annotations

import math
from typing import cast

from .._rules import ReceptiveFieldRuleContext, _RuleResult, register_rf_rule
from ._utils import sole_data_parent, whole_axes_by_parent

# Captured argument roles of a normalization's data operand.
_NORM_DATA_ROLES = frozenset({"args:0", "kwargs:input"})


def _bool_arg(context: ReceptiveFieldRuleContext, name: str) -> bool | None:
    """Read a normalization-mode switch from filtered capture metadata.

    Tensor arguments are omitted from ``non_tensor_pos_args``, while positional ``None``
    placeholders remain. For BatchNorm and InstanceNorm the statistics switch is therefore
    the first boolean in that filtered sequence, regardless of the tensor placeholders.
    """

    value = context.op.call.non_tensor_kwargs.get(name)
    if isinstance(value, bool):
        return value
    return next(
        (item for item in context.op.call.non_tensor_pos_args if isinstance(item, bool)),
        None,
    )


@register_rf_rule("batch_norm")
def batch_norm(context: ReceptiveFieldRuleContext) -> _RuleResult:
    """Represent captured batch-stat coupling as full batch and spatial geometry."""

    training = _bool_arg(context, "training")
    if training is None:
        return context.unknown("batch_norm statistics switch was not captured")
    if not training:
        return context.passthrough()
    rank = len(context.in_shapes[0]) if context.in_shapes else len(context.out_shape)
    coupled_axes = (0, *range(2, rank))
    full_axes_by_parent: dict[str, object] = {"default": coupled_axes}
    parent_to_child_axes: dict[str, dict[int, int]] = {}
    channel_parent_roles = {
        "args:1",
        "args:2",
        "args:3",
        "args:4",
        "kwargs:running_mean",
        "kwargs:running_var",
        "kwargs:weight",
        "kwargs:bias",
    }
    for parent_index, parent_label in enumerate(context.op.parents):
        if context.parent_role(parent_index) in channel_parent_roles:
            full_axes_by_parent[parent_label] = ()
            parent_to_child_axes[parent_label] = {0: 1}
    return _RuleResult(
        "full",
        {
            "axes": full_axes_by_parent,
            "exact": True,
            "parent_to_child_axes": parent_to_child_axes,
        },
        "batch-stat normalization couples batch",
    )


@register_rf_rule("instance_norm")
def instance_norm(context: ReceptiveFieldRuleContext) -> _RuleResult:
    """Distinguish current-stat instance normalization from running-stat evaluation."""

    use_input_stats = _bool_arg(context, "use_input_stats")
    if use_input_stats is None:
        return context.unknown("instance_norm statistics switch was not captured")
    if not use_input_stats:
        return context.passthrough()
    rank = len(context.in_shapes[0]) if context.in_shapes else len(context.out_shape)
    coupled_axes = tuple(range(2, rank))
    full_axes_by_parent: dict[str, object] = {"default": coupled_axes}
    parent_to_child_axes: dict[str, dict[int, int]] = {}
    channel_parent_roles = {
        "args:1",
        "args:2",
        "args:3",
        "args:4",
        "kwargs:running_mean",
        "kwargs:running_var",
        "kwargs:weight",
        "kwargs:bias",
    }
    for parent_index, parent_label in enumerate(context.op.parents):
        if context.parent_role(parent_index) in channel_parent_roles:
            full_axes_by_parent[parent_label] = ()
            parent_to_child_axes[parent_label] = {0: 1}
    return _RuleResult(
        "full",
        {
            "axes": full_axes_by_parent,
            "exact": True,
            "parent_to_child_axes": parent_to_child_axes,
        },
    )


@register_rf_rule("group_norm")
def group_norm(context: ReceptiveFieldRuleContext) -> _RuleResult:
    """Represent group statistics without claiming cross-group channel dependence."""

    rank = len(context.in_shapes[0]) if context.in_shapes else len(context.out_shape)
    groups = context.cfg("num_groups", context.arg("num_groups", None))
    if not isinstance(groups, int) or groups <= 0:
        return context.unknown("group_norm num_groups was not captured")
    if groups == 1:
        return context.full(axes=tuple(range(1, rank)))
    return context.full(
        axes=tuple(range(1, rank)),
        exact=False,
        note="group slices use a containing channel-and-spatial envelope",
    )


@register_rf_rule("layer_norm")
def layer_norm(context: ReceptiveFieldRuleContext) -> _RuleResult:
    """Mark the captured normalized suffix axes as whole-extent dependencies.

    Each output element reads every element of its normalized slice (through
    the slice mean and variance) and nothing outside it. The relation is exact
    when the data operand is the only traced parent and the derivative is
    generically nonzero on the whole slice: for ``n`` normalized elements the
    Jacobian of ``(x - mean) / std`` is ``(I - 11^T / n - y y^T / n) / std``,
    which has no structural zero once ``n >= 3``. At ``n == 2`` the
    off-diagonal term ``1 + y_0 y_1`` vanishes identically without ``eps``, so
    exactness needs a captured positive ``eps``; at ``n == 1`` the output is
    the bias alone and the claimed slice is only an upper bound.
    """

    shape = context.arg("normalized_shape", context.cfg("normalized_shape", None))
    if shape is None:
        return context.unknown("layer_norm normalized_shape was not captured")
    dims = tuple(shape) if isinstance(shape, (tuple, list)) else (shape,)
    width = len(dims)
    rank = len(context.in_shapes[0]) if context.in_shapes else len(context.out_shape)
    axes = tuple(range(rank - width, rank))
    if sole_data_parent(context, _NORM_DATA_ROLES) and _normalized_slice_is_exact(context, dims):
        return context.full(
            axes=axes,
            exact=True,
            note="layer_norm reads its whole normalized slice at each leading position",
        )
    return context.full(
        axes=whole_axes_by_parent(context, _NORM_DATA_ROLES, axes),
        exact=False,
        note="layer_norm slice is a containing bound (traced affine operands or a degenerate slice)",
    )


def _normalized_slice_is_exact(
    context: ReceptiveFieldRuleContext, dims: tuple[object, ...]
) -> bool:
    """Return whether a layer-norm slice has no structurally zero derivative.

    Parameters
    ----------
    context:
        Rule context for the captured ``layer_norm``.
    dims:
        Captured ``normalized_shape`` entries.

    Returns
    -------
    bool
        ``True`` for at least three normalized elements, or for two with a
        captured positive ``eps``.
    """

    if not dims or not all(isinstance(dim, int) and not isinstance(dim, bool) for dim in dims):
        return False
    count = math.prod(int(cast("int", dim)) for dim in dims)
    if count >= 3:
        return True
    if count != 2:
        return False
    eps = context.arg("eps", context.cfg("eps", None))
    return isinstance(eps, (int, float)) and not isinstance(eps, bool) and eps > 0
