"""Convolution and pooling receptive-field rules."""

from __future__ import annotations

from fractions import Fraction
from typing import cast

from .._query import map_adaptive_pool_index_set, map_transposed_convolution_index_set
from .._rules import ReceptiveFieldRuleContext, _RuleResult, register_rf_rule


def _tuple(value: object, rank: int) -> tuple[int, ...]:
    """Normalize a scalar or captured sequence to a fixed-rank integer tuple."""

    if isinstance(value, tuple):
        result = tuple(int(item) for item in value)
    elif isinstance(value, list):
        result = tuple(int(item) for item in value)
    else:
        result = (int(cast(int, value)),) * rank
    if len(result) == 1:
        result *= rank
    if len(result) != rank:
        raise ValueError("captured receptive-field parameter has the wrong rank")
    return result


@register_rf_rule("conv1d", "conv2d", "conv3d", "convolution")
def convolution(context: ReceptiveFieldRuleContext) -> _RuleResult:
    """Emit the exact standard convolution window recurrence."""

    kernel = context.cfg("kernel_size")
    if kernel is None:
        return context.unknown("convolution is missing kernel_size")
    rank = len(_tuple(kernel, 1)) if isinstance(kernel, (tuple, list)) else 1
    return context.window(
        kernel=kernel,
        stride=context.cfg("stride", (1,) * rank),
        padding=context.cfg("padding", (0,) * rank),
        dilation=context.cfg("dilation", (1,) * rank),
    )


@register_rf_rule("conv_transpose1d", "conv_transpose2d", "conv_transpose3d")
def transposed_convolution(context: ReceptiveFieldRuleContext) -> _RuleResult:
    """Emit phase-aware transposed-convolution geometry and its exact set callback."""

    kernel = context.cfg("kernel_size")
    if kernel is None or not context.in_shapes:
        return context.unknown("transposed convolution is missing captured kernel or input shape")
    rank = len(kernel) if isinstance(kernel, (tuple, list)) else 1
    kernels = _tuple(kernel, rank)
    strides = _tuple(context.cfg("stride", (1,) * rank), rank)
    paddings = _tuple(context.cfg("padding", (0,) * rank), rank)
    dilations = _tuple(context.cfg("dilation", (1,) * rank), rank)
    input_extent = context.in_shapes[0][-rank:]
    edges = tuple(
        (
            (Fraction(1, stride), Fraction(padding - dilation * (size - 1), stride)),
            (Fraction(1, stride), Fraction(padding, stride)),
        )
        for size, stride, padding, dilation in zip(
            kernels, strides, paddings, dilations, strict=True
        )
    )

    def map_index_set(axis: int, output_set: object) -> tuple[object, bool]:
        """Enumerate only feasible input residues for one transposed-convolution axis."""

        return map_transposed_convolution_index_set(
            axis,
            output_set,
            kernel=kernels,
            stride=strides,
            padding=paddings,
            dilation=dilations,
            input_extent=input_extent,
        )

    exact = all(stride == 1 for stride in strides)
    return context.window_edges(
        edges,
        exact=exact,
        note=None if exact else "stride > 1 uses a sub-pixel descriptor envelope",
        map_index_set=map_index_set,
    )


@register_rf_rule(
    "max_pool1d",
    "max_pool2d",
    "max_pool3d",
    "avg_pool1d",
    "avg_pool2d",
    "avg_pool3d",
    "lp_pool1d",
    "lp_pool2d",
)
def pool(context: ReceptiveFieldRuleContext) -> _RuleResult:
    """Emit standard pooling windows, honoring PyTorch's kernel-sized stride default."""

    kernel = context.cfg("kernel_size")
    if kernel is None:
        return context.unknown("pooling is missing kernel_size")
    rank = len(kernel) if isinstance(kernel, (tuple, list)) else 1
    ceil_mode = bool(context.cfg("ceil_mode", False))
    return context.window(
        kernel=kernel,
        stride=context.cfg("stride", kernel),
        padding=context.cfg("padding", (0,) * rank),
        dilation=context.cfg("dilation", (1,) * rank),
        exact=not ceil_mode,
        note="ceil_mode uses a final-window envelope" if ceil_mode else None,
    )


@register_rf_rule(
    "adaptive_avg_pool1d",
    "adaptive_avg_pool2d",
    "adaptive_avg_pool3d",
    "adaptive_max_pool1d",
    "adaptive_max_pool2d",
    "adaptive_max_pool3d",
)
def adaptive_pool(context: ReceptiveFieldRuleContext) -> _RuleResult:
    """Emit exact global pooling or a staircase descriptor with an exact unit callback."""

    if not context.in_shapes:
        return context.unknown("adaptive pooling is missing its input shape")
    output_size = context.cfg("output_size", context.arg("output_size"))
    if output_size is None:
        return context.unknown("adaptive pooling is missing output_size")
    rank = len(output_size) if isinstance(output_size, (tuple, list)) else 1
    outputs = _tuple(output_size, rank)
    inputs = tuple(int(value) for value in context.in_shapes[0][-rank:])
    if all(value == 1 for value in outputs):
        return context.full(
            axes=tuple(range(len(context.in_shapes[0]) - rank, len(context.in_shapes[0]))),
            exact=False,
            note="adaptive global pooling awaits a focused exactness golden",
        )
    edges = tuple(
        ((Fraction(input_size, output_size), -1), (Fraction(input_size, output_size), 1))
        for input_size, output_size in zip(inputs, outputs, strict=True)
    )

    def map_index_set(axis: int, output_set: object) -> tuple[object, bool]:
        """Map adaptive bins by their exact floor/ceil boundaries."""

        return map_adaptive_pool_index_set(
            axis, output_set, input_extent=inputs, output_extent=outputs
        )

    return context.window_edges(edges, exact=False, map_index_set=map_index_set)
