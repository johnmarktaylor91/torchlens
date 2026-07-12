"""Convolution and pooling receptive-field rules."""

from __future__ import annotations

from fractions import Fraction

from .._query import _IndexSet, map_adaptive_pool_index_set, map_transposed_convolution_index_set
from .._rules import ReceptiveFieldRuleContext, _RuleResult, register_rf_rule
from ._utils import int_config, int_tuple, spatial_rank


@register_rf_rule("conv1d", "conv2d", "conv3d", "convolution")
def convolution(context: ReceptiveFieldRuleContext) -> _RuleResult:
    """Emit the exact standard convolution window recurrence."""

    raw_kernel = context.cfg("kernel_size")
    rank = spatial_rank(context, raw_kernel)
    if raw_kernel is None or rank is None:
        return context.unknown("convolution is missing kernel_size")
    kernel = int_tuple(raw_kernel, rank)
    stride = int_config(context, "stride", rank, default=1)
    padding = int_config(context, "padding", rank, default=0)
    dilation = int_config(context, "dilation", rank, default=1)
    if kernel is None or stride is None or padding is None or dilation is None:
        return context.unknown("convolution has malformed spatial parameters")
    return context.window(
        kernel=kernel,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )


@register_rf_rule("conv_transpose1d", "conv_transpose2d", "conv_transpose3d")
def transposed_convolution(context: ReceptiveFieldRuleContext) -> _RuleResult:
    """Emit phase-aware transposed-convolution geometry and its exact set callback."""

    raw_kernel = context.cfg("kernel_size")
    rank = spatial_rank(context, raw_kernel)
    if raw_kernel is None or rank is None or not context.in_shapes:
        return context.unknown("transposed convolution is missing captured kernel or input shape")
    kernels = int_tuple(raw_kernel, rank)
    strides = int_config(context, "stride", rank, default=1)
    paddings = int_config(context, "padding", rank, default=0)
    dilations = int_config(context, "dilation", rank, default=1)
    if kernels is None or strides is None or paddings is None or dilations is None:
        return context.unknown("transposed convolution has malformed spatial parameters")
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

    def map_index_set(axis: int, output_set: _IndexSet) -> tuple[_IndexSet, bool]:
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

    raw_kernel = context.cfg("kernel_size")
    rank = spatial_rank(context, raw_kernel)
    if raw_kernel is None or rank is None:
        return context.unknown("pooling is missing kernel_size")
    kernel = int_tuple(raw_kernel, rank)
    stride = int_config(context, "stride", rank, default=raw_kernel)
    padding = int_config(context, "padding", rank, default=0)
    dilation = int_config(context, "dilation", rank, default=1)
    if kernel is None or stride is None or padding is None or dilation is None:
        return context.unknown("pooling has malformed spatial parameters")
    ceil_mode = bool(context.cfg("ceil_mode", False))
    return context.window(
        kernel=kernel,
        stride=stride,
        padding=padding,
        dilation=dilation,
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
    rank = spatial_rank(context, output_size)
    if rank is None:
        return context.unknown("adaptive pooling spatial rank is undeterminable")
    outputs = int_tuple(output_size, rank)
    if outputs is None:
        return context.unknown("adaptive pooling has malformed output_size")
    inputs = tuple(int(value) for value in context.in_shapes[0][-rank:])
    if all(value == 1 for value in outputs):
        return context.full(
            axes=tuple(range(len(context.in_shapes[0]) - rank, len(context.in_shapes[0]))),
            exact=True,
        )
    edges = tuple(
        ((Fraction(input_size, output_size), -1), (Fraction(input_size, output_size), 1))
        for input_size, output_size in zip(inputs, outputs, strict=True)
    )

    def map_index_set(axis: int, output_set: _IndexSet) -> tuple[_IndexSet, bool]:
        """Map adaptive bins by their exact floor/ceil boundaries."""

        return map_adaptive_pool_index_set(
            axis, output_set, input_extent=inputs, output_extent=outputs
        )

    return context.window_edges(edges, exact=False, map_index_set=map_index_set)
