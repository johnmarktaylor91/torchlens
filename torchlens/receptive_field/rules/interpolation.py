"""Interpolation receptive-field rules with branch-correct source scales."""

from __future__ import annotations

from fractions import Fraction

from .._query import _IndexSet, map_interpolation_index_set
from .._rules import ReceptiveFieldRuleContext, _RuleResult, register_rf_rule
from ._utils import number_tuple


def _interpolation_edges(
    mode: str,
    inputs: tuple[int, ...],
    outputs: tuple[int, ...],
    *,
    align_corners: bool | None,
    scale_factor: tuple[int | float | Fraction, ...] | None,
    recompute_scale_factor: bool | None,
) -> tuple[tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]], ...]:
    """Return a containing affine envelope for the captured interpolation branch."""

    edges = []
    for axis, (input_size, output_size) in enumerate(zip(inputs, outputs, strict=True)):
        supplied_scale = None if scale_factor is None else Fraction(scale_factor[axis])
        if align_corners and mode in {"linear", "bilinear", "trilinear", "bicubic"}:
            slope = Fraction(0) if output_size == 1 else Fraction(input_size - 1, output_size - 1)
            offset = Fraction(0)
        else:
            slope = (
                Fraction(input_size, output_size)
                if supplied_scale is None or recompute_scale_factor is True
                else Fraction(1, 1) / supplied_scale
            )
            if mode == "nearest":
                offset = Fraction(0)
            elif mode == "nearest-exact":
                offset = slope / 2
            else:
                offset = slope / 2 - Fraction(1, 2)
        radius = 2 if mode == "bicubic" else 1
        edges.append(((slope, offset - radius), (slope, offset + radius)))
    return tuple(edges)


@register_rf_rule(
    "interpolate",
    "upsample",
    "upsample_nearest1d",
    "upsample_nearest2d",
    "upsample_bilinear2d",
    "upsample_trilinear3d",
)
def interpolate(context: ReceptiveFieldRuleContext) -> _RuleResult:
    """Emit a phase envelope and exact supported-mode per-unit tap callback."""

    if not context.in_shapes:
        return context.unknown("interpolation is missing its input shape")
    mode = context.cfg("mode", context.arg("mode", None))
    if not isinstance(mode, str):
        return context.unknown("interpolation mode was not captured")
    antialias = context.arg("antialias", False)
    if antialias is True:
        return context.unknown("antialias interpolation is not geometrically certified")
    size = context.cfg("size", context.arg("size", None))
    scale = context.cfg("scale_factor", context.arg("scale_factor", None))
    align_corners = context.arg("align_corners", None)
    recompute = context.arg("recompute_scale_factor", None)
    if size is None and scale is None:
        return context.unknown("interpolation size/scale branch was not captured")
    rank = (
        len(size)
        if isinstance(size, (tuple, list))
        else (len(scale) if isinstance(scale, (tuple, list)) else len(context.out_shape) - 2)
    )
    if rank <= 0:
        return context.unknown("interpolation spatial rank is undeterminable")
    inputs = tuple(int(value) for value in context.in_shapes[0][-rank:])
    outputs = tuple(int(value) for value in context.out_shape[-rank:])
    if mode not in {"linear", "bilinear", "trilinear", "bicubic", "nearest", "nearest-exact"}:
        return context.unknown("interpolation mode is not supported")
    scale_factor = None if scale is None else number_tuple(scale, rank)
    if scale is not None and scale_factor is None:
        return context.unknown("interpolation scale_factor was malformed")
    edges = _interpolation_edges(
        mode,
        inputs,
        outputs,
        align_corners=align_corners if isinstance(align_corners, bool) else None,
        scale_factor=scale_factor,
        recompute_scale_factor=recompute if isinstance(recompute, bool) else None,
    )

    def map_index_set(axis: int, output_set: _IndexSet) -> tuple[_IndexSet, bool]:
        """Map one output progression through branch-correct interpolation taps."""

        return map_interpolation_index_set(
            axis,
            output_set,
            mode=mode,
            input_extent=inputs,
            output_extent=outputs,
            align_corners=align_corners if isinstance(align_corners, bool) else None,
            scale_factor=scale_factor,
            recompute_scale_factor=recompute if isinstance(recompute, bool) else None,
        )

    return context.window_edges(edges, exact=False, map_index_set=map_index_set)
