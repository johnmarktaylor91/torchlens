"""Index-set maps for the geometry-specific window families.

The window kinds whose parent interval is not a plain affine image
(transposed convolution, adaptive pooling, interpolation) compute their image
here, away from the walk itself. Each takes and returns index sets, so this
module stays free of descriptor and operation types.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from math import ceil, floor, gcd, ulp
from typing import Any, cast

from ._index_set import _clipped_progression, _IndexSet, _Progression, _progression_sumset


def _coerce_index_set(raw: object, *, exact: bool) -> _IndexSet:
    """Normalize callback values into the walk's bounded set representation."""
    if isinstance(raw, _IndexSet):
        return _IndexSet(raw.progressions, raw.exact and exact)
    if isinstance(raw, range):
        return _IndexSet.from_values(raw, exact=exact)
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return _IndexSet.from_values((int(value) for value in raw), exact=exact)
    raise TypeError("map_index_set/map_interval callbacks must return an integer index set")


def _parameter_tuple(value: int | Sequence[int], rank: int | None = None) -> tuple[int, ...]:
    """Normalize callback integer parameters without accepting strings."""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result = tuple(int(item) for item in value)
    else:
        result = (int(cast(Any, value)),)
    if rank is not None and len(result) == 1:
        result *= rank
    if rank is not None and len(result) != rank:
        raise ValueError("callback parameter rank mismatch")
    return result


def _fraction_tuple(
    value: int | float | Fraction | Sequence[int | float | Fraction], rank: int
) -> tuple[Fraction, ...]:
    """Normalize supplied interpolation scales using exact rational conversion."""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result = tuple(Fraction(item) for item in value)
    else:
        result = (Fraction(cast(Any, value)),) * rank
    if len(result) != rank:
        raise ValueError("scale_factor rank mismatch")
    return result


def _call_interval_callback(
    callback: Any, axis: int, output_set: _IndexSet
) -> tuple[_IndexSet, bool]:
    """Lift a monotone interval callback across each progression hull."""
    results: list[_IndexSet] = []
    exact = output_set.exact
    for progression in output_set.progressions:
        starts = progression.values() if progression.step > 1 else (progression.start,)
        for start in starts:
            stop = start if progression.step > 1 else progression.stop
            raw, interval_exact = callback(axis, (start, stop))
            results.append(_coerce_index_set(raw, exact=bool(interval_exact)))
            exact = exact and bool(interval_exact)
    combined = _IndexSet.union(results)
    return _IndexSet(combined.progressions, combined.exact and exact), exact


def map_transposed_convolution_index_set(
    axis: int,
    output_set: _IndexSet,
    *,
    kernel: int | Sequence[int],
    stride: int | Sequence[int],
    padding: int | Sequence[int],
    dilation: int | Sequence[int],
    input_extent: int | Sequence[int],
) -> tuple[_IndexSet, bool]:
    """Map transposed-convolution outputs by authoritative feasible-tap enumeration.

    The gcd test is only a necessary pre-filter. Every surviving output/tap pair must
    also produce an integral, in-range input index.
    """
    kernels = _parameter_tuple(kernel)
    strides = _parameter_tuple(stride, len(kernels))
    paddings = _parameter_tuple(padding, len(kernels))
    dilations = _parameter_tuple(dilation, len(kernels))
    extents = _parameter_tuple(input_extent, len(kernels))
    kernel_value = kernels[axis]
    stride_value = strides[axis]
    padding_value = paddings[axis]
    dilation_value = dilations[axis]
    extent = extents[axis]
    progressions = output_set.progressions
    if len(progressions) == 1 and stride_value == 1 and kernel_value > 0 and dilation_value > 0:
        # Stride 1: every output/tap pair is integral, so the feasible inputs are
        # the sumset of the outputs (shifted by the padding) and the negated taps,
        # clipped to the input extent; computed in closed form when it is one
        # progression, else enumerated below.
        item = progressions[0]
        image = _progression_sumset(
            _Progression(item.start + padding_value, item.step, item.count),
            _Progression(-dilation_value * (kernel_value - 1), dilation_value, kernel_value),
        )
        if image is not None:
            return _clipped_progression(image, extent, exact=output_set.exact), True
    feasible: list[int] = []
    divisor = gcd(stride_value, dilation_value)
    for output_index in output_set.values():
        shifted = output_index + padding_value
        if shifted % divisor:
            continue
        for tap in range(kernel_value):
            numerator = shifted - dilation_value * tap
            if numerator % stride_value:
                continue
            input_index = numerator // stride_value
            if 0 <= input_index < extent:
                feasible.append(input_index)
    return _IndexSet.from_values(feasible, exact=output_set.exact), True


def map_adaptive_pool_index_set(
    axis: int,
    output_set: _IndexSet,
    *,
    input_extent: int | Sequence[int],
    output_extent: int | Sequence[int],
) -> tuple[_IndexSet, bool]:
    """Map adaptive-pooling bins exactly using floor/ceil endpoint rules."""
    inputs = _parameter_tuple(input_extent)
    outputs = _parameter_tuple(output_extent, len(inputs))
    values = (
        input_index
        for output_index in output_set.values()
        for input_index in range(
            floor(Fraction(output_index * inputs[axis], outputs[axis])),
            ceil(Fraction((output_index + 1) * inputs[axis], outputs[axis])),
        )
    )
    return _IndexSet.from_values(values, exact=output_set.exact), True


def map_interpolation_index_set(
    axis: int,
    output_set: _IndexSet,
    *,
    mode: str,
    input_extent: int | Sequence[int],
    output_extent: int | Sequence[int],
    align_corners: bool | None = None,
    scale_factor: int | float | Fraction | Sequence[int | float | Fraction] | None = None,
    recompute_scale_factor: bool | None = None,
) -> tuple[_IndexSet, bool]:
    """Map supported interpolation modes with branch-correct rational source indices."""
    inputs = _parameter_tuple(input_extent)
    outputs = _parameter_tuple(output_extent, len(inputs))
    scales = _fraction_tuple(scale_factor, len(inputs)) if scale_factor is not None else None
    source_scale = (
        Fraction(inputs[axis], outputs[axis])
        if scales is None or recompute_scale_factor is True
        else Fraction(1, 1) / scales[axis]
    )
    linear_modes = {"linear", "bilinear", "trilinear", "bicubic"}
    nearest_modes = {"nearest", "nearest-exact"}
    if mode not in linear_modes | nearest_modes:
        return _IndexSet.interval(0, inputs[axis] - 1, exact=False), False
    support: list[int] = []
    exact = output_set.exact
    for output_index in output_set.values():
        candidates: tuple[int, ...]
        source = _interpolation_source(
            output_index,
            inputs[axis],
            outputs[axis],
            align_corners=align_corners,
            supplied_scale=None if scales is None else scales[axis],
            recompute_scale_factor=recompute_scale_factor,
        )
        if mode == "nearest":
            phase = Fraction(output_index) * source_scale
            candidates = (floor(phase),)
        elif mode == "nearest-exact":
            phase = (Fraction(output_index) + Fraction(1, 2)) * source_scale
            candidates = (floor(phase),)
        elif source.denominator == 1:
            phase = source
            candidates = (int(source),)
        elif mode == "bicubic":
            phase = source
            base = floor(source)
            candidates = (base - 1, base, base + 1, base + 2)
        else:
            phase = source
            candidates = (floor(source), ceil(source))
        phase_float = float(phase)
        boundary = round(phase_float)
        if phase.denominator != 1 and abs(phase_float - boundary) <= ulp(phase_float):
            radius = 2 if mode == "bicubic" else 1
            candidates += tuple(range(boundary - radius, boundary + radius + 1))
            exact = False
        support.extend(min(max(candidate, 0), inputs[axis] - 1) for candidate in candidates)
    return _IndexSet.from_values(support, exact=exact), exact


def _interpolation_source(
    output_index: int,
    input_extent: int,
    output_extent: int,
    *,
    align_corners: bool | None,
    supplied_scale: Fraction | None,
    recompute_scale_factor: bool | None,
) -> Fraction:
    """Return the branch-correct linear-family source coordinate."""
    if align_corners:
        if output_extent == 1:
            return Fraction(0)
        return Fraction(output_index * (input_extent - 1), output_extent - 1)
    scale = (
        Fraction(input_extent, output_extent)
        if supplied_scale is None or recompute_scale_factor is True
        else Fraction(1, 1) / supplied_scale
    )
    return (Fraction(output_index) + Fraction(1, 2)) * scale - Fraction(1, 2)


__all__: list[str] = []
