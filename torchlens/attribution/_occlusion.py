"""Occlusion attribution: scalar trace occlusion and the direct input map.

Two engines share one window discipline (attrib memo D8): direct eval/no-grad
forwards serve the input occlusion MAP (reference-implementation parity;
``overlap="average"`` default, ``"sum"`` explicit), while Trace fork/do
replay serves the scalar selection occlusion below (the activation SWEEP over
a trace selection is the lane's declared shed row -- its geometry,
replacement-policy seam, and per-engine pass-budget refusal tier ship here so
the sweep lands later without churn). Refusals are counted in DETERMINISTIC
per-engine pass counts, never wall-clock (D9): a time budget would make
identical inputs refuse on a laptop and run on a workstation. The refusal
message and fields carry the arithmetic -- computed pass count, measured
one-pass time, projected total.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from typing import Any, Literal, TypeAlias

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

from torchlens.attribution._core import (
    AttributionError,
    AttributionResult,
    InputKwargs,
    _AttributionTarget,
    _call_model,
    _normalize_model_inputs,
    _scalarize_output,
    _target_repr,
    _temporarily_eval,
)
from torchlens.intervention import mean_ablate, zero_ablate
from torchlens.selection import ResolvedSelection, Selection

# Deterministic per-engine pass budgets (attrib memo D9), sized by the
# measured 16.6x direct-vs-replay per-pass ratio (13.4 ms direct vs 222 ms
# replay per window on resnet18).
_DIRECT_MAX_PASSES = 4096
_TRACE_MAX_PASSES = 512

_OcclusionBaseline: TypeAlias = Literal["zeros", "mean", "blur"]
_OcclusionScore: TypeAlias = Callable[[Any], Tensor | float]


def _score_trace(trace: Any, score: _OcclusionScore) -> Tensor:
    """Evaluate and validate one scalar trace score.

    Parameters
    ----------
    trace
        Original or occluded Trace passed to the user scorer.
    score
        Callable returning one real scalar.

    Returns
    -------
    Tensor
        Detached scalar score.

    Raises
    ------
    AttributionError
        If the scorer does not return one finite real scalar.
    """

    value = score(trace)
    if isinstance(value, bool) or not isinstance(value, (Tensor, int, float)):
        raise AttributionError("occlusion score must return one real scalar")
    tensor = value if isinstance(value, Tensor) else torch.tensor(float(value))
    if tensor.numel() != 1 or tensor.is_complex():
        raise AttributionError("occlusion score must return one real scalar")
    scalar = tensor.detach().reshape(())
    if not bool(torch.isfinite(scalar)):
        raise AttributionError("occlusion score must return a finite scalar")
    return scalar


def _blur_edit(kernel_size: int) -> Callable[..., Tensor]:
    """Build a spatial mean-blur edit for the selection scatter engine.

    Parameters
    ----------
    kernel_size
        Odd spatial averaging kernel width.

    Returns
    -------
    Callable[..., Tensor]
        Hook that blurs an entire ``N,C,H,W`` activation. The selection engine
        scatters only selected elements from that replacement.
    """

    def blur(out: Tensor, *, hook: Any) -> Tensor:
        """Return a same-shaped spatially blurred activation."""

        del hook
        if out.ndim != 4 or not (out.is_floating_point() or out.is_complex()):
            raise AttributionError(
                "baseline='blur' requires a floating or complex N,C,H,W activation"
            )
        padding = kernel_size // 2
        if out.is_complex():
            real = F.avg_pool2d(out.real, kernel_size, stride=1, padding=padding)
            imag = F.avg_pool2d(out.imag, kernel_size, stride=1, padding=padding)
            return torch.complex(real, imag)
        return F.avg_pool2d(out, kernel_size, stride=1, padding=padding)

    return blur


def _occlusion_edit(baseline: _OcclusionBaseline, blur_kernel_size: int) -> Any:
    """Return the shipped or local edit implementing an explicit baseline.

    Parameters
    ----------
    baseline
        Named replacement policy.
    blur_kernel_size
        Odd width used only for ``baseline="blur"``.

    Returns
    -------
    Any
        Edit accepted by ``Trace.do``.
    """

    if baseline == "zeros":
        return zero_ablate()
    if baseline == "mean":
        return mean_ablate()
    if baseline == "blur":
        if isinstance(blur_kernel_size, bool) or blur_kernel_size < 3 or blur_kernel_size % 2 == 0:
            raise AttributionError("blur_kernel_size must be an odd integer of at least 3")
        return _blur_edit(blur_kernel_size)
    raise AttributionError("baseline must be 'zeros', 'mean', or 'blur'")


def _resolve_selection(trace: Any, selection: Any) -> ResolvedSelection:
    """Resolve a Selection or region producer against ``trace``.

    Parameters
    ----------
    trace
        Trace owning the values to occlude.
    selection
        Selection query, resolved selection, or ``__selection__`` producer.

    Returns
    -------
    ResolvedSelection
        Session-bound concrete selection.

    Raises
    ------
    AttributionError
        If ``selection`` does not implement the Selection protocol.
    """

    lifted = selection
    if not isinstance(lifted, (Selection, ResolvedSelection)):
        converter = getattr(lifted, "__selection__", None)
        if not callable(converter):
            raise AttributionError("occlusion selection must implement __selection__()")
        lifted = converter()
    if isinstance(lifted, ResolvedSelection):
        return lifted if lifted._trace is trace else lifted.align_to(trace)
    if isinstance(lifted, Selection):
        return lifted.resolve(trace)
    raise AttributionError("occlusion selection did not produce a Selection")


def occlusion(
    trace: Any,
    selection: Any,
    *,
    score: _OcclusionScore,
    baseline: _OcclusionBaseline = "zeros",
    blur_kernel_size: int = 3,
) -> AttributionResult:
    """Score the effect of explicitly occluding one resolved activation region.

    The result is ``score(original) - score(occluded)``. ``baseline`` is
    mandatory vocabulary even though it has a documented ``"zeros"`` default:
    zeros, the selected site's global mean, and a spatial blur answer different
    counterfactual questions. Occlusion runs through ``Trace.fork().do(...)`` so
    it inherits Selection masking and the shipped edit-then-scatter contract.

    Parameters
    ----------
    trace
        Live captured Trace with replayable saved values.
    selection
        Selection query, resolved selection, or region producer to occlude.
    score
        Callable receiving a Trace and returning one finite real scalar. It is
        invoked once on the original and once on the occluded fork.
    baseline
        Replacement policy: ``"zeros"`` (default), ``"mean"``, or ``"blur"``.
    blur_kernel_size
        Odd spatial averaging width for ``baseline="blur"``.

    Returns
    -------
    AttributionResult
        Scalar scored delta in ``values`` plus both endpoint scores and baseline
        disclosure in ``extra``.
    """

    resolved = _resolve_selection(trace, selection)
    if resolved._kind != "ACT":
        raise AttributionError("occlusion currently supports ACT selections only")
    edit = _occlusion_edit(baseline, blur_kernel_size)
    original_score = _score_trace(trace, score)
    fork = trace.fork(name="attribution_occlusion")
    try:
        fork_selection = resolved.align_to(fork)
        fork.do(fork_selection, edit)
        occluded_score = _score_trace(fork, score)
    finally:
        fork.cleanup()
    delta = original_score - occluded_score
    return AttributionResult(
        method="occlusion",
        values=delta,
        target_repr="trace score",
        extra={
            "baseline": baseline,
            "blur_kernel_size": blur_kernel_size if baseline == "blur" else None,
            "original_score": original_score,
            "occluded_score": occluded_score,
            "selection_digest": resolved.resolve_digest,
        },
    )


def _normalize_window(
    name: str,
    value: int | Sequence[int] | None,
    fallback: tuple[int, ...] | None,
    n_axes_hint: int | None,
) -> tuple[int, ...]:
    """Normalize a window or stride spec into a positive-int tuple.

    Parameters
    ----------
    name
        Parameter name for error reporting.
    value
        Int (one swept axis), sequence of ints, or ``None`` (use fallback).
    fallback
        Value used when ``value`` is ``None`` (strides default to the window).
    n_axes_hint
        Required length when known.

    Returns
    -------
    tuple[int, ...]
        Positive window extents per swept axis.

    Raises
    ------
    AttributionError
        If the spec is not positive ints of the right arity.
    """

    if value is None:
        if fallback is None:
            raise AttributionError(
                f"{name} is required. Remedy: pass an int or tuple of ints.",
                code="occlusion_geometry_invalid",
            )
        return fallback
    if isinstance(value, bool):
        raise AttributionError(
            f"{name} must be a positive int or tuple of positive ints. "
            "Remedy: pass window extents such as (15, 15).",
            code="occlusion_geometry_invalid",
        )
    normalized: tuple[int, ...]
    if isinstance(value, int):
        normalized = (value,)
    elif isinstance(value, Sequence):
        normalized = tuple(int(item) for item in value)
    else:
        raise AttributionError(
            f"{name} must be a positive int or tuple of positive ints. "
            "Remedy: pass window extents such as (15, 15).",
            code="occlusion_geometry_invalid",
        )
    if not normalized or any(item <= 0 for item in normalized):
        raise AttributionError(
            f"{name} must contain positive extents; got {normalized}. "
            "Remedy: pass window extents such as (15, 15).",
            code="occlusion_geometry_invalid",
        )
    if n_axes_hint is not None and len(normalized) != n_axes_hint:
        raise AttributionError(
            f"{name} has {len(normalized)} extents but the window sweeps "
            f"{n_axes_hint} axes. Remedy: align {name} with the window axes.",
            code="occlusion_geometry_invalid",
        )
    return normalized


def _axis_origins(size: int, window: int, stride: int) -> list[int]:
    """Enumerate window origins along one axis with full coverage.

    The policy is CLIPPED-EDGE full coverage: origins advance by ``stride``
    (never re-aligned), and the final window CLIPS at the boundary when it
    overhangs. Every element is covered by at least one window, and the
    origin sequence matches the reference implementation's occlusion grid
    exactly (parity-pinned in the captum oracle ledger).

    Parameters
    ----------
    size
        Axis extent.
    window
        Window extent along this axis.
    stride
        Stride along this axis.

    Returns
    -------
    list[int]
        Window origins in ascending order (``k * stride``).
    """

    if window >= size:
        return [0]
    n_steps = 1 + math.ceil((size - window) / stride)
    return [step * stride for step in range(n_steps)]


def _window_slices(
    leaf_shape: tuple[int, ...],
    window: tuple[int, ...],
    strides: tuple[int, ...],
) -> tuple[list[tuple[slice, ...]], list[int]]:
    """Enumerate the full window grid over the trailing swept axes.

    Axes before the swept tail (except the leading batch axis, which is never
    swept and never occluded away per-example) are fully covered by every
    window -- e.g. a ``(15, 15)`` window on a ``(1, 3, 224, 224)`` image
    occludes all channels of each patch.

    Parameters
    ----------
    leaf_shape
        Shape of the occluded leaf.
    window
        Window extents over the trailing ``len(window)`` axes.
    strides
        Strides over the same axes.

    Returns
    -------
    tuple[list[tuple[slice, ...]], list[int]]
        Per-window slice tuples (covering every axis except the batch axis)
        and the per-axis window counts.

    Raises
    ------
    AttributionError
        If the window does not fit the leaf geometry.
    """

    n_swept = len(window)
    if n_swept > len(leaf_shape) - 1:
        raise AttributionError(
            f"window sweeps {n_swept} axes but the occluded leaf has only "
            f"{len(leaf_shape) - 1} non-batch axes. Remedy: shorten the "
            "window tuple.",
            code="occlusion_geometry_invalid",
        )
    swept_sizes = leaf_shape[-n_swept:]
    for size, extent in zip(swept_sizes, window, strict=True):
        if extent > size:
            raise AttributionError(
                f"window extent {extent} exceeds the swept axis size {size}. "
                "Remedy: shrink the window to fit the input.",
                code="occlusion_geometry_invalid",
            )
    per_axis_origins = [
        _axis_origins(size, extent, stride)
        for size, extent, stride in zip(swept_sizes, window, strides, strict=True)
    ]
    middle = tuple(slice(None) for _ in range(len(leaf_shape) - 1 - n_swept))
    slices: list[tuple[slice, ...]] = []

    def _build(axis: int, prefix: tuple[slice, ...]) -> None:
        """Recurse over swept axes accumulating slice tuples."""

        if axis == n_swept:
            slices.append(middle + prefix)
            return
        for origin in per_axis_origins[axis]:
            extent = min(window[axis], swept_sizes[axis] - origin)
            _build(axis + 1, prefix + (slice(origin, origin + extent),))

    _build(0, ())
    return slices, [len(origins) for origins in per_axis_origins]


def _resolve_replacement(
    baseline_value: Any,
    leaf: Tensor,
) -> tuple[Callable[[Tensor, tuple[slice, ...]], Tensor], str]:
    """Resolve the replacement policy for occluded regions (the policy seam).

    Parameters
    ----------
    baseline_value
        ``"zeros"`` (default), ``"mean"`` (the leaf's global mean, disclosed),
        a scalar, or a tensor broadcastable into the occluded region.
    leaf
        Occluded leaf.

    Returns
    -------
    tuple[Callable[[Tensor, tuple[slice, ...]], Tensor], str]
        A function producing the occluded copy for one window slice, and the
        disclosed policy name.

    Raises
    ------
    AttributionError
        If the policy token is unknown.
    """

    if baseline_value == "zeros" or baseline_value is None:
        fill: Any = 0.0
        policy = "zeros"
    elif baseline_value == "mean":
        fill = leaf.detach().mean()
        policy = "mean(global leaf mean)"
    elif isinstance(baseline_value, (int, float)) and not isinstance(baseline_value, bool):
        fill = float(baseline_value)
        policy = f"constant({baseline_value})"
    elif isinstance(baseline_value, Tensor):
        fill = baseline_value.detach()
        policy = "tensor"
    else:
        raise AttributionError(
            "baseline_value must be 'zeros', 'mean', a scalar, or a "
            "broadcastable tensor. Remedy: choose one of those replacement "
            "policies.",
            code="occlusion_geometry_invalid",
        )

    def _occlude(source: Tensor, region: tuple[slice, ...]) -> Tensor:
        """Return a copy of ``source`` with ``region`` replaced (all batch rows)."""

        occluded = source.detach().clone()
        full_region = (slice(None), *region)
        occluded[full_region] = fill
        return occluded

    return _occlude, policy


def occlusion_map(
    model: Module,
    inputs: Any,
    input_kwargs: InputKwargs = None,
    *,
    target: _AttributionTarget,
    window: int | Sequence[int],
    strides: int | Sequence[int] | None = None,
    baseline_value: Any = "zeros",
    overlap: Literal["average", "sum"] = "average",
    occlude_leaf: int = 0,
    max_passes: int = _DIRECT_MAX_PASSES,
) -> AttributionResult:
    """Sweep an occlusion window over one input leaf and map the score deltas.

    The direct engine: eval/no-grad forwards, one per window, on copies of the
    occluded leaf with every other input held fixed. Each element's value is
    the ``overlap``-combined ``score(original) - score(occluded)`` over the
    windows covering it.

    Parameters
    ----------
    model
        PyTorch module to attribute.
    inputs
        Bare tensor or tuple/list of positional model arguments.
    input_kwargs
        Optional keyword arguments for ``model``.
    target
        Integer class index (per-example deltas) or callable
        ``output -> scalar tensor`` (batch-aggregate deltas, disclosed).
    window
        Window extents over the trailing swept axes of the occluded leaf
        (axes between the batch axis and the swept tail are fully occluded by
        every window -- a ``(15, 15)`` window on an image occludes all
        channels of each patch).
    strides
        Strides over the same axes; defaults to the window (non-overlapping).
    baseline_value
        Replacement policy: ``"zeros"`` (default), ``"mean"``, a scalar, or a
        broadcastable tensor.
    overlap
        ``"average"`` (default; per-element mean over covering windows) or
        ``"sum"``.
    occlude_leaf
        Traversal index of the attributed leaf to occlude; every other input
        is held fixed.
    max_passes
        Deterministic pass budget (default 4096). Exceeding it refuses typed
        AFTER measuring one pass, with the full arithmetic in the message and
        fields.

    Returns
    -------
    AttributionResult
        The occlusion map (same shape as the occluded leaf for int targets;
        leading batch axis of 1 for callable targets) plus the geometry,
        exact perturbation count, and cost ledger.

    Raises
    ------
    AttributionError
        On geometry violations or a pass budget overrun.
    """

    if overlap not in ("average", "sum"):
        raise AttributionError(
            f"overlap must be 'average' or 'sum'; got {overlap!r}. "
            "Remedy: choose one of the two documented combination modes.",
            code="occlusion_geometry_invalid",
        )
    prepared = _normalize_model_inputs(inputs, input_kwargs)
    if not 0 <= occlude_leaf < len(prepared.attributed_leaves):
        raise AttributionError(
            f"occlude_leaf={occlude_leaf} is out of range for "
            f"{len(prepared.attributed_leaves)} attributed leaves. Remedy: "
            "pass the traversal index of the leaf to occlude.",
            code="occlusion_geometry_invalid",
        )
    leaf = prepared.attributed_leaves[occlude_leaf]
    if leaf.ndim < 2:
        raise AttributionError(
            "occlusion_map requires the occluded leaf to carry a leading "
            "batch axis plus at least one sweepable axis. Remedy: reshape "
            "the input or use the scalar occlusion() on a trace selection.",
            code="occlusion_geometry_invalid",
        )
    window_extents = _normalize_window("window", window, None, None)
    stride_extents = _normalize_window("strides", strides, window_extents, len(window_extents))
    slices, windows_per_axis = _window_slices(tuple(leaf.shape), window_extents, stride_extents)
    n_passes = len(slices)
    occlude, policy = _resolve_replacement(baseline_value, leaf)

    def _score(candidate_leaf: Tensor) -> Tensor:
        """Forward once with ``candidate_leaf`` substituted; return raw target values."""

        replacements = tuple(
            candidate_leaf if index == occlude_leaf else original
            for index, original in enumerate(prepared.attributed_leaves)
        )
        output = _call_model(model, prepared, replacements)
        if isinstance(target, int):
            if not isinstance(output, Tensor):
                raise AttributionError(
                    "int targets require tensor model outputs. Remedy: use a callable target.",
                    code="attribution_target_invalid",
                )
            return output[..., target].detach()
        return _scalarize_output(output, target).detach()

    with _temporarily_eval(model), torch.no_grad():
        start = time.perf_counter()
        original_scores = _score(leaf)
        one_pass_seconds = time.perf_counter() - start
        if n_passes > max_passes:
            projected = n_passes * one_pass_seconds
            raise AttributionError(
                f"occlusion_map computed {n_passes} window passes, exceeding "
                f"max_passes={max_passes} (direct-engine default "
                f"{_DIRECT_MAX_PASSES}; trace-engine tier {_TRACE_MAX_PASSES}). "
                f"One measured pass took {one_pass_seconds:.4f}s, projecting "
                f"~{projected:.1f}s total. Remedy: enlarge the stride, shrink "
                "the swept region, or raise max_passes explicitly.",
                code="occlusion_pass_budget_exceeded",
                computed_passes=n_passes,
                max_passes=max_passes,
                one_pass_seconds=one_pass_seconds,
                projected_total_seconds=projected,
            )
        if isinstance(target, int):
            delta_shape = tuple(leaf.shape)
            per_example = True
        else:
            delta_shape = (1, *leaf.shape[1:])
            per_example = False
        accumulator = torch.zeros(delta_shape, dtype=original_scores.dtype, device=leaf.device)
        coverage = torch.zeros(delta_shape, dtype=original_scores.dtype, device=leaf.device)
        for region in slices:
            occluded_scores = _score(occlude(leaf, region))
            delta = original_scores - occluded_scores
            full_region = (slice(None), *region)
            if per_example:
                reshaped = delta.reshape((-1,) + (1,) * (leaf.ndim - 1))
                accumulator[full_region] += reshaped
            else:
                accumulator[full_region] += delta
            coverage[full_region] += 1
        total_seconds = time.perf_counter() - start
    values = accumulator / coverage.clamp(min=1) if overlap == "average" else accumulator
    return AttributionResult(
        method="occlusion_map",
        values=values,
        target_repr=_target_repr(target),
        extra={
            "engine": "direct",
            "window": list(window_extents),
            "strides": list(stride_extents),
            "windows_per_axis": windows_per_axis,
            "coverage_policy": "clipped_edge_full_coverage",
            "replacement_policy": policy,
            "overlap": overlap,
            "occlude_leaf": occlude_leaf,
            "perturbation_count": n_passes,
            "physical_forward_calls": n_passes + 1,
            "per_example_deltas": per_example,
            "render_hint": "zero_centered_diverging",
            "cost_ledger": {
                "one_pass_seconds": one_pass_seconds,
                "total_seconds": total_seconds,
                "max_passes": max_passes,
            },
        },
    )


__all__ = ["occlusion", "occlusion_map"]
