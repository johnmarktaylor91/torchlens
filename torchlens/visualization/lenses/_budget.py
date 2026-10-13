"""The visible-detail budget resolver (knob N10, themes memo section 3).

For any lens row that compacts, search the SHIPPED dials -- the float
collapse schedule below the optimizer ceiling, call-depth truncation above
it -- for the coarsest setting whose visible-unit count lands INSIDE the
band. The band is TWO-SIDED: floor 100, target 160, ceiling 220 (memo
section 13 item 7); a trace with fewer than 100 total visible units simply
renders in full. When a colour channel is active, the resolver additionally
requires coverage >= the floor (candidate 0.80). If ``collapse="none"``
already fits, it is used. The resolved dial is disclosed BY NAME plus the
coverage line, always.

Every spelling here is DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ...errors._base import TorchLensWarning

if TYPE_CHECKING:
    from ...data_classes.trace import Trace

__all__ = [
    "BAND_CEILING",
    "BAND_FLOOR",
    "BAND_TARGET",
    "COVERAGE_FLOOR",
    "BudgetResolution",
    "resolve_budget",
]

#: The two-sided readability band (memo s13 item 7: the review's spec, adopted).
BAND_FLOOR = 100
BAND_TARGET = 160
BAND_CEILING = 220

#: Coverage floor applied when a colour channel is active (candidate value;
#: R2 tunes it from rendered candidates).
COVERAGE_FLOOR = 0.80


@dataclass(frozen=True)
class BudgetResolution:
    """One resolved detail dial, disclosed by name.

    Attributes
    ----------
    dial:
        The dial BY NAME, e.g. ``collapse="none"``, ``collapse=0.75 (float
        schedule)``, ``vis_call_depth=3 (above-ceiling estimate)``.
    draw_kwargs:
        The draw parameters that apply the dial.
    visible_count:
        Visible-unit count under the dial (exact below the optimizer
        ceiling; a disclosed estimate above it).
    coverage:
        Encoded-visible fraction under the dial when channel values were
        supplied, else ``None``.
    in_band:
        Whether ``visible_count`` landed inside the two-sided band (or the
        render-in-full carve-out applied).
    disclosure:
        Mandatory rendered disclosure lines (dial name, counts, coverage).
    """

    dial: str
    draw_kwargs: dict[str, Any]
    visible_count: int
    coverage: float | None
    in_band: bool
    disclosure: tuple[str, ...]


def _coverage_for_step(
    values: dict[str, float],
    op_modules: dict[str, tuple[str, ...]],
    collapsed: frozenset[str],
    visible_count: int,
) -> float:
    """Return encoded-visible coverage for one schedule step.

    An op is hidden when its pass-qualified label is collapsed or any of its
    containing module addresses is collapsed; coverage is encoded visible
    ops over ALL visible units (module boxes and folds count as visible but
    unencoded -- the measured 50%-coverage defect this floor exists to
    bound).
    """

    if visible_count <= 0:
        return 0.0
    encoded_visible = 0
    for label, _ in values.items():
        if label in collapsed:
            continue
        if any(address in collapsed for address in op_modules.get(label, ())):
            continue
        encoded_visible += 1
    return encoded_visible / visible_count


def _call_depth_estimate(trace: Trace, depth: int) -> int:
    """Estimate visible units at ``vis_call_depth=depth`` without a render.

    Counts ops whose module call nesting is within ``depth`` plus one box
    per module at the cut depth that owns deeper content. A disclosed
    ESTIMATE (the above-ceiling path's per-subtree refinement is a named
    wave-2 item), deterministic for the same trace and depth.
    """

    visible_ops = 0
    cut_boxes: set[str] = set()
    for op in trace.ops:
        stack = tuple(getattr(op, "module_call_stack", ()) or ())
        if len(stack) <= depth:
            visible_ops += 1
        else:
            cut_boxes.add(stack[depth - 1] if depth >= 1 else stack[0])
    return visible_ops + len(cut_boxes)


def _op_value_maps(
    trace: Trace, values: dict[str, float] | None
) -> tuple[dict[str, float], dict[str, tuple[str, ...]]]:
    """Index channel values and containing-module addresses by op label."""

    if not values:
        return {}, {}
    op_modules: dict[str, tuple[str, ...]] = {}
    for op in trace.ops:
        label = getattr(op, "label", None) or op.layer_label
        if label in values:
            op_modules[label] = tuple(getattr(op, "modules", ()) or ())
    return values, op_modules


def resolve_budget(
    trace: Trace,
    *,
    values: dict[str, float] | None = None,
    context: Any = None,
) -> BudgetResolution:
    """Resolve the coarsest shipped dial landing inside the band.

    Parameters
    ----------
    trace:
        The trace to be rendered.
    values:
        Active colour-channel values keyed by pass-qualified op label; when
        supplied the coverage floor is enforced alongside the band.
    context:
        Optional render context forwarded to ``Trace.collapse_schedule``.

    Returns
    -------
    BudgetResolution
        The dial, its draw kwargs, and the mandatory disclosure lines. When
        no dial can satisfy every constraint the nearest candidate is
        returned with ``in_band=False`` and a coded warning
        (``lens_budget_band_missed``) -- disclosed, never silent.
    """

    from ..collapse_optimizer import COLLAPSE_OPTIMIZER_MAX_OPS

    channel_values, op_modules = _op_value_maps(trace, values)
    channel_active = bool(channel_values)
    op_count = len(trace.ops)

    if op_count > COLLAPSE_OPTIMIZER_MAX_OPS:
        return _resolve_above_ceiling(trace, op_count, channel_active)

    schedule = trace.collapse_schedule(context)
    full_visible = schedule.steps[0].visible_count
    if full_visible <= BAND_CEILING:
        coverage = (
            _coverage_for_step(channel_values, op_modules, frozenset(), full_visible)
            if channel_active
            else None
        )
        lines = [
            f'detail dial: collapse="none" ({full_visible} visible units; '
            f"band {BAND_FLOOR}-{BAND_CEILING})"
        ]
        if coverage is not None:
            lines.append(
                f"coverage: encoded {int(round(coverage * full_visible))} of {full_visible}"
            )
        return BudgetResolution(
            dial='collapse="none"',
            draw_kwargs={"collapse": "none"},
            visible_count=full_visible,
            coverage=coverage,
            in_band=True,
            disclosure=tuple(lines),
        )

    candidates = []
    for step in schedule.steps:
        coverage = (
            _coverage_for_step(
                channel_values, op_modules, step.collapsed_addresses, step.visible_count
            )
            if channel_active
            else None
        )
        candidates.append((step, coverage))

    in_band = [
        (step, coverage)
        for step, coverage in candidates
        if BAND_FLOOR <= step.visible_count <= BAND_CEILING
        and (coverage is None or coverage >= COVERAGE_FLOOR)
    ]
    if in_band:
        step, coverage = min(in_band, key=lambda pair: pair[0].visible_count)
        return _float_resolution(step, coverage, in_band=True)

    # No point satisfies every constraint: serve the nearest point, disclosed.
    def _distance(pair: Any) -> tuple[float, float]:
        """Rank a (step, coverage) pair by band distance, then target distance."""

        step, coverage = pair
        count = step.visible_count
        band_distance = float(
            0
            if BAND_FLOOR <= count <= BAND_CEILING
            else min(abs(count - BAND_FLOOR), abs(count - BAND_CEILING))
        )
        coverage_gap = 0.0 if coverage is None else max(0.0, COVERAGE_FLOOR - coverage)
        return (band_distance + coverage_gap * BAND_TARGET, float(count))

    # The dial must genuinely compact: the full graph (t=0.0, already known to
    # sit above the ceiling here) is never the served "nearest" point while the
    # schedule offers ANY compacting stop, even when it is arithmetically
    # closer to the band (a 282-op graph with a two-point schedule 282 -> 4
    # served the 282-op wall as nearest: |282-220| < |100-4|; AUD-CODE 3.9).
    compacting = [pair for pair in candidates if pair[0].visible_count < full_visible]
    step, coverage = min(compacting or candidates, key=_distance)
    warnings.warn(
        TorchLensWarning(
            "the visible-detail budget resolver found no float-schedule point "
            f"inside the {BAND_FLOOR}-{BAND_CEILING} band"
            + (" at the coverage floor" if channel_active else "")
            + f"; serving the nearest compacting point ({step.visible_count} "
            f"visible at collapse={step.t:g}, full graph {full_visible}). "
            "Remedy: focus the render with module= or vis_call_depth=, or pass "
            "an explicit collapse= override.",
            code="lens_budget_band_missed",
        ),
        stacklevel=3,
    )
    return _float_resolution(step, coverage, in_band=False)


def _float_resolution(step: Any, coverage: float | None, *, in_band: bool) -> BudgetResolution:
    """Build the resolution record for one float-schedule step."""

    lines = [
        f"detail dial: collapse={step.t:g} (float schedule; {step.visible_count} "
        f"visible units; band {BAND_FLOOR}-{BAND_CEILING})"
    ]
    if coverage is not None:
        lines.append(
            f"coverage: encoded {int(round(coverage * step.visible_count))} of "
            f"{step.visible_count} visible units"
        )
    if not in_band:
        lines.append("band missed: nearest schedule point served (see warning)")
    return BudgetResolution(
        dial=f"collapse={step.t:g} (float schedule)",
        draw_kwargs={"collapse": step.t},
        visible_count=step.visible_count,
        coverage=coverage,
        in_band=in_band,
        disclosure=tuple(lines),
    )


def _resolve_above_ceiling(trace: Trace, op_count: int, channel_active: bool) -> BudgetResolution:
    """Resolve the call-depth dial above the optimizer ceiling.

    Above ``COLLAPSE_OPTIMIZER_MAX_OPS`` no collapse spelling compacts (the
    measured above-ceiling no-op, memo build item 1); call-depth truncation
    is the shipped dial that still works. Counts are disclosed ESTIMATES.
    """

    from ..collapse_optimizer import COLLAPSE_OPTIMIZER_MAX_OPS

    best_depth = 1
    best_estimate = _call_depth_estimate(trace, 1)
    chosen: tuple[int, int] | None = None
    max_depth = max(
        (len(tuple(getattr(op, "module_call_stack", ()) or ())) for op in trace.ops),
        default=1,
    )
    for depth in range(1, max(max_depth, 1) + 1):
        estimate = _call_depth_estimate(trace, depth)
        if BAND_FLOOR <= estimate <= BAND_CEILING:
            chosen = (depth, estimate)
            break
        if abs(estimate - BAND_TARGET) < abs(best_estimate - BAND_TARGET):
            best_depth, best_estimate = depth, estimate
    if chosen is None:
        warnings.warn(
            TorchLensWarning(
                f"above the {COLLAPSE_OPTIMIZER_MAX_OPS}-op optimizer ceiling no "
                f"call depth lands in the {BAND_FLOOR}-{BAND_CEILING} band; serving "
                f"vis_call_depth={best_depth} (~{best_estimate} visible units, "
                "estimate). Remedy: focus the render with module= or pass an "
                "explicit vis_call_depth=.",
                code="lens_budget_above_ceiling_fallback",
            ),
            stacklevel=3,
        )
        depth, estimate = best_depth, best_estimate
        in_band = False
    else:
        depth, estimate = chosen
        in_band = True
    lines = [
        f"detail dial: vis_call_depth={depth} (above the "
        f"{COLLAPSE_OPTIMIZER_MAX_OPS}-op optimizer ceiling on {op_count} ops; "
        f"~{estimate} visible units, estimate)"
    ]
    if channel_active:
        lines.append("coverage floor: not measurable above the ceiling (estimate basis disclosed)")
    return BudgetResolution(
        dial=f"vis_call_depth={depth} (above-ceiling estimate)",
        draw_kwargs={"collapse": "none", "vis_call_depth": depth},
        visible_count=estimate,
        coverage=None,
        in_band=in_band,
        disclosure=tuple(lines),
    )
