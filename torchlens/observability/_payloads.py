"""Sink-neutral payload conversion for watch observations (D24; lane F25).

The explorer owns the record and the CONVERSION; tracker sinks own the
native emission call (D26). The completed dashboard histogram payload is
the spine's five verbatim ``add_histogram_raw`` fields plus the canonical
``bucket_limits`` / ``bucket_counts`` derived from the immutable signed
log2 grid (review r3's correction to D8's "zero translation" shorthand).

Nonfinite counts (NaN, +/-inf) are NOT representable in a dashboard
bucket axis: they are excluded from the buckets, ``num`` counts the
FINITE population, and the exact excluded counts ride the payload's
``excluded_nonfinite`` key so a sink can disclose them next to the plot.

Spellings are DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

from typing import Any

from ._kernels import HistogramResult, SpineResult
from ._quantiles import WatchRenderError
from ._schema import ObservationRecord

__tl_layer__ = "L5"


def _require_observed_sketch(
    observation: ObservationRecord,
) -> tuple[SpineResult, HistogramResult]:
    """Refuse payload conversion without an observed spine + sketch.

    Returns the narrowed ``(spine, sketch)`` pair it just proved present.
    """

    if observation.presence != "observed" or observation.spine is None:
        raise WatchRenderError(
            f"Cannot build a histogram payload from presence="
            f"{observation.presence!r}: a non-observed observation has no "
            "numbers, and missing must never render as zero (D6).",
            code="watch_payload_not_observed",
            presence=observation.presence,
            remedy="Convert only observations with presence='observed'.",
        )
    if observation.sketch is None:
        raise WatchRenderError(
            "This observation carries no sketch (spine-only cadence tier); "
            "a histogram payload cannot be invented from scalar fields.",
            code="watch_sketch_missing",
            remedy=(
                "Lower the sketch cadence for this stream, or emit the "
                "spine-only scalar payload instead."
            ),
        )
    return observation.spine, observation.sketch


def histogram_payload(observation: ObservationRecord) -> dict[str, Any]:
    """Build the completed sink-neutral histogram payload for one observation.

    Returns a dict with TensorBoard ``add_histogram_raw``'s value fields --
    ``min, max, num, sum, sum_squares, bucket_limits, bucket_counts`` --
    plus the disclosure keys ``excluded_nonfinite`` (exact NaN/inf counts)
    and ``grid`` (the immutable descriptor tuple). The caller supplies tag
    and step; the sink owns the native call (D26).

    Buckets ascend the signed real line: one below-window bucket per side
    (the exact overflow forensics band), the mirrored magnitude bins, and
    one zero-spanning bucket holding the exact zero + underflow counts.

    Raises
    ------
    WatchRenderError
        ``watch_payload_not_observed`` / ``watch_sketch_missing`` when the
        observation cannot honestly supply a histogram.
    """

    spine, sketch = _require_observed_sketch(observation)
    edges = sketch.descriptor.bucket_edges()
    floor, ceil = edges[0], edges[-1]
    specials = sketch.specials
    limits: list[float] = []
    counts: list[int] = []
    # Below-window negative overflow: everything at or below -ceil.
    limits.append(-ceil)
    counts.append(int(specials.get("neg_overflow", 0)))
    for i in range(len(sketch.neg_counts) - 1, -1, -1):
        limits.append(-edges[i])
        counts.append(int(sketch.neg_counts[i]))
    # Zero-spanning bucket: exact zeros plus both underflow bands.
    limits.append(floor)
    counts.append(
        int(specials.get("neg_underflow", 0))
        + int(specials.get("zero", 0))
        + int(specials.get("pos_underflow", 0))
    )
    for i in range(len(sketch.pos_counts)):
        limits.append(edges[i + 1])
        counts.append(int(sketch.pos_counts[i]))
    # Above-window positive overflow; the edge extends to the observed max
    # when it lies beyond the window so the bucket is honestly bounded.
    finite_max = spine.finite_max if spine.finite_max is not None else ceil
    limits.append(max(float(finite_max), ceil))
    counts.append(int(specials.get("pos_overflow", 0)))
    descriptor = sketch.descriptor
    return {
        "min": spine.finite_min,
        "max": spine.finite_max,
        "num": spine.count_finite,
        "sum": spine.sum,
        "sum_squares": spine.sum_squares,
        "bucket_limits": limits,
        "bucket_counts": counts,
        "excluded_nonfinite": {
            "nan": spine.count_nan,
            "posinf": spine.count_posinf,
            "neginf": spine.count_neginf,
        },
        "grid": (
            descriptor.base,
            descriptor.bins_per_octave,
            descriptor.lo_exp,
            descriptor.hi_exp,
            descriptor.signed,
            descriptor.encoding,
        ),
    }


def observation_row(
    observation: ObservationRecord,
    *,
    step_lo: int | None = None,
    step_hi: int | None = None,
) -> dict[str, Any]:
    """Flatten one observation into a plain tabular row (long form).

    Non-observed presences yield a row whose payload columns are ``None``
    (never zero); the presence token itself is the datum.
    """

    spine = observation.spine
    row: dict[str, Any] = {
        "step": observation.global_step,
        "step_lo": observation.global_step if step_lo is None else step_lo,
        "step_hi": observation.global_step if step_hi is None else step_hi,
        "site_id": observation.site_id,
        "stream": observation.stream,
        "phase": observation.phase,
        "presence": observation.presence,
        "grad_scale": observation.grad_scale,
        "estimated": observation.estimated,
        "sample_size": observation.sample_size,
        "reason": observation.reason,
        "has_sketch": observation.sketch is not None,
    }
    spine_fields = (
        "count_total",
        "count_finite",
        "count_zero",
        "count_negative",
        "count_nan",
        "count_posinf",
        "count_neginf",
        "finite_min",
        "finite_max",
        "finite_absmax",
        "sum",
        "sum_squares",
        "sum_abs",
        "mean",
        "m2",
        "reduction_dtype",
    )
    for name in spine_fields:
        row[name] = getattr(spine, name) if spine is not None else None
    return row


__all__ = ["histogram_payload", "observation_row"]
