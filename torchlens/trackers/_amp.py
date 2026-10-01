"""AMP grad-scale observation and the closed-form correction (memo 3.11).

The tracker NEVER calls ``GradScaler.unscale_`` -- it mutates ``p.grad`` in
place and changes what the user's own clipping sees. Instead the scale is
OBSERVED at the optimizer-step boundary (between ``backward()`` and
``scaler.update()``; ``update()`` halves the scale on an overflow step, so a
late read is wrong by exactly 2x on precisely the steps a user is
investigating), and any statistic reduced from a still-scaled tensor is
corrected in closed form on the tracker's OWN record: counts and nonfinite
counts are scale-invariant; first-power fields divide by the observed scale
``s``; second-power fields divide by ``s**2``. GradScaler scales are powers
of two, so the correction is BIT-exact against the ``unscale_`` oracle (the
tests assert equality, never allclose).

Evidence labels never default: ``unknown`` never masquerades as a factor of
one (C06 D7). The vocabulary here is the trackers-memo spelling; the C06
record stamp (``scaled | unscaled | unknown``) is derived from it.

Seam note (F23): the checks memo homes the one shared scale read in the
checks package; F23's shipped surface carries a ``ScaleLedger`` but not the
``observed_grad_scale()`` primitive, so this module owns the read for now.
When checks grows the shared primitive, this becomes a delegating consumer
(two implementations would diverge and one would be wrong).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from ..observability import CommittedBlock, HistogramResult, SpineResult
from ._errors import TrackersError

__tl_layer__ = "L8"

#: Scale-evidence vocabulary (memo 3.11, Sol's field). ``unscaled_observed``
#: = the value was reduced after the scaler's own unscale (the optimizer
#: pre-step site); ``unscaled_derived`` = the tracker applied the closed-form
#: correction; ``scaled_unknown_factor`` = known scaled, factor unobserved;
#: ``unavailable`` = no scaler evidence at all.
SCALE_EVIDENCE = (
    "unscaled_observed",
    "unscaled_derived",
    "scaled_unknown_factor",
    "unavailable",
)


def observed_grad_scale(source: Any) -> tuple[float | None, str]:
    """Read the live gradient scale off a ``GradScaler``-like object.

    Parameters
    ----------
    source:
        A ``torch.amp.GradScaler`` (or duck-typed equivalent exposing
        ``is_enabled()`` / ``get_scale()``), or ``None``.

    Returns
    -------
    tuple[float | None, str]
        ``(scale, evidence)``. A live enabled scaler returns its scale with
        evidence ``"scaled_unknown_factor"`` REPLACED by the call-site's
        knowledge -- this function only attests the read; the caller decides
        whether the tensors it reduced were pre- or post-unscale. ``None``
        source returns ``(None, "unavailable")``; a disabled scaler returns
        ``(1.0, "unscaled_observed")`` (scaling genuinely off is an observed
        fact, not an assumption).
    """

    if source is None:
        return (None, "unavailable")
    is_enabled = getattr(source, "is_enabled", None)
    get_scale = getattr(source, "get_scale", None)
    if not callable(is_enabled) or not callable(get_scale):
        return (None, "unavailable")
    if not is_enabled():
        return (1.0, "unscaled_observed")
    return (float(get_scale()), "scaled_unknown_factor")


#: ``GradScaler`` per-optimizer stage names -> whether ``p.grad`` is unscaled
#: at the optimizer-step boundary. READY = the scaler has not unscaled this
#: optimizer since the last ``update()`` (grads still carry the factor);
#: UNSCALED / STEPPED = ``unscale_`` ran (in place) before the boundary.
_STAGE_TO_TRUTH = {"READY": "scaled", "UNSCALED": "unscaled", "STEPPED": "unscaled"}


def unscale_stage(scaler: Any, optimizer: Any) -> str:
    """Whether ``optimizer``'s gradients are unscaled at the step boundary.

    Reads the scaler's per-optimizer stage record (``_per_optimizer_states``,
    a ``GradScaler`` implementation detail that has carried the READY /
    UNSCALED / STEPPED stage since torch 1.6; duck-typed here, degrading to
    ``"unknown"`` -- never to a guess -- when the record is absent or has an
    unrecognized shape). ``scaler.step(optimizer)`` unscales before it calls
    ``optimizer.step()``; a plain ``optimizer.step()`` after
    ``scaler.scale(loss).backward()`` does NOT, and the boundary sees grads
    still multiplied by the scale. Before this read the engine asserted
    ``unscaled="yes"`` whenever ``get_scale()`` merely worked (AUD-CODE 2.14b).

    Returns
    -------
    str
        ``"unscaled"`` | ``"scaled"`` | ``"unknown"``.
    """

    if scaler is None or optimizer is None:
        return "unknown"
    states = getattr(scaler, "_per_optimizer_states", None)
    if not isinstance(states, Mapping):
        return "unknown"
    # ``.get`` never materializes a defaultdict entry: a scaler that has not
    # touched this optimizer since ``update()`` holds no record, which is the
    # READY state by construction (nothing has unscaled these grads).
    state = states.get(id(optimizer))
    if state is None:
        return "scaled"
    if not isinstance(state, Mapping):
        return "unknown"
    stage = state.get("stage")
    name = getattr(stage, "name", stage)
    return _STAGE_TO_TRUTH.get(str(name), "unknown") if name is not None else "unknown"


def corrected_gradient_block(block: CommittedBlock, scale: float) -> tuple[CommittedBlock, int]:
    """Return ``block`` with every SCALED gradient observation corrected.

    Observations stamped ``grad_scale="scaled"`` on gradient streams get the
    closed-form correction (:func:`correct_spine` / :func:`correct_histogram`)
    and are re-stamped ``"unscaled"``; everything else is returned as is.
    The second element counts the corrected observations so the emitter can
    disclose the derivation (evidence ``unscaled_derived``) on the run-health
    series instead of presenting corrected numbers as observed ones.
    """

    corrected = 0
    observations = []
    for observation in block.observations:
        if (
            observation.stream in ("param_grad", "activation_grad")
            and observation.grad_scale == "scaled"
            and observation.spine is not None
        ):
            observation = replace(
                observation,
                spine=correct_spine(observation.spine, scale),
                sketch=(
                    correct_histogram(observation.sketch, scale)
                    if observation.sketch is not None
                    else None
                ),
                grad_scale="unscaled",
            )
            corrected += 1
        observations.append(observation)
    if not corrected:
        return (block, 0)
    return (replace(block, observations=tuple(observations)), corrected)


def _divide(value: float | None, factor: float) -> float | None:
    """Divide one nullable floating field by ``factor``."""

    return None if value is None else value / factor


def correct_spine(spine: SpineResult, scale: float) -> SpineResult:
    """Apply the closed-form unscale correction to one spine record.

    Counts (total/finite/zero/negative/nonfinite) are invariant under a
    positive rescale; ``min``/``max``/``absmax``/``sum``/``sum_abs``/``mean``
    divide by ``scale``; ``sum_squares`` and ``m2`` divide by ``scale**2``.
    Exact for power-of-two scales (each division adjusts only the exponent).
    """

    if not math.isfinite(scale) or scale <= 0.0:
        raise TrackersError(
            f"Cannot correct statistics by grad scale {scale!r}: the observed "
            "scale must be a finite positive factor. A nonfinite scale means "
            "the scaler itself is in a state the tracker must disclose, never "
            "silently divide by.",
            code="tracker_scale_invalid",
            scale=scale,
            remedy=(
                "Pass the scale read from GradScaler.get_scale() at the "
                "optimizer boundary, or emit the record with evidence "
                "'scaled_unknown_factor' instead of correcting it."
            ),
        )
    square = scale * scale
    return replace(
        spine,
        finite_min=_divide(spine.finite_min, scale),
        finite_max=_divide(spine.finite_max, scale),
        finite_absmax=_divide(spine.finite_absmax, scale),
        sum=_divide(spine.sum, scale),
        sum_squares=_divide(spine.sum_squares, square),
        sum_abs=_divide(spine.sum_abs, scale),
        mean=_divide(spine.mean, scale),
        m2=_divide(spine.m2, square),
    )


def correct_histogram(sketch: HistogramResult, scale: float) -> HistogramResult:
    """Shift a signed-log2 histogram by the exact bin offset of ``scale``.

    On the C06 grid, dividing every population element by a power-of-two
    scale ``s`` moves each magnitude bin DOWN by exactly
    ``bins_per_octave * log2(s)`` bins -- an integer for power-of-two scales,
    so the correction is an exact count shift, never a rebinning. Counts
    shifted below the grid fold into the per-side underflow specials (their
    corrected magnitudes genuinely lie below ``2**lo_exp``); zero / NaN /
    +-inf specials are invariant. Non-power-of-two scales refuse: they have
    no exact shift and rebinning is banned (C06 D9).
    """

    if not math.isfinite(scale) or scale <= 0.0 or scale != 2.0 ** round(math.log2(scale)):
        raise TrackersError(
            f"Cannot shift a log2 histogram by grad scale {scale!r}: only "
            "positive power-of-two scales have an exact integer bin shift, "
            "and rebinning is banned (C06 D9). GradScaler scales are powers "
            "of two by construction; anything else here is evidence of a "
            "foreign scaling source.",
            code="tracker_scale_invalid",
            scale=scale,
            remedy=(
                "Correct only GradScaler-observed power-of-two scales, or "
                "emit the record scaled with evidence 'scaled_unknown_factor'."
            ),
        )
    shift = sketch.descriptor.bins_per_octave * round(math.log2(scale))
    if shift == 0:
        return sketch

    def shifted(counts: tuple[int, ...]) -> tuple[tuple[int, ...], int, int]:
        """Shift one side's counts; return (counts, underflow_add, overflow_add)."""

        n = len(counts)
        moved = [0] * n
        under = over = 0
        for index, count in enumerate(counts):
            target = index - shift
            if target < 0:
                under += count
            elif target >= n:
                over += count
            else:
                moved[target] = count
        return (tuple(moved), under, over)

    pos, pos_under, pos_over = shifted(sketch.pos_counts)
    neg, neg_under, neg_over = shifted(sketch.neg_counts)
    specials = dict(sketch.specials)
    for key, add in (
        ("pos_underflow", pos_under),
        ("pos_overflow", pos_over),
        ("neg_underflow", neg_under),
        ("neg_overflow", neg_over),
    ):
        if add:
            specials[key] = specials.get(key, 0) + add
    return HistogramResult(
        descriptor=sketch.descriptor,
        pos_counts=pos,
        neg_counts=neg,
        specials=specials,
    )


__all__ = [
    "SCALE_EVIDENCE",
    "correct_histogram",
    "correct_spine",
    "corrected_gradient_block",
    "observed_grad_scale",
    "unscale_stage",
]
