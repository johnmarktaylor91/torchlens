"""Derived quantiles from the signed-log2 sketch (explorer D10; lane F25).

The DEFAULT quantile mode: interpolated from the full-population sketch,
always marked approximate, with the grid resolution and overflow state
disclosed on the result. Exact and sampled modes are explicit collector-side
choices and never silently substitute for this one.

Population and ordering: the quantile population is every FINITE element
(grid bins + zero + per-side underflow/overflow specials); NaN never enters
a quantile and +/-inf are excluded with the overflow state disclosing their
presence. On the real line the bands order as::

    neg_overflow < neg bins (desc magnitude) < neg_underflow < zero
        < pos_underflow < pos bins (asc magnitude) < pos_overflow

Within a magnitude bin the estimate interpolates log-linearly between the
bin edges; underflow bands interpolate linearly between 0 and the window
floor; an overflow band CLAMPS to the window edge and sets
``clamped_to_window=True`` -- the exact out-of-range count survives the
window choice (D9), so the clamp is disclosed, never silent.

Spellings are DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

from dataclasses import dataclass

from ._errors import ObservabilityError
from ._kernels import HistogramResult

__tl_layer__ = "L5"


class WatchRenderError(ObservabilityError):
    """Raised for watch renderer / derived-read refusals (lane F25)."""


@dataclass(frozen=True)
class DerivedQuantile:
    """One sketch-derived quantile estimate with its honesty disclosures.

    ``approximate`` is always True (this is the derived mode);
    ``grid_resolution`` names the sketch geometry the error bound comes
    from; ``clamped_to_window`` is True when the true value sits in an
    overflow band beyond the grid window.
    """

    q: float
    value: float
    approximate: bool
    grid_resolution: str
    clamped_to_window: bool = False


@dataclass(frozen=True)
class _Band:
    """One ordered real-line band: [lo, hi) with a count and a scale."""

    lo: float
    hi: float
    count: int
    log_scale: bool
    clamped: bool = False


def _signed_bands(hist: HistogramResult) -> list[_Band]:
    """Order the sketch's bins and finite specials along the real line."""

    desc = hist.descriptor
    edges = desc.bucket_edges()
    specials = hist.specials
    floor = edges[0]
    ceil = edges[-1]
    bands: list[_Band] = []
    bands.append(_Band(-ceil, -ceil, int(specials.get("neg_overflow", 0)), False, clamped=True))
    for i in range(len(hist.neg_counts) - 1, -1, -1):
        bands.append(_Band(-edges[i + 1], -edges[i], int(hist.neg_counts[i]), True))
    bands.append(_Band(-floor, 0.0, int(specials.get("neg_underflow", 0)), False))
    bands.append(_Band(0.0, 0.0, int(specials.get("zero", 0)), False))
    bands.append(_Band(0.0, floor, int(specials.get("pos_underflow", 0)), False))
    for i in range(len(hist.pos_counts)):
        bands.append(_Band(edges[i], edges[i + 1], int(hist.pos_counts[i]), True))
    bands.append(_Band(ceil, ceil, int(specials.get("pos_overflow", 0)), False, clamped=True))
    return bands


def _interpolate(band: _Band, fraction: float) -> float:
    """Position ``fraction`` of the way through one band's value range."""

    if band.lo == band.hi:
        return band.lo
    if not band.log_scale:
        return band.lo + fraction * (band.hi - band.lo)
    import math

    if band.lo > 0:
        lo_log, hi_log = math.log2(band.lo), math.log2(band.hi)
        return 2.0 ** (lo_log + fraction * (hi_log - lo_log))
    lo_log, hi_log = math.log2(-band.hi), math.log2(-band.lo)
    return -(2.0 ** (hi_log - fraction * (hi_log - lo_log)))


def derived_quantile(hist: HistogramResult, q: float) -> DerivedQuantile:
    """Estimate the ``q``-quantile of the sketched population (D10 default).

    Parameters
    ----------
    hist:
        Finalized signed-log2 histogram (grid counts + exact specials).
    q:
        Quantile in ``[0, 1]``.

    Returns
    -------
    DerivedQuantile
        The interpolated estimate, always marked approximate, with the
        grid resolution and any window clamp disclosed.

    Raises
    ------
    WatchRenderError
        ``watch_quantile_invalid`` for q outside [0, 1];
        ``watch_population_empty`` when no finite element was sketched.
    """

    if not 0.0 <= q <= 1.0:
        raise WatchRenderError(
            f"q={q} is not a quantile; quantiles live in [0, 1].",
            code="watch_quantile_invalid",
            q=q,
            remedy="Pass q in [0, 1].",
        )
    bands = _signed_bands(hist)
    total = sum(band.count for band in bands)
    desc = hist.descriptor
    resolution = (
        f"signed log2 grid, bpo={desc.bins_per_octave}, window [2^{desc.lo_exp}, 2^{desc.hi_exp}]"
    )
    if total == 0:
        raise WatchRenderError(
            "The sketched population has no finite elements; a quantile of "
            "nothing would invent a number (missing is never zero, D6).",
            code="watch_population_empty",
            remedy="Check presence/nonfinite disclosures before deriving quantiles.",
        )
    target = q * (total - 1)
    cumulative = 0
    for band in bands:
        if band.count == 0:
            continue
        band_end = cumulative + band.count
        if target < band_end or band is bands[-1]:
            fraction = 0.5 if band.count == 0 else (target - cumulative + 0.5) / band.count
            fraction = min(max(fraction, 0.0), 1.0)
            return DerivedQuantile(
                q=q,
                value=_interpolate(band, fraction),
                approximate=True,
                grid_resolution=resolution,
                clamped_to_window=band.clamped,
            )
        cumulative = band_end
    raise AssertionError("unreachable: nonzero total with no owning band")


def derived_quantiles(hist: HistogramResult, qs: tuple[float, ...]) -> tuple[DerivedQuantile, ...]:
    """Vector spelling of :func:`derived_quantile` for one sketch."""

    return tuple(derived_quantile(hist, q) for q in qs)


__all__ = [
    "DerivedQuantile",
    "WatchRenderError",
    "derived_quantile",
    "derived_quantiles",
]
