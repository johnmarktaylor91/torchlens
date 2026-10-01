"""Static watch renderers on the SVG/PIL chassis (explorer D23/D24; F25).

Four paper-ready views over the per-step history artifact, all
deterministic SVG from a small pure-python composer (base dependencies
only -- numpy and pillow, both already in the base set):

- ``render_contact_sheet``: ranked, hierarchically ordered site tiles with
  a NAMED ranking score, nonfinite events first, a mandatory "N of M"
  footer, and deterministic full pagination -- silent truncation is banned.
- ``render_fan``: the default per-site view -- sketch-derived median and
  p25-75 / p5-95 bands, exact min/max dotted from the spine, presence
  gaps, overflow bands, coarsening spans, and optimizer-skip marks drawn
  ON the figure, not footnoted.
- ``render_waterfall``: the torchexplorer lookalike minus its limits --
  step on x, signed log2 bins on y, log-count density as a PIL raster
  interior at a declared resolution, zero / nonfinite / out-of-range as
  separately visible annotated bands, variable-width columns where
  RAM-only coarsening happened, and a caption stating fixed bins.
- ``render_detail``: every stream of one site on one step axis with every
  disclosure (grad scale, estimation, sample sizes, presence).

The one genuinely raster object is the waterfall density field; axes,
labels, badges, and captions stay vector text (searchable, greppable).
The SVG root declares ``xmlns:xlink`` (cairosvg refuses the file
otherwise -- the round-3 gate finding).

Spellings are DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

import base64
import io
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from ._artifact import CommittedBlock, HistoryReader
from ._kernels import HistogramResult, SpineResult
from ._payloads import observation_row
from ._quantiles import WatchRenderError, derived_quantile
from ._schema import ObservationRecord, RunRecord, SiteRecord, StepBlockRecord

__tl_layer__ = "L5"

#: The one named v1 ranking score (D24): sites with nonfinite events rank
#: first (by event count), then by relative drift of the spine mean between
#: the first and last observed steps; ties break on site_id.
RANKING_SCORE = "nonfinite_first_drift_v1"

_FAN_QS = (0.05, 0.25, 0.5, 0.75, 0.95)

#: Fixed strip heights: ``width`` is the one public geometry knob across the
#: renderer family (render_detail/render_contact_sheet derive height too).
_FAN_HEIGHT = 240
_WATERFALL_HEIGHT = 320

# Colorblind-safe fixed palette (Okabe-Ito derived), hex only.
_C_MEDIAN = "#0072b2"
_C_BAND_INNER = "#8ecae6"
_C_BAND_OUTER = "#d0e7f5"
_C_MINMAX = "#555555"
_C_NONFINITE = "#d55e00"
_C_SKIP = "#cc79a7"
_C_SEGMENT = "#009e73"
_C_TEXT = "#222222"
_C_GRID = "#dddddd"
_C_COARSE = "#f4e3c1"


@dataclass(frozen=True)
class SeriesPoint:
    """One (step-span, block-truth, observation) sample of a series."""

    step_lo: int
    step_hi: int
    block: StepBlockRecord | None
    observation: ObservationRecord
    coarsened: bool = False


@dataclass(frozen=True)
class HistoryView:
    """Session-side read view over one run's committed history.

    Built from a :class:`HistoryReader` (disk artifact) or directly from
    committed blocks (RAM ring). Renderers, iterators, and the pandas
    conversion all read THIS one shape.
    """

    run: RunRecord
    sites: dict[str, SiteRecord]
    blocks: tuple[CommittedBlock, ...] = field(default_factory=tuple)

    @classmethod
    def from_reader(cls, reader: HistoryReader) -> HistoryView:
        """Materialize a view from a disk artifact (checksums verified)."""

        observations_by_step: dict[int, list[ObservationRecord]] = {}
        for obs in reader.observations():
            observations_by_step.setdefault(obs.global_step, []).append(obs)
        blocks = tuple(
            CommittedBlock(
                block=block,
                observations=tuple(observations_by_step.get(block.global_step, ())),
                step_lo=block.global_step,
                step_hi=block.global_step,
            )
            for block in reader.step_blocks()
        )
        return cls(run=reader.run, sites=dict(reader.sites), blocks=blocks)

    @classmethod
    def from_blocks(
        cls,
        run: RunRecord,
        sites: dict[str, SiteRecord],
        blocks: tuple[CommittedBlock, ...],
    ) -> HistoryView:
        """Wrap in-memory committed blocks (the RAM-ring spelling)."""

        return cls(run=run, sites=dict(sites), blocks=tuple(blocks))

    @classmethod
    def from_collector(cls, collector: Any) -> HistoryView:
        """Read the live collector's RAM ring (session-side spelling)."""

        return cls(
            run=collector.run,
            sites=collector.site_catalog,
            blocks=collector.ring.blocks,
        )

    def series(
        self,
        site_id: str,
        stream: str = "activation",
        phase: str | None = None,
    ) -> tuple[SeriesPoint, ...]:
        """Return one site/stream series in step order, phase-disambiguated.

        Raises
        ------
        WatchRenderError
            ``watch_site_unknown`` for an unknown site;
            ``watch_phase_ambiguous`` when multiple phases exist and none
            was named; ``watch_series_empty`` when nothing matches.
        """

        if site_id not in self.sites:
            known = ", ".join(sorted(self.sites)) or "<none>"
            raise WatchRenderError(
                f"site_id {site_id!r} is not in this run's catalog. Known sites: {known}.",
                code="watch_site_unknown",
                site_id=site_id,
                remedy="Use a site_id from view.sites.",
            )
        phases_seen: set[str] = set()
        points: list[SeriesPoint] = []
        for committed in self.blocks:
            for obs in committed.observations:
                if obs.site_id != site_id or obs.stream != stream:
                    continue
                phases_seen.add(obs.phase)
                if phase is not None and obs.phase != phase:
                    continue
                points.append(
                    SeriesPoint(
                        step_lo=committed.step_lo,
                        step_hi=committed.step_hi,
                        block=committed.block,
                        observation=obs,
                        coarsened=committed.coarsened,
                    )
                )
        if phase is None and len(phases_seen) > 1:
            raise WatchRenderError(
                f"site {site_id!r} stream {stream!r} carries multiple phases "
                f"{sorted(phases_seen)}; a silently merged phase axis would "
                "mix pre/post-clip (or scaled/unscaled) numbers.",
                code="watch_phase_ambiguous",
                phases=sorted(phases_seen),
                remedy="Pass phase= explicitly.",
            )
        if not points:
            raise WatchRenderError(
                f"No observations for site {site_id!r} stream {stream!r}"
                + (f" phase {phase!r}" if phase else "")
                + ". Missing is never zero; there is nothing to draw.",
                code="watch_series_empty",
                site_id=site_id,
                stream=stream,
                remedy="Check the plan table: was this series scheduled?",
            )
        return tuple(points)

    def site_streams(self, site_id: str) -> tuple[tuple[str, str], ...]:
        """Return the (stream, phase) pairs observed for one site."""

        pairs: OrderedDict[tuple[str, str], None] = OrderedDict()
        for committed in self.blocks:
            for obs in committed.observations:
                if obs.site_id == site_id:
                    pairs.setdefault((obs.stream, obs.phase))
        return tuple(pairs)

    def rows(self) -> list[dict[str, Any]]:
        """Long-form observation rows (the iterator surface)."""

        return [
            observation_row(obs, step_lo=committed.step_lo, step_hi=committed.step_hi)
            for committed in self.blocks
            for obs in committed.observations
        ]

    def to_pandas(self) -> Any:
        """Return the long-form rows as a pandas DataFrame.

        Raises
        ------
        WatchRenderError
            ``watch_tabular_extra_missing`` when pandas is not installed.
        """

        try:
            import pandas
        except ImportError as error:
            raise WatchRenderError(
                "to_pandas() needs pandas, which is not installed.",
                code="watch_tabular_extra_missing",
                remedy="pip install pandas (or the torchlens[test] extra).",
            ) from error
        return pandas.DataFrame(self.rows())

    def _repr_html_(self) -> str:
        """Static notebook repr: the first contact-sheet page."""

        try:
            return render_contact_sheet(self)
        except WatchRenderError as error:
            return f"<pre>HistoryView (unrenderable: {error.fields.get('code')})</pre>"


class _Svg:
    """Minimal deterministic SVG composer (vector text, no timestamps)."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self._parts: list[str] = []

    def add(self, element: str) -> None:
        """Append one raw SVG element."""

        self._parts.append(element)

    def line(  # noqa: PLR0913 -- flat coordinates mirror the SVG attribute grammar (x1/y1/x2/y2); packing them into point tuples reads worse at every call site
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: str,
        w: float = 1.0,
        dash: str | None = None,
    ) -> None:
        """Append a line element."""

        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{color}" stroke-width="{w:.2f}"{dash_attr}/>'
        )

    def rect(  # noqa: PLR0913 -- flat geometry mirrors the SVG rect attribute grammar (x/y/width/height)
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str,
        opacity: float = 1.0,
        stroke: str = "none",
    ) -> None:
        """Append a rectangle element."""

        self.add(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
            f'fill="{fill}" fill-opacity="{opacity:.2f}" stroke="{stroke}"/>'
        )

    def text(  # noqa: PLR0913 -- flat coordinates mirror the SVG text attribute grammar (x/y/text-anchor)
        self,
        x: float,
        y: float,
        content: str,
        size: int = 10,
        color: str = _C_TEXT,
        anchor: str = "start",
    ) -> None:
        """Append a vector text element (escaped)."""

        escaped = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.add(
            f'<text x="{x:.2f}" y="{y:.2f}" font-family="monospace" '
            f'font-size="{size}" fill="{color}" text-anchor="{anchor}">{escaped}</text>'
        )

    def polyline(
        self, pts: list[tuple[float, float]], color: str, w: float = 1.5, dash: str | None = None
    ) -> None:
        """Append an open polyline."""

        if len(pts) < 2:
            return
        coords = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<polyline points="{coords}" fill="none" stroke="{color}" '
            f'stroke-width="{w:.2f}"{dash_attr}/>'
        )

    def polygon(self, pts: list[tuple[float, float]], fill: str, opacity: float) -> None:
        """Append a filled polygon (the fan bands)."""

        if len(pts) < 3:
            return
        coords = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
        self.add(f'<polygon points="{coords}" fill="{fill}" fill-opacity="{opacity:.2f}"/>')

    def image_png(self, x: float, y: float, w: float, h: float, png_bytes: bytes) -> None:
        """Embed one PNG (the declared-resolution raster interior)."""

        encoded = base64.b64encode(png_bytes).decode("ascii")
        self.add(
            f'<image x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
            f'preserveAspectRatio="none" '
            f'xlink:href="data:image/png;base64,{encoded}"/>'
        )

    def to_string(self) -> str:
        """Serialize the document (root declares xmlns:xlink; cairosvg gate)."""

        body = "\n".join(self._parts)
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}">\n'
            f'<rect x="0" y="0" width="{self.width}" height="{self.height}" fill="white"/>\n'
            f"{body}\n</svg>\n"
        )


@dataclass(frozen=True)
class _Axis:
    """Linear axis mapping data coordinates to pixel coordinates."""

    lo: float
    hi: float
    px_lo: float
    px_hi: float

    def px(self, value: float) -> float:
        """Map one data value to its pixel coordinate."""

        span = self.hi - self.lo
        if span == 0:
            return (self.px_lo + self.px_hi) / 2.0
        frac = (value - self.lo) / span
        return self.px_lo + frac * (self.px_hi - self.px_lo)


def _step_axis(points: tuple[SeriesPoint, ...], px_lo: float, px_hi: float) -> _Axis:
    """Build the step axis over the union of spans."""

    lo = min(p.step_lo for p in points)
    hi = max(p.step_hi for p in points)
    return _Axis(float(lo), float(hi if hi > lo else lo + 1), px_lo, px_hi)


def _finite_or_none(value: float | None) -> float | None:
    """Return the value when finite, else None."""

    if value is None or not math.isfinite(value):
        return None
    return value


def _value_range(points: tuple[SeriesPoint, ...]) -> tuple[float, float]:
    """Exact finite min/max over the series' spines (fan y-range)."""

    los: list[float] = []
    his: list[float] = []
    for point in points:
        spine = point.observation.spine
        if spine is None:
            continue
        lo = _finite_or_none(spine.finite_min)
        hi = _finite_or_none(spine.finite_max)
        if lo is not None:
            los.append(lo)
        if hi is not None:
            his.append(hi)
    if not los:
        return (0.0, 1.0)
    lo, hi = min(los), max(his)
    if lo == hi:
        pad = abs(lo) * 0.1 or 1.0
        return (lo - pad, hi + pad)
    return (lo, hi)


def _quantile_row(observation: ObservationRecord) -> dict[float, float] | None:
    """Derived fan quantiles for one observation, or None without a sketch."""

    sketch = observation.sketch
    if sketch is None or observation.presence != "observed":
        return None
    if _sketch_finite_total(sketch) == 0:
        return None
    return {q: derived_quantile(sketch, q).value for q in _FAN_QS}


def _sketch_finite_total(sketch: HistogramResult) -> int:
    """Finite population folded into the sketch's grid + finite specials."""

    finite_specials = ("zero", "pos_underflow", "neg_underflow", "pos_overflow", "neg_overflow")
    return (
        sum(sketch.pos_counts)
        + sum(sketch.neg_counts)
        + sum(int(sketch.specials.get(key, 0)) for key in finite_specials)
    )


def _nonfinite_count(spine: SpineResult | None) -> int:
    """Total NaN/inf events on one spine (0 for absent spines)."""

    if spine is None:
        return 0
    return int(spine.count_nan + spine.count_posinf + spine.count_neginf)


def _draw_step_marks(
    svg: _Svg, points: tuple[SeriesPoint, ...], axis: _Axis, y_top: float, y_bot: float
) -> None:
    """Draw skip marks, segment boundaries, and coarsening spans."""

    seen_segments: set[str] = set()
    for point in points:
        x_lo = axis.px(point.step_lo)
        x_hi = axis.px(point.step_hi)
        if point.coarsened:
            svg.rect(x_lo, y_top, max(x_hi - x_lo, 1.0), y_bot - y_top, _C_COARSE, opacity=0.5)
        block = point.block
        if block is None:
            continue
        if block.optimizer_status == "skipped":
            svg.text((x_lo + x_hi) / 2.0, y_top + 8, "x", size=9, color=_C_SKIP, anchor="middle")
        if block.segment_id not in seen_segments:
            seen_segments.add(block.segment_id)
            if len(seen_segments) > 1:
                svg.line(x_lo, y_top, x_lo, y_bot, _C_SEGMENT, w=1.0, dash="4,3")
                svg.text(x_lo + 2, y_top + 18, "segment", size=8, color=_C_SEGMENT)


def _draw_fan_body(
    svg: _Svg, points: tuple[SeriesPoint, ...], axis: _Axis, y_axis: _Axis
) -> tuple[int, int]:
    """Draw bands, median, min/max, presence gaps; return (gaps, nonfinite)."""

    outer_hi: list[tuple[float, float]] = []
    outer_lo: list[tuple[float, float]] = []
    inner_hi: list[tuple[float, float]] = []
    inner_lo: list[tuple[float, float]] = []
    median: list[tuple[float, float]] = []
    mins: list[tuple[float, float]] = []
    maxs: list[tuple[float, float]] = []
    gap_count = 0
    nonfinite_total = 0
    for point in points:
        x = (axis.px(point.step_lo) + axis.px(point.step_hi)) / 2.0
        obs = point.observation
        nonfinite_total += _nonfinite_count(obs.spine)
        if obs.presence != "observed":
            gap_count += 1
            svg.text(
                x, y_axis.px_hi - 2, obs.presence[0], size=8, color=_C_NONFINITE, anchor="middle"
            )
            continue
        quantiles = _quantile_row(obs)
        if quantiles is not None:
            outer_hi.append((x, y_axis.px(quantiles[0.95])))
            outer_lo.append((x, y_axis.px(quantiles[0.05])))
            inner_hi.append((x, y_axis.px(quantiles[0.75])))
            inner_lo.append((x, y_axis.px(quantiles[0.25])))
            median.append((x, y_axis.px(quantiles[0.5])))
        spine = obs.spine
        if spine is not None:
            lo = _finite_or_none(spine.finite_min)
            hi = _finite_or_none(spine.finite_max)
            if lo is not None:
                mins.append((x, y_axis.px(lo)))
            if hi is not None:
                maxs.append((x, y_axis.px(hi)))
        if _nonfinite_count(obs.spine):
            svg.text(x, y_axis.px_hi - 12, "!", size=10, color=_C_NONFINITE, anchor="middle")
    svg.polygon(outer_hi + outer_lo[::-1], _C_BAND_OUTER, 0.8)
    svg.polygon(inner_hi + inner_lo[::-1], _C_BAND_INNER, 0.8)
    svg.polyline(median, _C_MEDIAN, w=1.8)
    svg.polyline(mins, _C_MINMAX, w=0.8, dash="2,3")
    svg.polyline(maxs, _C_MINMAX, w=0.8, dash="2,3")
    return gap_count, nonfinite_total


def render_fan(
    view: HistoryView,
    site_id: str,
    stream: str = "activation",
    phase: str | None = None,
    *,
    width: int = 640,
) -> str:
    """Render the quantile-fan strip for one series (the default view).

    Median plus p25-75 and p5-95 bands come from sketch-DERIVED quantiles
    (approximate, resolution disclosed in the caption); the dotted min/max
    envelope is EXACT from the spine, so true extremes are never lost --
    quantiles are outlier-robust by construction and need no
    outlier-rejection knob (D24). Strip height is fixed (``width`` is the
    one geometry knob, matching :func:`render_detail`).
    """

    height = _FAN_HEIGHT
    points = view.series(site_id, stream, phase)
    svg = _Svg(width, height)
    margin_l, margin_r, margin_t, margin_b = 56, 12, 26, 34
    axis = _step_axis(points, margin_l, width - margin_r)
    lo, hi = _value_range(points)
    y_axis = _Axis(lo, hi, height - margin_b, margin_t)
    for frac in (0.0, 0.5, 1.0):
        value = lo + frac * (hi - lo)
        y = y_axis.px(value)
        svg.line(margin_l, y, width - margin_r, y, _C_GRID)
        svg.text(margin_l - 4, y + 3, f"{value:.3g}", size=8, anchor="end")
    _draw_step_marks(svg, points, axis, margin_t, height - margin_b)
    gaps, nonfinite = _draw_fan_body(svg, points, axis, y_axis)
    site = view.sites[site_id]
    resolved_phase = phase or points[0].observation.phase
    svg.text(margin_l, 12, f"{site.display_label} [{stream}/{resolved_phase}]", size=11)
    caption = (
        f"median + p25-75/p5-95 derived from sketch (approximate; "
        f"bpo={view.run.descriptor.bins_per_octave}); min/max exact"
    )
    if gaps:
        caption += f"; {gaps} presence gap(s)"
    if nonfinite:
        caption += f"; {nonfinite} nonfinite event(s)"
    svg.text(margin_l, height - 8, caption, size=8)
    svg.text(
        margin_l,
        height - 20,
        f"steps {points[0].step_lo}..{points[-1].step_hi} ({len(points)} observations)",
        size=8,
    )
    return svg.to_string()


def _waterfall_raster(points: tuple[SeriesPoint, ...], view: HistoryView) -> tuple[bytes, int, int]:
    """Build the log-count density PNG (rows: bands top-to-bottom)."""

    import numpy as np
    from PIL import Image

    desc = view.run.descriptor
    bins = desc.bins_per_side
    # Row layout: nonfinite, pos_overflow, pos bins (desc), pos_underflow,
    # zero, neg_underflow, neg bins (asc magnitude), neg_overflow.
    n_rows = bins * 2 + 6
    n_cols = len(points)
    counts = np.zeros((n_rows, n_cols), dtype=np.float64)
    special = np.zeros((n_rows, n_cols), dtype=bool)
    for col, point in enumerate(points):
        sketch = point.observation.sketch
        spine = point.observation.spine
        row = 0
        counts[row, col] = _nonfinite_count(spine)
        special[row, col] = True
        row += 1
        if sketch is None:
            continue
        sp = sketch.specials
        for key in ("pos_overflow",):
            counts[row, col] = sp.get(key, 0)
            special[row, col] = True
        row += 1
        for i in range(bins - 1, -1, -1):
            counts[row, col] = sketch.pos_counts[i]
            row += 1
        for key in ("pos_underflow", "zero", "neg_underflow"):
            counts[row, col] = sp.get(key, 0)
            special[row, col] = True
            row += 1
        for i in range(bins):
            counts[row, col] = sketch.neg_counts[i]
            row += 1
        counts[row, col] = sp.get("neg_overflow", 0)
        special[row, col] = True
    density = np.log1p(counts)
    peak = density.max()
    if peak > 0:
        density = density / peak
    rgb = np.full((n_rows, n_cols, 3), 255, dtype=np.uint8)
    # Grid cells ramp white -> deep blue; special bands white -> vermillion.
    ramp = (density * 255).astype(np.uint8)
    grid_mask = ~special
    rgb[..., 0] = np.where(
        grid_mask, 255 - (ramp * 0.85).astype(np.uint8), 255 - (ramp * 0.16).astype(np.uint8)
    )
    rgb[..., 1] = np.where(
        grid_mask, 255 - (ramp * 0.55).astype(np.uint8), 255 - (ramp * 0.63).astype(np.uint8)
    )
    rgb[..., 2] = np.where(
        grid_mask, 255 - (ramp * 0.30).astype(np.uint8), 255 - (ramp * 1.0).astype(np.uint8)
    )
    image = Image.fromarray(rgb, mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue(), n_rows, n_cols


def render_waterfall(
    view: HistoryView,
    site_id: str,
    stream: str = "activation",
    phase: str | None = None,
    *,
    width: int = 640,
) -> str:
    """Render the histogram waterfall (step x signed-log2-bin density).

    The density interior is a PIL raster embedded at its natural declared
    resolution (one pixel per bin per logged step); zero, nonfinite, and
    out-of-range counts are separately visible annotated bands, never
    folded into edge bins. Variable-width columns mark RAM-only
    coarsening spans. Fixed bins, never rebinned; one column per logged
    step (their ten averaged columns are the credited contrast). Strip
    height is fixed (``width`` is the one geometry knob, matching
    :func:`render_detail`).
    """

    height = _WATERFALL_HEIGHT
    points = view.series(site_id, stream, phase)
    png, n_rows, n_cols = _waterfall_raster(points, view)
    svg = _Svg(width, height)
    margin_l, margin_r, margin_t, margin_b = 72, 12, 26, 40
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    svg.image_png(margin_l, margin_t, plot_w, plot_h, png)
    desc = view.run.descriptor
    row_h = plot_h / n_rows
    labels = (
        (0, "nonfinite"),
        (1, f">2^{desc.hi_exp}"),
        (1 + desc.bins_per_side + 1, "~0"),
        (n_rows - 1, f"<-2^{desc.hi_exp}"),
    )
    for row_index, label in labels:
        svg.text(margin_l - 4, margin_t + (row_index + 0.7) * row_h, label, size=7, anchor="end")
    coarse_cols = [i for i, p in enumerate(points) if p.coarsened]
    col_w = plot_w / n_cols
    for i in coarse_cols:
        svg.rect(margin_l + i * col_w, margin_t - 4, col_w, 3, _C_COARSE)
    site = view.sites[site_id]
    resolved_phase = phase or points[0].observation.phase
    svg.text(margin_l, 12, f"{site.display_label} [{stream}/{resolved_phase}] waterfall", size=11)
    svg.text(
        margin_l,
        height - 24,
        f"steps {points[0].step_lo}..{points[-1].step_hi}; raster "
        f"{n_cols}x{n_rows} px (1 col per logged step, 1 row per bin)",
        size=8,
    )
    svg.text(
        margin_l,
        height - 12,
        f"fixed signed log2 bins (bpo={desc.bins_per_octave}, window "
        f"[2^{desc.lo_exp}, 2^{desc.hi_exp}]), never rebinned; specials "
        "as annotated bands; coarsened spans ticked above",
        size=8,
    )
    return svg.to_string()


def rank_sites(view: HistoryView, stream: str = "activation") -> list[tuple[str, float, str]]:
    """Rank sites by the NAMED score (nonfinite first, then mean drift).

    Returns ``(site_id, score, reason)`` triples sorted most-interesting
    first with deterministic site_id tiebreaks. The score name is
    :data:`RANKING_SCORE`; scores are comparable only within one run.
    """

    rows: list[tuple[str, float, str]] = []
    for site_id in sorted(view.sites):
        first_mean: float | None = None
        last_mean: float | None = None
        nonfinite = 0
        seen = False
        for committed in view.blocks:
            for obs in committed.observations:
                if obs.site_id != site_id or obs.stream != stream:
                    continue
                seen = True
                nonfinite += _nonfinite_count(obs.spine)
                if obs.spine is not None and obs.spine.mean is not None:
                    if first_mean is None:
                        first_mean = obs.spine.mean
                    last_mean = obs.spine.mean
        if not seen:
            continue
        drift = 0.0
        if first_mean is not None and last_mean is not None:
            drift = abs(last_mean - first_mean) / (abs(first_mean) + 1e-12)
        if nonfinite:
            rows.append((site_id, 1e9 + float(nonfinite), f"nonfinite x{nonfinite}"))
        else:
            rows.append((site_id, drift, f"mean drift {drift:.3g}"))
    rows.sort(key=lambda row: (-row[1], row[0]))
    return rows


def contact_sheet_pages(
    view: HistoryView, stream: str = "activation", *, tiles_per_page: int = 12
) -> int:
    """Number of contact-sheet pages under deterministic full pagination."""

    ranked = rank_sites(view, stream)
    return max(1, math.ceil(len(ranked) / tiles_per_page))


def _draw_tile(
    svg: _Svg,
    view: HistoryView,
    ranked_row: tuple[str, float, str],
    stream: str,
    box: tuple[float, float, float, float],
) -> None:
    """Draw one contact-sheet tile (a :func:`rank_sites` row): sparkline + badges."""

    site_id, _score, reason = ranked_row
    x, y, w, h = box
    site = view.sites[site_id]
    svg.rect(x, y, w, h, "white", stroke=_C_GRID)
    label = site.display_label
    if len(label) > 30:
        label = label[:27] + "..."
    svg.text(x + 4, y + 12, label, size=9)
    try:
        points = view.series(site_id, stream, None)
    except WatchRenderError:
        svg.text(x + 4, y + h / 2, "(phase-ambiguous; see detail)", size=8)
        return
    axis = _step_axis(points, x + 4, x + w - 4)
    lo, hi = _value_range(points)
    y_axis = _Axis(lo, hi, y + h - 16, y + 18)
    means: list[tuple[float, float]] = []
    gaps = 0
    for point in points:
        px = (axis.px(point.step_lo) + axis.px(point.step_hi)) / 2.0
        obs = point.observation
        if obs.presence != "observed" or obs.spine is None or obs.spine.mean is None:
            gaps += obs.presence != "observed"
            continue
        means.append((px, y_axis.px(obs.spine.mean)))
        spine_lo = _finite_or_none(obs.spine.finite_min)
        spine_hi = _finite_or_none(obs.spine.finite_max)
        if spine_lo is not None and spine_hi is not None:
            svg.line(px, y_axis.px(spine_lo), px, y_axis.px(spine_hi), _C_BAND_INNER, w=1.0)
    svg.polyline(means, _C_MEDIAN, w=1.2)
    badge = reason + (f"; {gaps} gap(s)" if gaps else "")
    color = _C_NONFINITE if "nonfinite" in reason else _C_TEXT
    svg.text(x + 4, y + h - 4, badge, size=7, color=color)


def render_contact_sheet(
    view: HistoryView,
    stream: str = "activation",
    *,
    page: int = 0,
    tiles_per_page: int = 12,
    width: int = 900,
) -> str:
    """Render one page of the ranked overview contact sheet (D24).

    Hierarchically ordered tiles (catalog order within equal rank), ranked
    by :data:`RANKING_SCORE` with nonfinite events first; a mandatory
    "N of M" footer and deterministic full pagination -- there is no
    hidden top-N truncation, only numbered pages.

    Raises
    ------
    WatchRenderError
        ``watch_page_out_of_range`` when ``page`` exceeds the deterministic
        page count.
    """

    ranked = rank_sites(view, stream)
    total = len(ranked)
    pages = max(1, math.ceil(total / tiles_per_page)) if total else 1
    if page < 0 or page >= pages:
        raise WatchRenderError(
            f"page {page} is out of range; this sheet has {pages} page(s).",
            code="watch_page_out_of_range",
            page=page,
            pages=pages,
            remedy=f"Pass page in [0, {pages - 1}].",
        )
    subset = ranked[page * tiles_per_page : (page + 1) * tiles_per_page]
    cols = 3
    tile_w = (width - 24) / cols
    tile_h = 110.0
    rows_needed = max(1, math.ceil(len(subset) / cols))
    height = int(rows_needed * (tile_h + 8) + 60)
    svg = _Svg(width, height)
    svg.text(12, 16, f"watch contact sheet -- run {view.run.run_id[:12]} [{stream}]", size=12)
    for index, ranked_row in enumerate(subset):
        row, col = divmod(index, cols)
        _draw_tile(
            svg,
            view,
            ranked_row,
            stream,
            (12 + col * (tile_w + 4), 28 + row * (tile_h + 8), tile_w, tile_h),
        )
    shown_lo = page * tiles_per_page + 1 if subset else 0
    shown_hi = page * tiles_per_page + len(subset)
    svg.text(
        12,
        height - 10,
        f"showing sites {shown_lo}-{shown_hi} of {total} -- page {page + 1} of {pages} "
        f"(rank: {RANKING_SCORE}; nonfinite first)",
        size=9,
    )
    return svg.to_string()


def render_detail(
    view: HistoryView,
    site_id: str,
    *,
    width: int = 720,
) -> str:
    """Render the per-site detail sheet: every stream on one step axis.

    Each observed (stream, phase) pair gets a sparkline row plus its
    disclosure line (grad scale, estimation, sample size, presence gaps,
    nonfinite events). Optimizer skips and segment boundaries mark every
    row identically, so cross-stream alignment is readable.
    """

    pairs = view.site_streams(site_id)
    if not pairs:
        raise WatchRenderError(
            f"site {site_id!r} has no observations in this view.",
            code="watch_series_empty",
            site_id=site_id,
            remedy="Check the plan table: was this site scheduled?",
        )
    row_h = 92.0
    height = int(40 + row_h * len(pairs))
    svg = _Svg(width, height)
    site = view.sites[site_id]
    svg.text(12, 16, f"site detail -- {site.display_label} ({site_id})", size=12)
    for index, (stream, phase) in enumerate(pairs):
        y0 = 30 + index * row_h
        points = view.series(site_id, stream, phase)
        axis = _step_axis(points, 150, width - 16)
        lo, hi = _value_range(points)
        y_axis = _Axis(lo, hi, y0 + row_h - 26, y0 + 12)
        _draw_step_marks(svg, points, axis, y0 + 8, y0 + row_h - 26)
        gaps, nonfinite = _draw_fan_body(svg, points, axis, y_axis)
        svg.text(12, y0 + 24, f"{stream}", size=9)
        svg.text(12, y0 + 36, f"phase={phase}", size=7)
        disclosures: list[str] = []
        scales = {p.observation.grad_scale for p in points if p.observation.grad_scale}
        if scales:
            disclosures.append(f"grad_scale={'/'.join(sorted(scales))}")
        if any(p.observation.estimated for p in points):
            sizes = {p.observation.sample_size for p in points if p.observation.sample_size}
            disclosures.append(f"estimated (n={'/'.join(str(s) for s in sorted(sizes))})")
        if gaps:
            disclosures.append(f"{gaps} gap(s)")
        if nonfinite:
            disclosures.append(f"{nonfinite} nonfinite")
        cadence = view.run.cadences.get(stream)
        if cadence and cadence > 1:
            disclosures.append(f"cadence {cadence}")
        svg.text(12, y0 + 48, "; ".join(disclosures) if disclosures else "no disclosures", size=7)
    return svg.to_string()


def color_by_watch(
    view: HistoryView,
    *,
    step: int,
    stream: str = "activation",
    stat: str = "finite_absmax",
) -> Any:
    """Graph join (D24): a ``draw(color_by=...)`` callable from watch stats.

    Maps each drawn op node to the named spine stat of the watch site whose
    module path matches the node's owning module at ``step``. Nodes without
    a matching site return ``None`` and stay honestly unencoded (the
    encoding channel's disclosed not-available note).
    """

    values: dict[str, float] = {}
    for committed in view.blocks:
        if not (committed.step_lo <= step <= committed.step_hi):
            continue
        for obs in committed.observations:
            if obs.stream != stream or obs.presence != "observed" or obs.spine is None:
                continue
            site = view.sites.get(obs.site_id)
            module_path = site.module_path if site is not None else None
            value = getattr(obs.spine, stat, None)
            if module_path is not None and value is not None and math.isfinite(value):
                values[module_path] = float(value)

    def node_value(node: Any) -> float | None:
        """Resolve one drawn node to its watch stat (None = unencoded)."""

        module = getattr(node, "module", None)
        address = module[0] if isinstance(module, tuple) and module else None
        if address is None:
            return None
        return values.get(str(address))

    return node_value


__all__ = [
    "RANKING_SCORE",
    "HistoryView",
    "SeriesPoint",
    "color_by_watch",
    "contact_sheet_pages",
    "rank_sites",
    "render_contact_sheet",
    "render_detail",
    "render_fan",
    "render_waterfall",
]
