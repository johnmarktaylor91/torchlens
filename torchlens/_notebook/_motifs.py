"""Six-state status motifs and the diverging-around-zero value map (B3, F16).

Two treescope conventions adopted NATIVELY (treescope memo section 4,
tier 1; credited on the acknowledgments page):

- **Exceptional values are PATTERNS, not colors**: NaN is an X motif,
  +Inf/-Inf are directional marks, masked/truncated cells are dots, finite
  out-of-range values carry a signed edge mark, and unchecked/unknown
  coverage is its own explicit pattern. Patterns work in any colormap, any
  skin, grayscale, and for colorblind users -- the one visual convention
  that carries honesty rather than aesthetics.
- **Diverging-around-zero is the signed-float default** (activations,
  gradients, deltas, attributions): zero-centered with a 3-sigma trim,
  bounds disclosed, all-positive fallback disclosed.

ONE table, one palette authority: both the motif table and the value
colormap are CONSUMPTION SEAMS over the themes registry
(:func:`resolve_motif_table` / :func:`resolve_diverging_anchors`). Until
the themes lane lands its skin-record rows the module-level defaults here
are the disclosed provisional stand-ins; when the registry rows exist they
win, so there is never a second authority.

Every spelling is DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass

__all__ = [
    "Motif",
    "MOTIF_STATES",
    "SignedBounds",
    "classify_value",
    "resolve_diverging_anchors",
    "resolve_motif_table",
    "resolve_signed_bounds",
    "value_color",
]

#: The closed six-state exceptional vocabulary (finite-in-range is the
#: unmarked base state and deliberately NOT in this tuple).
MOTIF_STATES = ("nan", "posinf", "neginf", "masked", "out_of_range", "unknown")


@dataclass(frozen=True)
class Motif:
    """One exceptional-value motif.

    Attributes
    ----------
    state:
        State token from :data:`MOTIF_STATES`.
    glyph:
        ASCII pattern glyph rendered in the cell (data, escaped at the
        leaf; never markup).
    css_class:
        Scoped style class under the card CSS root.
    title:
        Human hover/title text naming the state.
    """

    state: str
    glyph: str
    css_class: str
    title: str


#: Provisional stand-in table until the themes skin record ships its rows
#: (the registry wins once present; see :func:`resolve_motif_table`).
_DEFAULT_MOTIF_TABLE: tuple[Motif, ...] = (
    Motif("nan", "X", "tl-motif-nan", "NaN"),
    Motif("posinf", "^", "tl-motif-posinf", "+Inf"),
    Motif("neginf", "v", "tl-motif-neginf", "-Inf"),
    Motif("masked", "..", "tl-motif-masked", "truncated / masked (not shown)"),
    Motif("out_of_range", "!", "tl-motif-oor", "finite, outside disclosed bounds"),
    Motif("unknown", "?", "tl-motif-unknown", "unchecked / unknown coverage"),
)

#: Provisional diverging anchors (negative / zero / positive) until the
#: themes ColorMapping ships; deliberately a colorblind-safe blue-red pair.
_DEFAULT_DIVERGING_ANCHORS = ("#2166ac", "#f7f7f7", "#b2182b")


def resolve_motif_table(theme: str = "torchlens") -> tuple[Motif, ...]:
    """Return the six-state motif table for one theme.

    Consumption seam: when the themes registry publishes a
    ``status_motif_table`` (the skin-record row the themes memo assigns),
    that table is the ONE authority; until then the provisional module
    default serves, so cards, report, and draw() consume one vocabulary
    either way.
    """

    # Cosmetic resolution may never break a render: any registry fault
    # falls through to the provisional default table.
    with contextlib.suppress(Exception):
        from ..visualization import theme_registry

        table = getattr(theme_registry, "status_motif_table", None)
        if callable(table):
            rows = tuple(table(theme))
            if rows:
                return rows
    return _DEFAULT_MOTIF_TABLE


def resolve_diverging_anchors(theme: str = "torchlens") -> tuple[str, str, str]:
    """Return (negative, zero, positive) anchor colors for signed floats.

    Same seam discipline as :func:`resolve_motif_table`: a themes-registry
    ``diverging_anchors`` row wins once it exists; the module default is
    the disclosed provisional stand-in.
    """

    # Same never-break-a-render discipline as the motif seam above.
    with contextlib.suppress(Exception):
        from ..visualization import theme_registry

        anchors = getattr(theme_registry, "diverging_anchors", None)
        if callable(anchors):
            resolved = tuple(str(anchor) for anchor in anchors(theme))
            if len(resolved) == 3:
                return (resolved[0], resolved[1], resolved[2])
    return _DEFAULT_DIVERGING_ANCHORS


@dataclass(frozen=True)
class SignedBounds:
    """Resolved value-map bounds plus their mandatory disclosure.

    Attributes
    ----------
    vmin / vmax:
        Resolved display bounds (vmin < vmax unless the data is constant).
    mode:
        ``"diverging"`` (zero-centered signed map) or ``"sequential"``
        (the disclosed all-positive fallback).
    trimmed:
        Whether the 3-sigma trim narrowed the raw extrema.
    disclosure:
        One human line stating the bounds and any trim/fallback -- rendered
        beside every grid that uses them, never omitted.
    """

    vmin: float
    vmax: float
    mode: str
    trimmed: bool
    disclosure: str


def resolve_signed_bounds(
    finite_min: float | None,
    finite_max: float | None,
    mean: float | None,
    sd: float | None,
) -> SignedBounds:
    """Resolve display bounds for a float payload (memo section 4 tier 1).

    Signed data gets the zero-centered diverging map with a 3-sigma trim
    (the trim radius is three standard deviations about ZERO, i.e.
    ``3 * sqrt(mean^2 + sd^2)``, so the center of the map is honest);
    all-non-negative data falls back to a sequential ramp, disclosed.

    Parameters
    ----------
    finite_min:
        Exact finite minimum (``None`` when no finite values exist).
    finite_max:
        Exact finite maximum (``None`` when no finite values exist).
    mean:
        First moment when available; an absent moment simply skips the trim.
    sd:
        Second moment when available; an absent moment simply skips the trim.

    Returns
    -------
    SignedBounds
        Bounds, mode, and the mandatory disclosure line.
    """

    if finite_min is None or finite_max is None:
        return SignedBounds(0.0, 0.0, "sequential", False, "no finite values")
    if finite_min >= 0.0:
        return SignedBounds(
            0.0,
            float(finite_max),
            "sequential",
            False,
            f"all-positive: sequential ramp on [0, {finite_max:.4g}]",
        )
    raw = max(abs(float(finite_min)), abs(float(finite_max)))
    limit = raw
    trimmed = False
    if mean is not None and sd is not None and math.isfinite(mean) and math.isfinite(sd):
        sigma_about_zero = math.sqrt(mean * mean + sd * sd)
        three_sigma = 3.0 * sigma_about_zero
        if 0.0 < three_sigma < raw:
            limit = three_sigma
            trimmed = True
    detail = f"diverging around 0 on [-{limit:.4g}, {limit:.4g}]"
    if trimmed:
        detail += f" (3-sigma trim; |x| max {raw:.4g})"
    return SignedBounds(-limit, limit, "diverging", trimmed, detail)


def classify_value(value: float, in_mask: bool, bounds: SignedBounds) -> str:
    """Classify one cell into the base state or a motif state.

    Parameters
    ----------
    value:
        Cell value (float-coerced).
    in_mask:
        Whether the cell holds a real source value (``False`` = the
        truncation band; memo law: a band travels as mask, never a zero).
    bounds:
        Resolved display bounds for the out-of-range check.

    Returns
    -------
    str
        ``"finite"`` or one of :data:`MOTIF_STATES`.
    """

    if not in_mask:
        return "masked"
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "posinf" if value > 0 else "neginf"
    if value < bounds.vmin or value > bounds.vmax:
        return "out_of_range"
    return "finite"


def _blend(color_a: str, color_b: str, fraction: float) -> str:
    """Linearly blend two ``#rrggbb`` colors."""

    fraction = min(1.0, max(0.0, fraction))
    a = tuple(int(color_a[i : i + 2], 16) for i in (1, 3, 5))
    b = tuple(int(color_b[i : i + 2], 16) for i in (1, 3, 5))
    mixed = tuple(round(a_c + (b_c - a_c) * fraction) for a_c, b_c in zip(a, b, strict=True))
    return "#{:02x}{:02x}{:02x}".format(*mixed)


def value_color(value: float, bounds: SignedBounds, theme: str = "torchlens") -> str:
    """Map one finite in-range value to its background color.

    Diverging mode blends the themes anchors around the zero center;
    sequential mode ramps zero-anchor to positive-anchor.
    """

    negative, zero, positive = resolve_diverging_anchors(theme)
    if bounds.vmax <= bounds.vmin:
        return zero
    if bounds.mode == "sequential":
        return _blend(zero, positive, (value - bounds.vmin) / (bounds.vmax - bounds.vmin))
    if value >= 0:
        return _blend(zero, positive, value / bounds.vmax)
    return _blend(zero, negative, value / bounds.vmin)
