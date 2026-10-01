"""Stage-0 deterministic audit (themes memo section 7).

Every render, no model in the loop: a Stage-0 failure is a build bug and
never reaches an evaluator. The checks here are the mechanically decidable
half of the protocol -- exact ``dot -Tjson`` label geometry, output-size
caps, colour spread, legend-node agreement, compaction non-identity, unit
invariance, coverage, disclosure presence, the label-spelling audit, and
the CVD accessibility gate over skin palettes.

The geometry gate is BASELINED in R0 on today's corpus before it blocks
anything (a screening pass already found shipped ``arg N`` head-labels
penetrating nodes on stock resnet18); the runner's report states how
head/tail labels are treated: they are counted as penetration candidates
and reported, and the R0 baseline is the blocking reference.

Every spelling here is DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from typing import Any

from ...._io import _json

__all__ = [
    "AuditFinding",
    "AuditReport",
    "Stage0Checks",
    "geometry_findings",
    "luminance",
    "palette_distinguishability",
    "run_stage0",
    "simulate_cvd",
]

#: Geometry tolerance: overlaps above this many points are penetrations.
PENETRATION_TOLERANCE_PT = 0.25

#: Output caps (memo section 7): PNG <= 40 MPx, longest side <= 12,000 px,
#: aspect within [1:6, 6:1].
MAX_MEGAPIXELS = 40.0
MAX_SIDE_PX = 12_000
MAX_ASPECT = 6.0

#: Colour-spread gates.
MIN_DISTINCT_FILLS = 8
MAX_LUMINANCE_BUCKET_SHARE = 0.40
LUMINANCE_BUCKETS = 5
MAX_NEAR_WHITE_SHARE = 0.10
NEAR_WHITE_LUMINANCE = 245.0
MIN_LEGEND_CONTRAST = 3.0


@dataclass(frozen=True)
class AuditFinding:
    """One Stage-0 finding: a named check, a verdict, and its evidence."""

    check: str
    passed: bool
    detail: str
    measurements: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AuditReport:
    """One artifact's Stage-0 audit: findings plus the run manifest."""

    findings: tuple[AuditFinding, ...]
    manifest: dict[str, Any]

    @property
    def passed(self) -> bool:
        """Return whether every finding passed."""

        return all(finding.passed for finding in self.findings)

    def failed_checks(self) -> tuple[str, ...]:
        """Return the names of failed checks."""

        return tuple(finding.check for finding in self.findings if not finding.passed)


def _hex_to_rgb(color: str) -> tuple[int, int, int] | None:
    """Parse ``#RRGGBB`` (first segment of a colon-list) to RGB, else None."""

    text = color.split(":")[0].strip()
    if not re.fullmatch(r"#[0-9A-Fa-f]{6}", text):
        return None
    return int(text[1:3], 16), int(text[3:5], 16), int(text[5:7], 16)


def luminance(color: str) -> float | None:
    """Return the Rec. 601 luminance of a hex colour, or None if unparseable."""

    rgb = _hex_to_rgb(color)
    if rgb is None:
        return None
    red, green, blue = rgb
    return 0.299 * red + 0.587 * green + 0.114 * blue


def _relative_luminance(color: str) -> float | None:
    """WCAG relative luminance in [0, 1] for contrast ratios."""

    rgb = _hex_to_rgb(color)
    if rgb is None:
        return None

    def _channel(value: int) -> float:
        """Linearize one 8-bit sRGB channel (IEC 61966-2-1)."""

        scaled = value / 255.0
        return scaled / 12.92 if scaled <= 0.04045 else ((scaled + 0.055) / 1.055) ** 2.4

    red, green, blue = (_channel(component) for component in rgb)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def contrast_ratio(color_a: str, color_b: str) -> float | None:
    """WCAG contrast ratio between two hex colours."""

    lum_a = _relative_luminance(color_a)
    lum_b = _relative_luminance(color_b)
    if lum_a is None or lum_b is None:
        return None
    lighter, darker = max(lum_a, lum_b), min(lum_a, lum_b)
    return (lighter + 0.05) / (darker + 0.05)


# ---------------------------------------------------------------------------
# CVD simulation (Vienot/Brettel-style linear approximations) + grayscale.
# ---------------------------------------------------------------------------

_CVD_MATRICES: dict[str, tuple[tuple[float, float, float], ...]] = {
    "deuteranopia": (
        (0.625, 0.375, 0.0),
        (0.7, 0.3, 0.0),
        (0.0, 0.3, 0.7),
    ),
    "protanopia": (
        (0.567, 0.433, 0.0),
        (0.558, 0.442, 0.0),
        (0.0, 0.242, 0.758),
    ),
    "tritanopia": (
        (0.95, 0.05, 0.0),
        (0.0, 0.433, 0.567),
        (0.0, 0.475, 0.525),
    ),
}


def simulate_cvd(color: str, kind: str) -> tuple[float, float, float]:
    """Simulate one colour under a colour-vision deficiency.

    ``kind`` is ``deuteranopia`` / ``protanopia`` / ``tritanopia`` /
    ``grayscale``. Returns simulated RGB (floats, 0-255).
    """

    rgb = _hex_to_rgb(color)
    if rgb is None:
        raise ValueError(f"unparseable colour {color!r}")
    if kind == "grayscale":
        gray = luminance(color) or 0.0
        return (gray, gray, gray)
    matrix = _CVD_MATRICES.get(kind)
    if matrix is None:
        raise ValueError(f"unknown CVD kind {kind!r}")
    red, green, blue = rgb
    return tuple(  # type: ignore[return-value]
        row[0] * red + row[1] * green + row[2] * blue for row in matrix
    )


def _rgb_distance(
    color_a: tuple[float, float, float], color_b: tuple[float, float, float]
) -> float:
    """Euclidean RGB distance (0-441)."""

    return sum((a - b) ** 2 for a, b in zip(color_a, color_b, strict=True)) ** 0.5


def palette_distinguishability(
    palette: dict[str, str],
    *,
    min_distance: float = 40.0,
    grayscale_min_distance: float = 5.0,
) -> list[AuditFinding]:
    """Gate a semantic palette under all four simulations.

    Every pair of DISTINCT roles must stay at least ``min_distance`` apart
    (simulated RGB) under deuteranopia, protanopia, and tritanopia. The
    grayscale floor is deliberately LOWER (provisional, R2-tunable): a
    hue-coded palette cannot carry large luminance distances across eight
    roles -- that is exactly what the wave-2 border/corner-glyph redundant
    coding exists for -- but a near-zero grayscale distance (the measured
    legacy boolean~input 1.5) is a hard failure today. Roles mapped to the
    same colour on purpose are exempt; the gradient role compares by first
    segment.
    """

    findings: list[AuditFinding] = []
    roles = {
        role: color
        for role, color in palette.items()
        if _hex_to_rgb(color) is not None and role != "params_gradient"
    }
    for kind in ("deuteranopia", "protanopia", "tritanopia", "grayscale"):
        floor = grayscale_min_distance if kind == "grayscale" else min_distance
        simulated = {role: simulate_cvd(color, kind) for role, color in roles.items()}
        collisions: list[str] = []
        role_names = sorted(simulated)
        for index, role_a in enumerate(role_names):
            for role_b in role_names[index + 1 :]:
                if roles[role_a].lower() == roles[role_b].lower():
                    continue
                distance = _rgb_distance(simulated[role_a], simulated[role_b])
                if distance < floor:
                    collisions.append(f"{role_a}~{role_b} ({distance:.0f})")
        findings.append(
            AuditFinding(
                check=f"cvd_{kind}",
                passed=not collisions,
                detail=(
                    "all role pairs distinguishable"
                    if not collisions
                    else "confusable role pairs: " + ", ".join(collisions)
                ),
                measurements={"min_distance": floor, "collisions": collisions},
            )
        )
    return findings


# ---------------------------------------------------------------------------
# dot -Tjson geometry.
# ---------------------------------------------------------------------------


def _dot_json(dot_source: str, *, timeout: int = 120) -> dict[str, Any]:
    """Run ``dot -Tjson`` on a DOT source and parse the layout."""

    completed = subprocess.run(
        ["dot", "-Tjson"],
        input=dot_source.encode("utf-8"),
        capture_output=True,
        check=True,
        timeout=timeout,
    )
    # The bounded reader guards even this trusted-subprocess boundary: a
    # pathological layout cannot exhaust the parser (parse-depth guard).
    return _json.loads_bounded(completed.stdout.decode("utf-8"))


def _node_boxes(layout: dict[str, Any]) -> dict[str, tuple[float, float, float, float]]:
    """Return node name -> (x0, y0, x1, y1) in points."""

    boxes: dict[str, tuple[float, float, float, float]] = {}
    for obj in layout.get("objects", ()):
        pos = obj.get("pos")
        width = obj.get("width")
        height = obj.get("height")
        name = obj.get("name")
        if not pos or width is None or height is None or name is None:
            continue
        try:
            center_x, center_y = (float(part) for part in str(pos).split(","))
            half_w = float(width) * 72.0 / 2.0
            half_h = float(height) * 72.0 / 2.0
        except ValueError:
            continue
        boxes[str(name)] = (
            center_x - half_w,
            center_y - half_h,
            center_x + half_w,
            center_y + half_h,
        )
    return boxes


def _label_boxes(layout: dict[str, Any]) -> list[tuple[str, tuple[float, float, float, float]]]:
    """Estimate edge-label bounding boxes from label positions and text.

    ``dot -Tjson`` gives the label CENTER (``lp``); the box is estimated
    from text length at the emitted font size (default 8pt labels) --
    deterministic, disclosed as an estimate in the report manifest.
    """

    boxes: list[tuple[str, tuple[float, float, float, float]]] = []
    for edge in layout.get("edges", ()):
        label = edge.get("label")
        lp = edge.get("lp")
        if not label or not lp:
            continue
        try:
            center_x, center_y = (float(part) for part in str(lp).split(","))
        except ValueError:
            continue
        font_size = float(edge.get("fontsize", 8.0))
        text = max(str(label).splitlines(), key=len)
        half_w = 0.30 * font_size * len(text)
        half_h = 0.60 * font_size
        boxes.append(
            (
                str(label),
                (center_x - half_w, center_y - half_h, center_x + half_w, center_y + half_h),
            )
        )
    return boxes


def _overlap_pt(
    box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]
) -> float:
    """Return the overlap depth between two boxes in points (0 = disjoint)."""

    dx = min(box_a[2], box_b[2]) - max(box_a[0], box_b[0])
    dy = min(box_a[3], box_b[3]) - max(box_a[1], box_b[1])
    return min(dx, dy) if dx > 0 and dy > 0 else 0.0


def geometry_findings(dot_source: str) -> list[AuditFinding]:
    """Exact-layout geometry checks via ``dot -Tjson``.

    Two checks: node-node box overlap, and edge-label-vs-node penetration
    (label boxes estimated from text metrics; head/tail labels are counted
    as candidates -- the R0 baseline is the blocking reference for the
    shipped ``arg N`` family).
    """

    layout = _dot_json(dot_source)
    nodes = _node_boxes(layout)
    labels = _label_boxes(layout)
    node_names = sorted(nodes)
    node_overlaps: list[str] = []
    for index, name_a in enumerate(node_names):
        for name_b in node_names[index + 1 :]:
            depth = _overlap_pt(nodes[name_a], nodes[name_b])
            if depth > PENETRATION_TOLERANCE_PT:
                node_overlaps.append(f"{name_a}~{name_b} ({depth:.1f}pt)")
    label_penetrations: list[str] = []
    for label_text, label_box in labels:
        for name, node_box in nodes.items():
            depth = _overlap_pt(label_box, node_box)
            if depth > PENETRATION_TOLERANCE_PT:
                label_penetrations.append(f"{label_text[:24]!r}->{name} ({depth:.1f}pt)")
    return [
        AuditFinding(
            check="geometry_node_overlap",
            passed=not node_overlaps,
            detail="no node-node overlaps"
            if not node_overlaps
            else f"{len(node_overlaps)} overlaps: " + "; ".join(node_overlaps[:5]),
            measurements={"count": len(node_overlaps)},
        ),
        AuditFinding(
            check="geometry_label_penetration",
            passed=not label_penetrations,
            detail="no label-node penetrations"
            if not label_penetrations
            else f"{len(label_penetrations)} penetrations: " + "; ".join(label_penetrations[:5]),
            measurements={"count": len(label_penetrations), "label_boxes": len(labels)},
        ),
    ]


# ---------------------------------------------------------------------------
# DOT-source checks (spelling audit, fills, caps via SVG attrs).
# ---------------------------------------------------------------------------

_FILL_PATTERN = re.compile(r'fillcolor="(#[0-9A-Fa-f:]{6,})"')


def _encoded_fills(dot_source: str) -> list[str]:
    """Return the CHANNEL-encoded fills emitted in the DOT source.

    Semantic-role fills (every skin's palettes), neutral aggregate fills,
    and default grounds are excluded: the colour-spread gates govern the
    encoded ramp, not the role colours.
    """

    from ...themes import THEME_PRESETS

    excluded: set[str] = set()
    for theme in THEME_PRESETS.values():
        excluded.update(color.lower() for color in theme.semantic_palette.values())
        excluded.add(theme.neutral_aggregate_fill.lower())
        excluded.add(theme.default_fill.lower())
    fills: list[str] = []
    for line in dot_source.splitlines():
        # Legend chips (semantic legend + encoding disclosure legend) teach
        # the vocabulary; they are not encoded NODES and never gate spread.
        if "tl_legend_" in line or "tl_encoding_legend_" in line:
            continue
        fills.extend(fill for fill in _FILL_PATTERN.findall(line) if fill.lower() not in excluded)
    return fills


def label_spelling_findings(
    dot_source: str, *, baseline_headlabels: int = 0, baseline_xlabels: int = 0
) -> list[AuditFinding]:
    """The label-spelling audit: no NEW xlabel/headlabel family.

    Bridged-edge disclosures must be midpoint ``label=``; presets may not
    introduce a new ``xlabel``/``headlabel`` family beyond the shipped
    baselines supplied by the caller (R0 baselines both counts: the shipped
    ``arg N`` argument labeler emits head/xlabels today, and relocating it
    is the shed build-list item 16).
    """

    xlabels = len(re.findall(r"\bxlabel=", dot_source))
    headlabels = len(re.findall(r"\bheadlabel=", dot_source))
    return [
        AuditFinding(
            check="label_spelling",
            passed=xlabels <= baseline_xlabels and headlabels <= baseline_headlabels,
            detail=(
                f"xlabel={xlabels} (allowance {baseline_xlabels}), "
                f"headlabel={headlabels} (allowance {baseline_headlabels})"
            ),
            measurements={"xlabel": xlabels, "headlabel": headlabels},
        )
    ]


def fill_spread_findings(fills: list[str], *, cardinality: int) -> list[AuditFinding]:
    """Colour-spread checks over ENCODED fills.

    >= 8 distinct encoded fills where cardinality permits; <= 40% of encoded
    nodes in any 5-bucket luminance band; <= 10% above luminance 245.
    """

    findings: list[AuditFinding] = []
    distinct = sorted(set(fills))
    required = min(MIN_DISTINCT_FILLS, cardinality)
    findings.append(
        AuditFinding(
            check="distinct_fills",
            passed=len(distinct) >= required or not fills,
            detail=f"{len(distinct)} distinct encoded fills (required {required})",
            measurements={"distinct": len(distinct), "required": required},
        )
    )
    lums = [value for value in (luminance(color) for color in fills) if value is not None]
    if lums:
        bucket_counts = [0] * LUMINANCE_BUCKETS
        for value in lums:
            bucket = min(int(value / (256.0 / LUMINANCE_BUCKETS)), LUMINANCE_BUCKETS - 1)
            bucket_counts[bucket] += 1
        worst_share = max(bucket_counts) / len(lums)
        near_white = sum(1 for value in lums if value > NEAR_WHITE_LUMINANCE) / len(lums)
        findings.append(
            AuditFinding(
                check="luminance_buckets",
                passed=worst_share <= MAX_LUMINANCE_BUCKET_SHARE,
                detail=f"worst 5-bucket share {worst_share:.0%} (cap {MAX_LUMINANCE_BUCKET_SHARE:.0%})",
                measurements={"bucket_counts": bucket_counts},
            )
        )
        findings.append(
            AuditFinding(
                check="near_white",
                passed=near_white <= MAX_NEAR_WHITE_SHARE,
                detail=f"{near_white:.0%} of encoded fills above luminance {NEAR_WHITE_LUMINANCE:.0f}",
                measurements={"near_white_share": near_white},
            )
        )
    return findings


def output_caps_findings(svg_path: str | None) -> list[AuditFinding]:
    """Output-size caps from the SVG's declared pixel geometry."""

    if svg_path is None:
        return []
    try:
        with open(svg_path, encoding="utf-8") as handle:
            head = handle.read(4096)
    except OSError as error:
        return [
            AuditFinding(check="output_caps", passed=False, detail=f"artifact unreadable: {error}")
        ]
    match = re.search(r'width="(\d+)(?:pt|px)?"\s+height="(\d+)(?:pt|px)?"', head)
    if match is None:
        return [
            AuditFinding(
                check="output_caps",
                passed=True,
                detail="no declared pixel geometry (vector-only header)",
            )
        ]
    width, height = int(match.group(1)), int(match.group(2))
    megapixels = width * height / 1e6
    aspect = max(width, height) / max(min(width, height), 1)
    passed = (
        megapixels <= MAX_MEGAPIXELS and max(width, height) <= MAX_SIDE_PX and aspect <= MAX_ASPECT
    )
    return [
        AuditFinding(
            check="output_caps",
            passed=passed,
            detail=f"{width}x{height} ({megapixels:.1f} MPx, aspect {aspect:.1f}:1)",
            measurements={"width": width, "height": height},
        )
    ]


def disclosure_findings(dot_source: str, resolution: Any) -> list[AuditFinding]:
    """Disclosure-presence checks against the resolution's own claims."""

    findings: list[AuditFinding] = []
    required: list[tuple[str, str]] = []
    if resolution is not None:
        if resolution.source is not None:
            required.append(("coverage_line", "coverage: encoded"))
            if resolution.source.aggregation_line is not None:
                required.append(("aggregation_line", "total across passes"))
        if resolution.display_filter is not None:
            required.append(("filter_caption", "filtered ("))
            required.append(("bridged_legend_line", "reachability through omitted"))
        if resolution.budget is not None:
            required.append(("dial_by_name", "detail dial:"))
    for check, needle in required:
        findings.append(
            AuditFinding(
                check=check,
                passed=needle in dot_source,
                detail=f"required disclosure {needle!r} "
                + ("present" if needle in dot_source else "MISSING"),
            )
        )
    return findings


@dataclass(frozen=True)
class Stage0Checks:
    """Check knobs for one Stage-0 audit run.

    Attributes
    ----------
    cardinality:
        Distinct-value cardinality the channel could express (caps the
        distinct-fill requirement on tiny graphs).
    baseline_headlabels:
        The R0-baselined shipped headlabel allowance.
    baseline_xlabels:
        The R0-baselined shipped xlabel allowance.
    geometry:
        Whether to run the ``dot -Tjson`` geometry pass (needs the dot
        binary; the caller may disable it where graphviz is unavailable).
    """

    cardinality: int = MIN_DISTINCT_FILLS
    baseline_headlabels: int = 0
    baseline_xlabels: int = 0
    geometry: bool = True


def run_stage0(
    dot_source: str,
    *,
    resolution: Any = None,
    svg_path: str | None = None,
    manifest: dict[str, Any] | None = None,
    checks: Stage0Checks | None = None,
) -> AuditReport:
    """Run the Stage-0 deterministic audit over one rendered artifact.

    Parameters
    ----------
    dot_source:
        The rendered DOT source (the render's own emission, not a re-render).
    resolution:
        The :class:`~torchlens.visualization.lenses.LensResolution` that
        produced the render, for disclosure checks. ``None`` audits a bare
        draw.
    svg_path:
        Rendered SVG artifact for the output-caps check.
    manifest:
        The per-artifact run manifest (``corpus.build_run_manifest``).
    checks:
        Per-run check knobs (:class:`Stage0Checks`); ``None`` runs the
        defaults.
    """

    checks = checks or Stage0Checks()
    findings: list[AuditFinding] = []
    if checks.geometry:
        findings.extend(geometry_findings(dot_source))
    findings.extend(
        label_spelling_findings(
            dot_source,
            baseline_headlabels=checks.baseline_headlabels,
            baseline_xlabels=checks.baseline_xlabels,
        )
    )
    channel_active = resolution is not None and (
        resolution.source is not None or resolution.draw_kwargs.get("color_by") is not None
    )
    if channel_active:
        # The colour-spread gates govern the encoded ramp; a channel-free
        # artifact has no encoded fills to gate.
        fills = _encoded_fills(dot_source)
        findings.extend(fill_spread_findings(fills, cardinality=checks.cardinality))
    findings.extend(output_caps_findings(svg_path))
    findings.extend(disclosure_findings(dot_source, resolution))
    return AuditReport(findings=tuple(findings), manifest=dict(manifest or {}))
