"""Geometry audit v2: the vizmech wave-3 oracle set (items 18-20).

The v1 instrument (``tests/support/label_geometry.py``) audits EDGE labels
only, pairs every label against every element (O(labels x everything); >20
minutes on full rolled gpt2 while the layout itself takes 0.26 s), measures
strokes at zero pen width, and cannot see a legend that tripled page width or
877 fused arrowheads under a perfect label score. This module is the widened,
indexed instrument the memo commissioned:

- **Element classes** (item 19): edge head/tail/midpoint labels PLUS node
  label text, cluster captions, the graph caption, legend-table text, and the
  reserved port-cell class (the sidecar-rail entry ticket's prerequisite:
  a rail conversion must never score a free zero because its text left the
  drawing).
- **Penwidth inflation** (item 19): every outline carries its stroke width;
  signed gaps subtract ``penwidth / 2`` so a fat border collides honestly.
- **Anti-vacuity + inventory** (item 19, planks 1/6): results carry per-class
  element counts and a text multiset; assertions pair geometry with the
  inventory so a "fix" that deletes labels cannot score a perfect zero.
- **Spatial index** (item 18): a uniform grid prefilters candidate pairs
  (measured 39x candidate reduction at 200 pt cells in the panel's
  prototype); exact signed distances run only on grid-near candidates.
- **Usability envelopes** (item 20): canvas aspect/area, whitespace, spine
  drift, min effective text size, legend-to-content ratio, arrowhead fused
  pairs + min within-node gap, caption distance, cluster width vs member
  extent -- the second oracle, calibrated by tests in BOTH directions
  (known-bad fails, known-good passes with headroom).
- **Engine discipline** (item 21 / D4, D8): the audit runs through the engine
  that rendered (``dot`` or ``neato -n``), records it, and exposes the two
  free discriminators (position-pin count, bounding-box presence).

The v1 instrument stays untouched as the locked positive control; its
calibrated glyph model and the 0.25 pt threshold are reproduced here
verbatim (LOCKED -- never weaken).

Stdlib only. Deterministic. Internal (test-support instrument; spellings
DOCUMENTED-UNSTABLE pending the naming session).
"""

from __future__ import annotations

import math
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import cast

from .._io._json import loads_bounded

__all__ = [
    "ASCENT",
    "DESCENT",
    "GRID_CELL_PT",
    "PEN_EPS",
    "AuditResult",
    "AuditViolation",
    "ClusterRecord",
    "EdgeRecord",
    "NodeRecord",
    "ParsedLayout",
    "TextBox",
    "UsabilityEnvelope",
    "audit_layout",
    "compute_usability_envelope",
    "edges_with_competing_labels",
    "graphviz_version",
    "parse_layout_json",
    "run_layout_json",
]

# Calibrated constants -- LOCKED, byte-identical to the v1 sweep values.
ASCENT = 0.78
DESCENT = 0.22
PEN_EPS = 0.25
SPLINE_SAMPLES = 24
ELLIPSE_SAMPLES = 64
#: Grid cell size for the candidate-pair prefilter (Fable r4: a 200 pt grid
#: cut 825k candidate pairs to 21k on full rolled gpt2).
GRID_CELL_PT = 200.0
#: Arrowhead health floor (D15): min within-node arrowhead gap measured
#: 2.5 pt on healthy renders, 0.86 pt on the DenseNet knot.
ARROWHEAD_GAP_FLOOR_PT = 2.5
#: Reserved node-name marker for future sidecar-rail port cells (wave 5
#: entry ticket): text in cells is tracked, never free-scored.
PORT_CELL_NAME_MARKER = "tl_rail"
#: Legend-table node name (must match ``_legend.LEGEND_NODE_NAME``).
LEGEND_NODE_MARKER = "tl_legend"

Rect = tuple[float, float, float, float]
Point = tuple[float, float]


# ---------------------------------------------------------------------------
# geometry primitives (rects are (x0, y0, x1, y1), y-up)
# ---------------------------------------------------------------------------


def _rect_union(a: Rect, b: Rect) -> Rect:
    """Return the bounding rect of two rects."""

    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _rect_rect_signed(a: Rect, b: Rect) -> float:
    """Signed AABB separation: > 0 gap, <= 0 -(min-axis penetration)."""

    ox = min(a[2], b[2]) - max(a[0], b[0])
    oy = min(a[3], b[3]) - max(a[1], b[1])
    if ox > 0 and oy > 0:
        return -min(ox, oy)
    return math.hypot(max(-ox, 0.0), max(-oy, 0.0))


def _depth_in_rect(p: Point, r: Rect) -> float:
    """Distance from a point to the rect boundary; positive iff inside."""

    return min(p[0] - r[0], r[2] - p[0], p[1] - r[1], r[3] - p[1])


def _point_rect_dist(p: Point, r: Rect) -> float:
    """Euclidean distance from a point to a rect (0 inside)."""

    dx = max(r[0] - p[0], 0.0, p[0] - r[2])
    dy = max(r[1] - p[1], 0.0, p[1] - r[3])
    return math.hypot(dx, dy)


def _points_rect_signed(points: list[Point], r: Rect) -> float:
    """Signed distance of a sampled outline/polyline to a rect.

    > 0: min gap of any sample; <= 0: -(max inside-depth of any sample).
    Sampling density is the accuracy bound, as in the v1 instrument.
    """

    pen = 0.0
    crossed = False
    gap = float("inf")
    for p in points:
        d = _depth_in_rect(p, r)
        if d >= 0.0:
            crossed = True
            if d > pen:
                pen = d
        else:
            g = _point_rect_dist(p, r)
            if g < gap:
                gap = g
    if crossed:
        return -pen
    return gap


def _sample_ellipse(cx: float, cy: float, rx: float, ry: float) -> list[Point]:
    """Sample an ellipse outline."""

    return [
        (
            cx + rx * math.cos(2 * math.pi * i / ELLIPSE_SAMPLES),
            cy + ry * math.sin(2 * math.pi * i / ELLIPSE_SAMPLES),
        )
        for i in range(ELLIPSE_SAMPLES)
    ]


def _sample_polygon(points: list[Point], per_edge: int = 8) -> list[Point]:
    """Sample a polygon outline densely enough for signed-distance checks."""

    sampled: list[Point] = []
    count = len(points)
    for i in range(count):
        a, b = points[i], points[(i + 1) % count]
        for j in range(per_edge):
            t = j / per_edge
            sampled.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
    return sampled


def _sample_beziers(ctrl: list[Point]) -> list[Point]:
    """Sample a piecewise cubic bezier (3n+1 control points)."""

    pts: list[Point] = []
    nseg = (len(ctrl) - 1) // 3
    for s in range(nseg):
        p0, p1, p2, p3 = ctrl[3 * s : 3 * s + 4]
        for i in range(SPLINE_SAMPLES + 1):
            if s > 0 and i == 0:
                continue
            t = i / SPLINE_SAMPLES
            mt = 1.0 - t
            x = mt**3 * p0[0] + 3 * mt * mt * t * p1[0] + 3 * mt * t * t * p2[0] + t**3 * p3[0]
            y = mt**3 * p0[1] + 3 * mt * mt * t * p1[1] + 3 * mt * t * t * p2[1] + t**3 * p3[1]
            pts.append((x, y))
    return pts


def _points_bbox(points: list[Point]) -> Rect:
    """Bounding rect of a point list."""

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


# ---------------------------------------------------------------------------
# parsed layout model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TextBox:
    """One rendered text block (any element class).

    ``kind`` is the element class: ``edge-head`` / ``edge-tail`` /
    ``edge-midpoint`` / ``node-label`` / ``legend-text`` / ``port-cell`` /
    ``cluster-caption`` / ``graph-caption``.
    """

    kind: str
    owner: str
    text: str
    bbox: Rect
    fontsize: float


@dataclass(frozen=True)
class NodeRecord:
    """One laid-out node: outline samples + stroke width."""

    name: str
    outline: tuple[Point, ...]
    bbox: Rect
    penwidth: float


@dataclass(frozen=True)
class EdgeRecord:
    """One laid-out edge: spline samples and arrowhead outlines."""

    name: str
    tail: str
    head: str
    spline_points: tuple[Point, ...]
    spline_bbox: Rect | None
    arrowheads: tuple[tuple[Point, ...], ...]


@dataclass(frozen=True)
class ClusterRecord:
    """One cluster: border rect, caption boxes, member node names."""

    name: str
    rect: Rect
    caption_boxes: tuple[Rect, ...]
    member_nodes: tuple[str, ...]
    penwidth: float


@dataclass(frozen=True)
class ParsedLayout:
    """Engine-attributed parse of one ``-Tjson`` layout."""

    engine: str
    bb: Rect
    nodes: tuple[NodeRecord, ...]
    clusters: tuple[ClusterRecord, ...]
    edges: tuple[EdgeRecord, ...]
    textboxes: tuple[TextBox, ...]
    pin_count: int

    def class_counts(self) -> dict[str, int]:
        """Per-element-class counts (the anti-vacuity denominator)."""

        counts: Counter[str] = Counter(box.kind for box in self.textboxes)
        counts["node"] = len(self.nodes)
        counts["cluster"] = len(self.clusters)
        counts["edge"] = len(self.edges)
        counts["arrowhead"] = sum(len(e.arrowheads) for e in self.edges)
        return dict(counts)

    def text_multiset(self, kinds: tuple[str, ...] | None = None) -> Counter[str]:
        """Text multiset for inventory assertions (plank 1)."""

        return Counter(box.text for box in self.textboxes if kinds is None or box.kind in kinds)


def graphviz_version(binary: str = "dot") -> str:
    """Return the engine's version line (for failure-message context)."""

    proc = subprocess.run([binary, "-V"], capture_output=True, text=True, timeout=30, check=False)
    return (proc.stderr or proc.stdout).strip()


def run_layout_json(dot_source: str, engine: str = "dot", timeout: int = 300) -> dict:
    """Run the NAMED engine on DOT source and return the parsed JSON layout.

    ``engine="dot"`` runs plain dot; ``engine="rank"`` (or ``"neato"``) runs
    ``neato -n`` -- the engine the rank path actually renders with. Auditing
    a rank-path artifact through dot is the D4 instrument failure (the same
    cluster measured 356 pt under dot and 1260 pt under ``neato -n``).
    """

    if engine == "dot":
        cmd = ["dot", "-Tjson"]
    elif engine in ("rank", "neato", "neato -n"):
        cmd = ["neato", "-n", "-Tjson"]
    else:
        raise ValueError(f"unknown layout engine {engine!r}; pass 'dot' or 'rank'")
    proc = subprocess.run(
        cmd,
        input=dot_source.encode(),
        capture_output=True,
        timeout=timeout,
        check=True,
    )
    # Bounded parse (the repo-wide guarded-reader law): dot's JSON nests one
    # level per draw-op array, but the reader still refuses adversarial or
    # corrupt output instead of recursing on it.
    return loads_bounded(proc.stdout.decode())


def _parse_text_ops(ops: list[dict], kind: str, owner: str) -> list[TextBox]:
    """Parse one ``*draw*`` op array's T ops into text boxes (v1 glyph model)."""

    boxes: list[TextBox] = []
    size = 14.0
    for op in ops or []:
        if op["op"] == "F":
            size = float(op["size"])
        elif op["op"] == "T":
            x, y = float(op["pt"][0]), float(op["pt"][1])
            width = float(op["width"])
            align = op.get("align", "l")
            if align == "r":
                x0 = x - width
            elif align == "c":
                x0 = x - width / 2.0
            else:
                x0 = x
            boxes.append(
                TextBox(
                    kind=kind,
                    owner=owner,
                    text=op["text"],
                    bbox=(x0, y - DESCENT * size, x0 + width, y + ASCENT * size),
                    fontsize=size,
                )
            )
    return boxes


def _shape_outlines(draw_ops: list[dict] | None) -> list[list[Point]]:
    """Extract closed outlines from a ``_draw_`` op array."""

    outlines: list[list[Point]] = []
    for op in draw_ops or []:
        if op["op"] in ("e", "E"):
            cx, cy, rx, ry = (float(v) for v in op["rect"])
            outlines.append(_sample_ellipse(cx, cy, rx, ry))
        elif op["op"] in ("p", "P"):
            outlines.append(_sample_polygon([(float(p[0]), float(p[1])) for p in op["points"]]))
    return outlines


def _node_text_kind(node_name: str) -> str:
    """Classify a node's label text into its element class."""

    if node_name.startswith(LEGEND_NODE_MARKER):
        return "legend-text"
    if PORT_CELL_NAME_MARKER in node_name:
        return "port-cell"
    return "node-label"


def _parse_edges(
    json_doc: dict,
    gvid_name: dict[int, str],
    textboxes: list[TextBox],
) -> list[EdgeRecord]:
    """Parse edge objects (splines, arrowheads); edge labels join the text set."""

    edges: list[EdgeRecord] = []
    for index, edge in enumerate(json_doc.get("edges", [])):
        tail = gvid_name.get(edge["tail"], str(edge["tail"]))
        head = gvid_name.get(edge["head"], str(edge["head"]))
        name = f"{tail}->{head}#{index}"
        spline_points: list[Point] = []
        for op in edge.get("_draw_", []) or []:
            if op["op"] in ("b", "B"):
                spline_points.extend(
                    _sample_beziers([(float(p[0]), float(p[1])) for p in op["points"]])
                )
        arrowheads: list[tuple[Point, ...]] = []
        for key in ("_hdraw_", "_tdraw_"):
            for outline in _shape_outlines(edge.get(key)):
                arrowheads.append(tuple(outline))
        for key, kind in (
            ("_hldraw_", "edge-head"),
            ("_tldraw_", "edge-tail"),
            ("_ldraw_", "edge-midpoint"),
        ):
            textboxes.extend(_parse_text_ops(edge.get(key, []), kind, name))
        edges.append(
            EdgeRecord(
                name=name,
                tail=tail,
                head=head,
                spline_points=tuple(spline_points),
                spline_bbox=_points_bbox(spline_points) if spline_points else None,
                arrowheads=tuple(arrowheads),
            )
        )
    return edges


def parse_layout_json(json_doc: dict, *, engine: str, dot_source: str = "") -> ParsedLayout:
    """Parse a ``-Tjson`` layout into the widened element model."""

    sub_cnt = json_doc.get("_subgraph_cnt", 0)
    objects = json_doc.get("objects", [])
    gvid_name = {obj["_gvid"]: obj.get("name", f"obj{obj['_gvid']}") for obj in objects}
    textboxes: list[TextBox] = []
    bb_text = json_doc.get("bb", "0,0,0,0")
    bb = tuple(float(v) for v in bb_text.split(","))
    # Graph caption rides the top-level _ldraw_.
    textboxes.extend(_parse_text_ops(json_doc.get("_ldraw_", []), "graph-caption", "__graph__"))

    clusters: list[ClusterRecord] = []
    for obj in objects[:sub_cnt]:
        raw_bb = obj.get("bb")
        if not raw_bb:
            continue
        rect = tuple(float(v) for v in raw_bb.split(","))
        caption = _parse_text_ops(obj.get("_ldraw_", []), "cluster-caption", obj.get("name", "?"))
        textboxes.extend(caption)
        member_gvids = obj.get("nodes", [])
        clusters.append(
            ClusterRecord(
                name=obj.get("name", "?"),
                rect=rect,  # type: ignore[arg-type]
                caption_boxes=tuple(box.bbox for box in caption),
                member_nodes=tuple(gvid_name.get(gvid, str(gvid)) for gvid in member_gvids),
                penwidth=float(obj.get("penwidth", 1.0)),
            )
        )

    nodes: list[NodeRecord] = []
    for obj in objects[sub_cnt:]:
        name = obj.get("name", f"obj{obj['_gvid']}")
        outlines = _shape_outlines(obj.get("_draw_"))
        label_boxes = _parse_text_ops(obj.get("_ldraw_", []), _node_text_kind(name), name)
        textboxes.extend(label_boxes)
        points: list[Point] = [p for outline in outlines for p in outline]
        if not points and label_boxes:
            # Plaintext nodes (the legend table) draw no shape outline; their
            # text union is the footprint.
            box = label_boxes[0].bbox
            for extra in label_boxes[1:]:
                box = _rect_union(box, extra.bbox)
            points = [
                (box[0], box[1]),
                (box[2], box[1]),
                (box[2], box[3]),
                (box[0], box[3]),
            ]
        if not points:
            continue
        nodes.append(
            NodeRecord(
                name=name,
                outline=tuple(points),
                bbox=_points_bbox(points),
                penwidth=float(obj.get("penwidth", 1.0)),
            )
        )

    edges = _parse_edges(json_doc, gvid_name, textboxes)

    return ParsedLayout(
        engine=engine,
        bb=bb,  # type: ignore[arg-type]
        nodes=tuple(nodes),
        clusters=tuple(clusters),
        edges=tuple(edges),
        textboxes=tuple(textboxes),
        pin_count=dot_source.count('!"'),
    )


# ---------------------------------------------------------------------------
# spatial index (item 18)
# ---------------------------------------------------------------------------


class _Grid:
    """Uniform-grid spatial index over bboxed items."""

    def __init__(self, cell: float = GRID_CELL_PT) -> None:
        """Initialize with the given cell size in points."""

        self.cell = cell
        self.cells: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
        self.items: list[tuple[Rect, object]] = []

    def _span(self, bbox: Rect, margin: float = 0.0) -> tuple[range, range]:
        """Cell index ranges covering a bbox plus margin."""

        x0 = int((bbox[0] - margin) // self.cell)
        y0 = int((bbox[1] - margin) // self.cell)
        x1 = int((bbox[2] + margin) // self.cell)
        y1 = int((bbox[3] + margin) // self.cell)
        return range(x0, x1 + 1), range(y0, y1 + 1)

    def insert(self, bbox: Rect, item: object) -> None:
        """Register one item under every cell its bbox touches."""

        index = len(self.items)
        self.items.append((bbox, item))
        xs, ys = self._span(bbox)
        for cx in xs:
            for cy in ys:
                self.cells[(cx, cy)].append(index)

    def query(self, bbox: Rect, margin: float = 0.0) -> list[tuple[Rect, object]]:
        """Return de-duplicated items whose cells intersect the bbox+margin."""

        seen: set[int] = set()
        out: list[tuple[Rect, object]] = []
        xs, ys = self._span(bbox, margin)
        for cx in xs:
            for cy in ys:
                for index in self.cells.get((cx, cy), ()):
                    if index not in seen:
                        seen.add(index)
                        out.append(self.items[index])
        return out


# ---------------------------------------------------------------------------
# the audit (items 18-19)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuditViolation:
    """One hard collision: a text box penetrating a foreign element."""

    kind: str
    text_owner: str
    text: str
    other: str
    signed_gap: float


@dataclass
class AuditResult:
    """Widened audit outcome: violations + the anti-vacuity inventory."""

    engine: str
    violations: list[AuditViolation]
    class_counts: dict[str, int]
    text_multiset: Counter = field(default_factory=Counter)
    candidate_pairs: int = 0
    checked_pairs: int = 0

    @property
    def hard_violation_count(self) -> int:
        """Count of distinct hard violations."""

        return len(self.violations)

    def describe(self, name: str = "") -> str:
        """Human-readable violation list with the engine named (plank 8)."""

        lines = [
            f"{self.hard_violation_count} hard geometry violation(s)"
            f"{f' in {name}' if name else ''} [engine={self.engine}]"
        ]
        for violation in self.violations:
            lines.append(
                f"  [{violation.kind}] {violation.text_owner} '{violation.text}' vs "
                f"{violation.other}: {violation.signed_gap:.3f} pt"
            )
        return "\n".join(lines)

    def require_minimums(self, **minimums: int) -> None:
        """Anti-vacuity floor: refuse a pass over an empty denominator.

        ``result.require_minimums(edge_head=4, node=10)`` raises AssertionError
        when a class is under its floor -- the instrument-excludes-its-case
        failure pattern the panel hit four times.
        """

        for key, minimum in minimums.items():
            klass = key.replace("_", "-")
            actual = self.class_counts.get(klass, 0)
            if actual < minimum:
                raise AssertionError(
                    f"anti-vacuity: expected >= {minimum} {klass!r} elements, found "
                    f"{actual} -- the audit may be pointed at the wrong artifact "
                    f"(class counts: {self.class_counts})"
                )


def _build_grid(parsed: ParsedLayout) -> _Grid:
    """Index every collidable element (nodes, splines, arrowheads, text)."""

    grid = _Grid()
    for node in parsed.nodes:
        grid.insert(node.bbox, node)
    for edge in parsed.edges:
        if edge.spline_bbox is not None:
            grid.insert(edge.spline_bbox, edge)
        for arrow_index, outline in enumerate(edge.arrowheads):
            grid.insert(_points_bbox(list(outline)), (edge, arrow_index, outline))
    for box in parsed.textboxes:
        grid.insert(box.bbox, box)
    return grid


#: Endpoint-label kinds carry NO own-edge exemption (FIXD03-F13, D03-R5):
#: graphviz paints head/tail labels post-layout beside their own spline near
#: their own arrowhead, so "own spline/arrowhead" is exactly where this
#: class collides -- the blanket exemption made the commonest head-label
#: defect structurally invisible (demonstrated false negative: toy_branchy,
#: a visible own-spline glyph crossing scored 0 by both oracles).
_ENDPOINT_LABEL_KINDS = ("edge-head", "edge-tail")


def _is_own_pair(box: TextBox, item: object) -> bool:
    """True when ``item`` is ``box``'s own element (the OWN exemption).

    A node's own label, an edge's own MIDPOINT label vs its spline, and
    same-owner text stay exempt; an edge-head/edge-tail label is NEVER
    exempt from its own spline or arrowheads (see
    ``_ENDPOINT_LABEL_KINDS``).
    """

    if isinstance(item, TextBox):
        return item is box or (item.owner == box.owner and item.kind == box.kind)
    if isinstance(item, NodeRecord):
        return box.kind in ("node-label", "legend-text", "port-cell") and box.owner == item.name
    if isinstance(item, EdgeRecord):
        return box.kind not in _ENDPOINT_LABEL_KINDS and box.owner == item.name
    arrow_edge = cast("tuple[EdgeRecord, int, tuple[Point, ...]]", item)[0]
    return box.kind not in _ENDPOINT_LABEL_KINDS and box.owner == arrow_edge.name


def _classify_pair(box: TextBox, item: object) -> tuple[str, str, float] | None:
    """Classify one grid-candidate pair: (violation kind, other name, signed gap).

    Returns ``None`` for self-pairs (see ``_is_own_pair``). Outline gaps
    subtract ``penwidth / 2`` so a fat stroke collides honestly.
    """

    if _is_own_pair(box, item):
        return None
    if isinstance(item, TextBox):
        gap = _rect_rect_signed(box.bbox, item.bbox)
        return "text-text", f"{item.kind} {item.owner} '{item.text}'", gap
    if isinstance(item, NodeRecord):
        gap = _points_rect_signed(list(item.outline), box.bbox) - item.penwidth / 2.0
        return "text-node", item.name, gap
    if isinstance(item, EdgeRecord):
        gap = _points_rect_signed(list(item.spline_points), box.bbox)
        return "text-spline", item.name, gap
    arrow_edge, arrow_index, arrow_outline = cast("tuple[EdgeRecord, int, tuple[Point, ...]]", item)
    gap = _points_rect_signed(list(arrow_outline), box.bbox)
    return "text-arrowhead", f"{arrow_edge.name} arrow#{arrow_index}", gap


def audit_layout(parsed: ParsedLayout, pen_eps: float = PEN_EPS) -> AuditResult:
    """Run the widened collision audit over one parsed layout.

    Every text box is checked against foreign nodes, splines, arrowheads,
    cluster borders, and other text boxes -- through the grid prefilter, with
    penwidth-inflated outlines. Self-pairs are exempt: a node's own label, a
    cluster's own caption ON its border row, an edge's MIDPOINT label against
    its own spline/arrowheads. Head/tail labels get NO own-edge exemption
    (D03-R5): they are painted beside their own spline near their own
    arrowhead, so an own-spline crossing there is a real placement defect,
    not expected contact.
    """

    grid = _build_grid(parsed)

    violations: list[AuditViolation] = []
    seen_pairs: set[tuple[str, str, str]] = set()
    candidate_pairs = 0
    checked_pairs = 0

    def _record(kind: str, box: TextBox, other: str, gap: float) -> None:
        """De-duplicated violation append."""

        key = (kind, f"{box.owner}/{box.text}", other)
        if key not in seen_pairs:
            seen_pairs.add(key)
            violations.append(
                AuditViolation(
                    kind=kind,
                    text_owner=f"{box.kind} {box.owner}",
                    text=box.text,
                    other=other,
                    signed_gap=round(gap, 3),
                )
            )

    for box in parsed.textboxes:
        for _bbox, item in grid.query(box.bbox, margin=pen_eps + 8.0):
            candidate_pairs += 1
            pair = _classify_pair(box, item)
            if pair is None:
                continue
            checked_pairs += 1
            kind, other, gap = pair
            if gap < -pen_eps:
                _record(kind, box, other, gap)
        # Cluster borders are few; check directly (they are page-scale, a grid
        # gives them no selectivity).
        for cluster in parsed.clusters:
            if box.kind == "cluster-caption" and box.owner == cluster.name:
                continue
            checked_pairs += 1
            gap = _border_signed(box.bbox, cluster.rect) - cluster.penwidth / 2.0
            if gap < -pen_eps:
                _record("text-clusterborder", box, cluster.name, gap)

    return AuditResult(
        engine=parsed.engine,
        violations=violations,
        class_counts=parsed.class_counts(),
        text_multiset=parsed.text_multiset(),
        candidate_pairs=candidate_pairs,
        checked_pairs=checked_pairs,
    )


def _border_signed(r: Rect, cb: Rect) -> float:
    """Signed distance of a rect to a cluster border OUTLINE (v1 semantics).

    Fully inside or outside is fine (positive gap to the border line);
    crossing returns -(max depth of any border sample inside the rect).
    """

    corners = [(cb[0], cb[1]), (cb[2], cb[1]), (cb[2], cb[3]), (cb[0], cb[3])]
    samples: list[Point] = []
    per_side = 64
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        for j in range(per_side):
            t = j / per_side
            samples.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
    pen = 0.0
    gap = float("inf")
    crossed = False
    for p in samples:
        d = _depth_in_rect(p, r)
        if d >= 0.0:
            crossed = True
            pen = max(pen, d)
        else:
            gap = min(gap, _point_rect_dist(p, r))
    return -pen if crossed else gap


def edges_with_competing_labels(
    parsed: ParsedLayout,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return edges whose text rides MORE THAN ONE label channel.

    The wave-4 ``EdgeAnnotationPlan`` seam (shed-tail NOW-plumbing): today
    the only thing deciding label composition is which Graphviz attribute is
    written last -- naive midpointing DELETED a live conditional-arm ``IF``
    label on stock gpt2 while the v1 audit scored the result clean (memo
    D11). Until the typed plan lands, any widening that touches edges
    reported here must prove the multiset unchanged (``text_multiset``);
    the plan itself plugs in as the ONE composer for these channels.
    """

    channels: defaultdict[str, set[str]] = defaultdict(set)
    for box in parsed.textboxes:
        if box.kind in ("edge-head", "edge-tail", "edge-midpoint"):
            channels[box.owner].add(box.kind)
    return tuple(
        (owner, tuple(sorted(kinds))) for owner, kinds in sorted(channels.items()) if len(kinds) > 1
    )


# ---------------------------------------------------------------------------
# usability envelopes (item 20)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UsabilityEnvelope:
    """The second oracle: page-scale usability metrics (memo D2).

    Attributes are plain measurements; tests calibrate them in both
    directions (a known-bad must fail, a known-good must pass with headroom;
    never tuned-tight).
    """

    engine: str
    canvas_width: float
    canvas_height: float
    canvas_area: float
    canvas_aspect: float
    ink_coverage: float
    spine_drift: float
    min_text_size: float
    legend_ratio: float
    arrowhead_fused_pairs: int
    arrowhead_min_gap: float
    caption_distance_max: float
    caption_distances: tuple[tuple[str, float], ...]
    cluster_width_ratio_max: float
    cluster_width_ratios: tuple[tuple[str, float], ...]


def _ink_coverage(parsed: ParsedLayout, bb: Rect) -> float:
    """Grid-occupancy ink coverage (deterministic, resolution-bounded)."""

    x0, y0, x1, y1 = bb
    width = max(x1 - x0, 1e-9)
    height = max(y1 - y0, 1e-9)
    cell = max(min(width, height) / 40.0, 1.0)
    occupied: set[tuple[int, int]] = set()
    for node in parsed.nodes:
        bx0, by0, bx1, by1 = node.bbox
        for cell_x in range(int(bx0 // cell), int(bx1 // cell) + 1):
            for cell_y in range(int(by0 // cell), int(by1 // cell) + 1):
                occupied.add((cell_x, cell_y))
    total_cells = max(
        (int(x1 // cell) - int(x0 // cell) + 1) * (int(y1 // cell) - int(y0 // cell) + 1),
        1,
    )
    return len(occupied) / total_cells


def _spine_drift(parsed: ParsedLayout, width: float, height: float) -> float:
    """Cross-axis centroid wander of rank bands, as a fraction of the extent."""

    vertical = height >= width
    bands: defaultdict[int, list[float]] = defaultdict(list)
    for node in parsed.nodes:
        center_x = (node.bbox[0] + node.bbox[2]) / 2.0
        center_y = (node.bbox[1] + node.bbox[3]) / 2.0
        if vertical:
            bands[int(center_y // 72.0)].append(center_x)
        else:
            bands[int(center_x // 72.0)].append(center_y)
    centroids = [sum(vals) / len(vals) for vals in bands.values() if vals]
    cross_extent = width if vertical else height
    if len(centroids) <= 1:
        return 0.0
    return (max(centroids) - min(centroids)) / cross_extent


def _legend_ratio(parsed: ParsedLayout, width: float) -> float:
    """Legend footprint width over content width (0.0 when no legend)."""

    legend_boxes = [node.bbox for node in parsed.nodes if node.name.startswith(LEGEND_NODE_MARKER)]
    if not legend_boxes:
        return 0.0
    legend_bbox = legend_boxes[0]
    for extra in legend_boxes[1:]:
        legend_bbox = _rect_union(legend_bbox, extra)
    legend_width = legend_bbox[2] - legend_bbox[0]
    return legend_width / max(width - legend_width, 1e-9)


def _arrowhead_metrics(parsed: ParsedLayout) -> tuple[int, float]:
    """(fused pair count, min within-node arrowhead gap) -- memo D15."""

    heads_by_node: defaultdict[str, list[Rect]] = defaultdict(list)
    for edge in parsed.edges:
        for outline in edge.arrowheads:
            heads_by_node[edge.head].append(_points_bbox(list(outline)))
    fused_pairs = 0
    min_gap = float("inf")
    for boxes in heads_by_node.values():
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                gap = _rect_rect_signed(boxes[i], boxes[j])
                min_gap = min(min_gap, gap)
                if gap <= 0.0:
                    fused_pairs += 1
    return fused_pairs, (min_gap if min_gap != float("inf") else float("nan"))


def _cluster_members(cluster: ClusterRecord, node_bbox_by_name: dict[str, Rect]) -> list[Rect]:
    """Member node bboxes: declared membership, geometric-containment fallback."""

    members = [
        node_bbox_by_name[name] for name in cluster.member_nodes if name in node_bbox_by_name
    ]
    if members:
        return members
    return [
        bbox
        for bbox in node_bbox_by_name.values()
        if _depth_in_rect(((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0), cluster.rect) > 0
    ]


def _caption_distances(parsed: ParsedLayout) -> tuple[tuple[str, float], ...]:
    """Caption-box distance to the nearest member node, per caption."""

    node_bbox_by_name = {node.name: node.bbox for node in parsed.nodes}
    out: list[tuple[str, float]] = []
    for cluster in parsed.clusters:
        member_boxes = _cluster_members(cluster, node_bbox_by_name)
        if not member_boxes:
            continue
        for caption in cluster.caption_boxes:
            distance = min(max(_rect_rect_signed(caption, member), 0.0) for member in member_boxes)
            out.append((cluster.name, distance))
    return tuple(out)


def _cluster_width_ratios(parsed: ParsedLayout) -> tuple[tuple[str, float], ...]:
    """Cluster box width over the member extent width, per cluster."""

    node_bbox_by_name = {node.name: node.bbox for node in parsed.nodes}
    out: list[tuple[str, float]] = []
    for cluster in parsed.clusters:
        member_boxes = [
            node_bbox_by_name[name] for name in cluster.member_nodes if name in node_bbox_by_name
        ]
        if not member_boxes:
            continue
        extent = member_boxes[0]
        for member in member_boxes[1:]:
            extent = _rect_union(extent, member)
        member_width = max(extent[2] - extent[0], 1e-9)
        out.append((cluster.name, (cluster.rect[2] - cluster.rect[0]) / member_width))
    return tuple(out)


def compute_usability_envelope(parsed: ParsedLayout) -> UsabilityEnvelope:
    """Measure the usability envelope of one parsed layout.

    Composes the per-metric functions above; each metric is independently
    calibrated (known-bad fails, known-good passes with headroom).
    """

    x0, y0, x1, y1 = parsed.bb
    width = max(x1 - x0, 1e-9)
    height = max(y1 - y0, 1e-9)
    fused_pairs, min_gap = _arrowhead_metrics(parsed)
    caption_distances = _caption_distances(parsed)
    width_ratios = _cluster_width_ratios(parsed)
    return UsabilityEnvelope(
        engine=parsed.engine,
        canvas_width=width,
        canvas_height=height,
        canvas_area=width * height,
        canvas_aspect=max(width, height) / max(min(width, height), 1e-9),
        ink_coverage=_ink_coverage(parsed, parsed.bb),
        spine_drift=_spine_drift(parsed, width, height),
        min_text_size=min((box.fontsize for box in parsed.textboxes), default=0.0),
        legend_ratio=_legend_ratio(parsed, width),
        arrowhead_fused_pairs=fused_pairs,
        arrowhead_min_gap=min_gap,
        caption_distance_max=max((d for _, d in caption_distances), default=0.0),
        caption_distances=caption_distances,
        cluster_width_ratio_max=max((r for _, r in width_ratios), default=0.0),
        cluster_width_ratios=width_ratios,
    )
