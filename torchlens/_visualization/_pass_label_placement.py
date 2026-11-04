"""Place rolled ``In``/``Out`` pass labels beside their nodes, clear of everything else.

A rolled edge names the passes it leaves its tail on (``Out``, the tail label) and
enters its head on (``In``, the head label), so each sits at the node it describes.
Graphviz reserves no layout space for head and tail labels, and one
``labeldistance``/``labelangle`` pair moves both ends of an edge, so no fixed
placement keeps them off nodes, lines, arrowheads, other labels and module borders
in every drawing and direction.  Head and tail labels never move nodes or edges,
though, so one ``dot -Tjson`` pass gives the final geometry: each pass-labelled edge
takes the candidate placement whose labels clear every obstacle and sit at the end
of their own edge, clearly nearer it than any other edge at that node.  A label the
shared pair still leaves colliding, astray or ambiguous moves on its own by blank
padding on one side of its HTML table.  The choices are written onto the edge
statement before the real render.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
from collections.abc import Iterable, Sequence
from typing import Any, NamedTuple

from . import _render_utils
from ._render_common import GraphvizUnavailableError

Rect = tuple[float, float, float, float]
Point = tuple[float, float]

#: a head or tail label that carries a rolled pass range (``In 2-3``, ``Out 1``)
_PASS_TEXT = re.compile(r"(?<![a-z])(?:headlabel|taillabel)=<<TABLE[^\n]*?>(?:In|Out) [0-9]")
_EDGE_LINE = re.compile(r'^\s*("(?:[^"\\]|\\.)*"|[^\s"\[]+)\s*->\s*("(?:[^"\\]|\\.)*"|[^\s"\[]+)')
_ASCENT, _DESCENT = 0.78, 0.22  # glyph box model of tests/support/label_geometry.py
_MIN_GAP = 0.5  # (pt) a label nearer than this to anything collides (the gate allows 0.25 overlap)
#: (pt) extra room text keeps from other text: two labels each nudged by padding can
#: land a point or so off their predicted spots on older Graphviz
_TEXT_MARGIN = 1.5
_CLEAR_TARGET = 3.0  # clearance (pt) after which closeness to the node wins
_PORT_LABEL_DISTANCE = 10.0  # Graphviz's PORT_LABEL_DISTANCE
_REACH = 90.0  # (pt) the gap reported when there is nothing to measure against
_NEAR_END = 12.0  # (pt) a label farther than this from its own edge's end no longer touches it
_ATTACH_REACH = 30.0  # (pt) the stretch of a label's own edge, from its end, it reads against
_OTHER_REACH = 60.0  # (pt) other edges at the same node within this of the end compete for it
_ATTACH_MARGIN = 4.0  # (pt) a label must be this much nearer its own edge than any other there
#: outward normals of a node's sides; a label never reaches past a side another edge meets
_SIDES: dict[str, Point] = {
    "left": (-1.0, 0.0),
    "right": (1.0, 0.0),
    "bottom": (0.0, -1.0),
    "top": (0.0, 1.0),
}
_ON_SIDE = 2.0  # (pt) an end this close to a side of its node's box is on that side
_AT_NODE = 16.0  # (pt) an edge end (arrowhead base included) this close to a node box is at it
_SAMPLE_STEP = 1.5  # (pt) spacing of outline and spline samples
_CHUNK = 12  # samples per indexed piece of a line, so a distant stretch is skipped whole
#: (labeldistance, labelangle) pairs tried, nearest first; angles near 0 sit on the line
_CANDIDATES: tuple[tuple[float, float], ...] = tuple(
    (distance, float(angle))
    for distance in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0)
    for angle in range(-165, 166, 15)
    if abs(angle) > 15
)
#: (pt) distances from its edge end at which a label that still collides or strays is
#: tried on its own, every 15 degrees, by padding its HTML table on one side
_NUDGE_RADII = (8.0, 11.0, 14.0, 17.0, 20.0, 24.0, 28.0, 34.0, 40.0)
_LABEL_HTML = r"(?<![a-z]){kind}=<(<TABLE.*?</TABLE>)>"


def choose_pass_label_placements(body: list[str], layout_json: str) -> list[str]:
    """Return ``body`` with placement attributes on pass-labelled edges that need them.

    Parameters
    ----------
    body:
        The Graphviz object's statement lines, in source order.
    layout_json:
        ``dot -Tjson`` output for exactly these statements.

    Returns
    -------
    list[str]
        A new list; every pass-labelled edge line gains ``labeldistance`` and
        ``labelangle``.  Unparseable layouts leave ``body`` as is.
    """

    targets = _pass_label_lines(body)
    if not targets:
        return list(body)
    try:
        layout = json.loads(layout_json)
        scene = _Scene(layout)
    except (ValueError, KeyError, TypeError, IndexError):
        return list(body)
    scene.add_fixed_labels(set(targets))
    new_body = list(body)
    for edge in scene.edges:
        line_index = targets.get((edge["tail"], edge["head"], edge["occurrence"]))
        if line_index is None or not edge["labels"]:
            continue
        choice, shifts = scene.choose(edge)
        line = _with_placement(new_body[line_index], choice)
        for label, shift in zip(edge["labels"], shifts, strict=True):
            line = _shift_label(line, f"{label['kind']}label", shift)
        new_body[line_index] = line
    return new_body


def place_pass_labels(source: str, *, engine: str, timeout: float, cwd: str | None = None) -> str:
    """Return ``source`` with each rolled pass label pair placed clear of the drawing.

    Parameters
    ----------
    source:
        Complete DOT source about to be rendered.
    engine:
        Graphviz layout engine of the render.
    timeout:
        Seconds the extra layout may take.
    cwd:
        Working directory of the render (relative image paths resolve against it).

    Returns
    -------
    str
        ``source`` unchanged when it has no pass labels or the layout fails (the real
        render then reports the failure); otherwise with placement attributes added.
    """

    lines = source.splitlines(keepends=True)
    if not has_pass_labels(lines):
        return source
    layout = pass_label_layout(source, engine=engine, timeout=timeout, cwd=cwd)
    if layout is None:
        return source
    return "".join(choose_pass_label_placements(lines, layout))


def place_forward_pass_labels(
    dot: Any, source_override: str | None, trace: Any, timeout: float
) -> str:
    """Return the forward render's final source with its pass labels placed.

    ``source_override`` is the sibling-ordering verifier's source when it rewrote
    the graph; Graphviz runs in the trace's visualizer directory, so the extra
    layout resolves relative image paths the same way the render does.
    """

    source = source_override if source_override is not None else dot.source
    cwd = getattr(trace, "_visualizer_dir", None)
    return place_pass_labels(
        source, engine=dot.engine, timeout=timeout, cwd=str(cwd) if cwd else None
    )


def pass_label_layout(
    source: str, *, engine: str, timeout: float, cwd: str | None = None
) -> str | None:
    """Return ``engine -Tjson`` output for ``source``, or ``None`` when the layout fails.

    A missing Graphviz executable is ``None`` too: DOT source is still written
    without one, and a real render reports the missing binary itself.
    """

    try:
        completed = _render_utils.run_bounded_subprocess(
            [engine, "-Tjson"], input=source.encode("utf-8"), timeout=timeout, cwd=cwd
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, GraphvizUnavailableError):
        return None
    stdout = completed.stdout
    return stdout.decode("utf-8", "replace") if isinstance(stdout, bytes) else str(stdout)


def has_pass_labels(body: Iterable[str]) -> bool:
    """Return whether any statement line carries a rolled ``In``/``Out`` head or tail label."""

    return any(_PASS_TEXT.search(line) for line in body)


def _pass_label_lines(body: Sequence[str]) -> dict[tuple[str, str, int], int]:
    """Map ``(tail, head, occurrence)`` of each pass-labelled edge line to its index."""

    seen: dict[tuple[str, str], int] = {}
    targets: dict[tuple[str, str, int], int] = {}
    for index, line in enumerate(body):
        match = _EDGE_LINE.match(line)
        if match is None:
            continue
        pair = (_unquote(match.group(1)), _unquote(match.group(2)))
        occurrence = seen.get(pair, 0)
        seen[pair] = occurrence + 1
        if _PASS_TEXT.search(line):
            targets[(*pair, occurrence)] = index
    return targets


def _unquote(name: str) -> str:
    name = name.split(":", 1)[0] if not name.startswith('"') else name
    if name.startswith('"') and name.endswith('"'):
        return name[1:-1].replace('\\"', '"')
    return name


def _shift_label(line: str, kind: str, shift: Point) -> str:
    """Move one HTML head/tail label's text by ``shift`` (pt, y up) with blank padding.

    One ``labeldistance``/``labelangle`` pair places both labels of an edge; padding
    a label's table on the side away from where its text should go moves that text
    alone, by half the padding, without changing the layout.
    """

    width, height = round(abs(shift[0]) * 2), round(abs(shift[1]) * 2)
    if not width and not height:
        return line
    left = f'<TD WIDTH="{width}"></TD>' if width and shift[0] > 0 else ""
    right = f'<TD WIDTH="{width}"></TD>' if width and shift[0] < 0 else ""
    columns = 2 if width else 1
    pad_row = f'<TR><TD COLSPAN="{columns}" HEIGHT="{height}"></TD></TR>' if height else ""
    above, below = (pad_row, "") if shift[1] < 0 else ("", pad_row)

    def wrap(match: re.Match[str]) -> str:
        return (
            f'{kind}=<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0">'
            f"{above}<TR>{left}<TD>{match.group(1)}</TD>{right}</TR>{below}</TABLE>>"
        )

    return re.sub(_LABEL_HTML.format(kind=kind), wrap, line, count=1)


def _with_placement(line: str, choice: tuple[float, float]) -> str:
    stripped = line.rstrip("\n")
    newline = line[len(stripped) :]
    distance, angle = choice
    return f"{stripped[:-1]} labelangle={angle:g} labeldistance={distance:g}]{newline}"


class _Scene:
    """Obstacles and pass-labelled edges of one laid-out graph (points, y up)."""

    def __init__(self, layout: dict[str, Any]) -> None:
        # dot places port labels before it turns the drawing to ``rankdir``; bottom-up
        # and right-to-left are reflections of top-down and left-to-right, so a
        # ``labelangle`` turns the other way in the final drawing there.
        self.turn = -1.0 if str(layout.get("rankdir", "TB")).upper() in ("BT", "RL") else 1.0
        objects = layout.get("objects", [])
        sub_count = int(layout.get("_subgraph_cnt", 0))
        names = {obj["_gvid"]: obj.get("name", "") for obj in objects}
        self.nodes: dict[str, list[list[Point]]] = {}
        for obj in objects[sub_count:]:
            outlines = _outlines(obj.get("_draw_"))
            if outlines:
                self.nodes[obj.get("name", "")] = outlines
        self.borders = [_ring(_bb(obj["bb"])) for obj in objects[:sub_count] if obj.get("bb")]
        self.splines: list[list[Point]] = []
        self.polygons: list[list[Point]] = [o for outs in self.nodes.values() for o in outs]
        self.rects: list[Rect] = []
        self.edges: list[dict[str, Any]] = []
        seen: dict[tuple[str, str], int] = {}
        for edge in layout.get("edges", []):
            self._add_edge(edge, names, seen)
        self.indexed_polygons = [(_bounds(p), p) for p in self.polygons]
        self.indexed_lines = [
            (_bounds(chunk), chunk)
            for line in (*self.splines, *self.borders)
            for start in range(0, max(1, len(line) - 1), _CHUNK)
            if (chunk := line[start : start + _CHUNK + 1])
        ]
        self._attach_lines()

    def _add_edge(
        self, edge: dict[str, Any], names: dict[int, str], seen: dict[tuple[str, str], int]
    ) -> None:
        tail, head = names.get(edge["tail"], ""), names.get(edge["head"], "")
        occurrence = seen.get((tail, head), 0)
        seen[(tail, head)] = occurrence + 1
        samples = [p for ctrl in _beziers(edge.get("_draw_")) for p in _sample_bezier(ctrl)]
        self.splines.extend(_sample_bezier(ctrl) for ctrl in _beziers(edge.get("_draw_")))
        for key in ("_hdraw_", "_tdraw_"):
            self.polygons.extend(_outlines(edge.get(key)))
        self.rects.extend(_label_lines(edge.get("_ldraw_")))
        labels = []
        for kind, key, lp_key in (("head", "_hldraw_", "head_lp"), ("tail", "_tldraw_", "tail_lp")):
            lines = _label_lines(edge.get(key))
            if lines and edge.get(lp_key) and edge.get("pos"):
                end, inner = _port_tangent(edge["pos"], head=kind == "head")
                centre = _point(edge[lp_key])
                offsets = [
                    (r[0] - centre[0], r[1] - centre[1], r[2] - centre[0], r[3] - centre[1])
                    for r in lines
                ]
                labels.append(
                    {
                        "kind": kind,
                        "default": lines,
                        "offsets": offsets,
                        "end": end,
                        "inner": inner,
                        "node": head if kind == "head" else tail,
                    }
                )
            else:
                self.rects.extend(lines)
        self.edges.append(
            {
                "tail": tail,
                "head": head,
                "occurrence": occurrence,
                "labels": labels,
                "samples": samples,
            }
        )

    def _attach_lines(self) -> None:
        """Give each label its own edge's last stretch and the other edges at its node."""

        for edge in self.edges:
            for label in edge["labels"]:
                end, node = label["end"], label["node"]
                label["own_line"] = [
                    p for p in edge["samples"] if math.dist(p, end) <= _ATTACH_REACH
                ] or [end]
                ahead = min(label["own_line"], key=lambda p: abs(math.dist(p, end) - 15.0))
                label["box"] = _union(
                    [_bounds(o) for o in self.nodes.get(node, [])] or [(*end, *end)]
                )
                used = {
                    _nearest_side(label["box"], point)
                    for other in self.edges
                    if other is not edge
                    and other["samples"]
                    and node in (other["tail"], other["head"])
                    for point in (other["samples"][0], other["samples"][-1])
                    if _point_rect_distance(point, label["box"]) <= _AT_NODE
                }
                heading = (ahead[0] - end[0], ahead[1] - end[1])
                label["side"] = _side(label["box"], end, heading, used)
                label["taken"] = used - {label["side"]}
                label["other_lines"] = [
                    near
                    for other in self.edges
                    if other is not edge and node in (other["tail"], other["head"])
                    if (near := [p for p in other["samples"] if math.dist(p, end) <= _OTHER_REACH])
                ]

    def choose(self, edge: dict[str, Any]) -> tuple[tuple[float, float], list[Point]]:
        """Pick the edge's placement and record its labels as obstacles for later edges.

        Always an explicit pair: Graphviz's default spot for an unplaced label is
        chosen around the other labels, so it moves when they move and cannot be
        trusted from this layout.  A label the shared pair leaves colliding or far
        from its node then gets its own offset (one per label, in label order).
        """

        placements = [
            ((distance, angle), [self._placed(label, distance, angle) for label in edge["labels"]])
            for distance, angle in _CANDIDATES
        ]
        local = self._near(_union([r for _, boxes in placements for b in boxes for r in b]))
        choice, chosen = min(placements, key=lambda item: self._score(edge, item[1], local))
        boxes = list(chosen)
        shifts = [(0.0, 0.0)] * len(boxes)
        for _ in range(2):  # a second round lets an earlier label make way for a later one
            for index, label in enumerate(edge["labels"]):
                others = [r for j, lines in enumerate(boxes) if j != index for r in lines]
                dx, dy = self._nudge(label, boxes[index], others, local)
                boxes[index] = _shifted(boxes[index], (dx, dy))
                shifts[index] = (shifts[index][0] + dx, shifts[index][1] + dy)
        for lines in boxes:
            self.rects.extend(lines)
        return choice, shifts

    def _near(self, region: Rect) -> _Obstacles:
        """Return the obstacles that can come within ``_CLEAR_TARGET`` of ``region``."""

        grown = _grow(region, _CLEAR_TARGET + 1.0)
        return _Obstacles(
            [item for item in self.indexed_polygons if _overlaps(grown, item[0])],
            [item for item in self.indexed_lines if _overlaps(grown, item[0])],
            [r for r in self.rects if _overlaps(grown, r)],
        )

    def _nudge(
        self, label: dict[str, Any], lines: list[Rect], others: list[Rect], local: _Obstacles
    ) -> Point:
        """Return the offset that moves one label clear and beside its node, if it needs one."""

        single = {"labels": [label]}
        best = (self._score(single, [lines], local, others), (0.0, 0.0))
        if best[0][0] == 0.0 and not best[0][1] and not best[0][2]:
            return best[1]
        (ex, ey), (x0, y0, x1, y1) = label["end"], _union(lines)
        for radius in _NUDGE_RADII:
            for step in range(24):
                phi = step * math.pi / 12
                shift = (
                    round((ex + radius * math.cos(phi) - (x0 + x1) / 2) * 2) / 2,
                    round((ey + radius * math.sin(phi) - (y0 + y1) / 2) * 2) / 2,
                )
                score = self._score(single, [_shifted(lines, shift)], local, others)
                if score < best[0]:
                    best = (score, shift)
        return best[1]

    def _placed(self, label: dict[str, Any], distance: float, angle: float) -> list[Rect]:
        """Return the label's text boxes for one ``labeldistance``/``labelangle`` pair."""

        (ex, ey), (fx, fy) = label["end"], label["inner"]
        theta = math.atan2(fy - ey, fx - ex) + self.turn * math.radians(angle)
        cx = ex + _PORT_LABEL_DISTANCE * distance * math.cos(theta)
        cy = ey + _PORT_LABEL_DISTANCE * distance * math.sin(theta)
        return [(cx + o[0], cy + o[1], cx + o[2], cy + o[3]) for o in label["offsets"]]

    def _score(
        self,
        edge: dict[str, Any],
        boxes: list[list[Rect]],
        local: _Obstacles,
        extra: Sequence[Rect] = (),
    ) -> tuple[float, bool, bool, float, float]:
        """Rank one placement of an edge's labels; lower is better.

        Collision depth first; then whether a label reads as belonging to another edge
        at its node (it is not clearly nearer its own edge's last stretch than every
        other edge there, or it reaches past a side of the node another edge meets);
        then whether it is more than ``_NEAR_END`` from its own edge's end; then
        clearance short of ``_CLEAR_TARGET`` (half-point steps); then the farthest
        distance from a label to its own edge's end.
        """

        depth, clearance, end_gap = 0.0, math.inf, 0.0
        ambiguous = False
        for label, lines in zip(edge["labels"], boxes, strict=True):
            others = [r for other in boxes if other is not lines for r in other] + list(extra)
            for rect in lines:
                gap = _clearance(rect, others, local)
                depth += max(0.0, _MIN_GAP - gap)
                clearance = min(clearance, gap)
            end_gap = max(end_gap, min(_point_rect_distance(label["end"], r) for r in lines))
            own = min(_rect_line_gap(r, label["own_line"]) for r in lines)
            rival = min(
                (_rect_line_gap(r, line) for r in lines for line in label["other_lines"]),
                default=math.inf,
            )
            ambiguous = ambiguous or rival < own + _ATTACH_MARGIN or _wrong_side(label, lines)
        shortfall = math.ceil(max(0.0, _CLEAR_TARGET - clearance) * 2) / 2
        return (round(depth, 3), ambiguous, end_gap > _NEAR_END, shortfall, round(end_gap, 3))

    def add_fixed_labels(self, targeted: set[tuple[str, str, int]]) -> None:
        """Count the head/tail labels of edges that are not being placed as obstacles."""

        for edge in self.edges:
            if (edge["tail"], edge["head"], edge["occurrence"]) not in targeted:
                self.rects.extend(r for label in edge["labels"] for r in label["default"])


class _Obstacles(NamedTuple):
    """Obstacles near one edge's candidate labels, each line and outline with its bounds."""

    polygons: list[tuple[Rect, list[Point]]]
    lines: list[tuple[Rect, list[Point]]]
    rects: list[Rect]


def _clearance(rect: Rect, extra: Iterable[Rect], local: _Obstacles) -> float:
    """Gap from ``rect`` to its nearest obstacle, or ``_CLEAR_TARGET`` when none is that close.

    Only gaps below ``_CLEAR_TARGET`` change a score, so obstacles whose bounds stay
    farther away are skipped.
    """

    probe = _grow(rect, _CLEAR_TARGET + _TEXT_MARGIN)
    gaps = [_rect_poly_gap(rect, p) for b, p in local.polygons if _overlaps(probe, b)]
    gaps += [_rect_line_gap(rect, p) for b, p in local.lines if _overlaps(probe, b)]
    gaps += [
        _rect_rect_gap(rect, r) - _TEXT_MARGIN
        for r in (*local.rects, *extra)
        if _overlaps(probe, r)
    ]
    return min(gaps, default=_CLEAR_TARGET)


def _side(box: Rect, end: Point, heading: Point, used: set[str]) -> str:
    """The side of a node's box where its edge meets it.

    The side its end lies on.  At a corner, the touching side no other edge at the node
    uses (a label there cannot be read as that edge's), then the side the edge heads
    away from the node through; an end off the box (a curved outline) counts every side.
    """

    gaps = {
        "left": abs(end[0] - box[0]),
        "right": abs(box[2] - end[0]),
        "bottom": abs(end[1] - box[1]),
        "top": abs(box[3] - end[1]),
    }
    touching = [side for side, gap in gaps.items() if gap <= _ON_SIDE] or list(_SIDES)
    free = [side for side in touching if side not in used] or touching
    return max(free, key=lambda side: _SIDES[side][0] * heading[0] + _SIDES[side][1] * heading[1])


def _nearest_side(box: Rect, point: Point) -> str:
    """The side of a node's box nearest a point just outside it (another edge's end)."""

    gaps = {
        "left": abs(point[0] - box[0]),
        "right": abs(box[2] - point[0]),
        "bottom": abs(point[1] - box[1]),
        "top": abs(box[3] - point[1]),
    }
    return min(gaps, key=gaps.__getitem__)


def _wrong_side(label: dict[str, Any], lines: list[Rect]) -> bool:
    """Whether a label reaches past a side of its node that another edge meets.

    Beside such a side it reads as that edge's label; beyond its own edge's side, or
    beyond a side no other edge meets, it can only be its own edge's.
    """

    x0, y0, x1, y1 = _union(lines)
    nx0, ny0, nx1, ny1 = label["box"]
    beyond = {
        "left": x0 < nx0 - _MIN_GAP,
        "right": x1 > nx1 + _MIN_GAP,
        "bottom": y0 < ny0 - _MIN_GAP,
        "top": y1 > ny1 + _MIN_GAP,
    }
    return any(beyond[side] for side in label["taken"])


def _shifted(lines: list[Rect], shift: Point) -> list[Rect]:
    dx, dy = shift
    return [(r[0] + dx, r[1] + dy, r[2] + dx, r[3] + dy) for r in lines]


def _union(rects: Sequence[Rect]) -> Rect:
    return (
        min(r[0] for r in rects),
        min(r[1] for r in rects),
        max(r[2] for r in rects),
        max(r[3] for r in rects),
    )


def _bb(text: str) -> Rect:
    x0, y0, x1, y1 = (float(v) for v in text.split(","))
    return (x0, y0, x1, y1)


def _point(text: str) -> Point:
    x, y = (float(v) for v in text.split(",")[:2])
    return (x, y)


def _outlines(ops: Any) -> list[list[Point]]:
    outlines: list[list[Point]] = []
    for op in ops or []:
        if op["op"] in ("e", "E"):
            cx, cy, rx, ry = (float(v) for v in op["rect"])
            ellipse = [
                (cx + rx * math.cos(k * math.pi / 32), cy + ry * math.sin(k * math.pi / 32))
                for k in range(64)
            ]
            outlines.append(_densify([*ellipse, ellipse[0]]))
        elif op["op"] in ("p", "P"):
            corners = [(float(x), float(y)) for x, y in op["points"]]
            outlines.append(_densify([*corners, corners[0]]))
        elif op["op"] in ("b", "B"):  # curved shapes (cylinder, rounded box)
            outlines.append(_sample_bezier([(float(x), float(y)) for x, y in op["points"]]))
    return outlines


def _ring(rect: Rect) -> list[Point]:
    x0, y0, x1, y1 = rect
    return _densify([(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)])


def _densify(points: Sequence[Point]) -> list[Point]:
    """Resample a polyline so consecutive samples are at most ``_SAMPLE_STEP`` apart."""

    dense = [points[0]]
    for (x0, y0), (x1, y1) in zip(points, points[1:], strict=False):
        steps = max(1, math.ceil(math.hypot(x1 - x0, y1 - y0) / _SAMPLE_STEP))
        dense += [
            (x0 + (x1 - x0) * k / steps, y0 + (y1 - y0) * k / steps) for k in range(1, steps + 1)
        ]
    return dense


def _beziers(ops: Any) -> list[list[Point]]:
    return [
        [(float(x), float(y)) for x, y in op["points"]]
        for op in ops or []
        if op["op"] in ("b", "B")
    ]


def _label_lines(ops: Any) -> list[Rect]:
    """Glyph boxes of a label's text lines (the geometry gate's box model)."""

    size, rects = 14.0, []
    for op in ops or []:
        if op["op"] == "F":
            size = float(op["size"])
        elif op["op"] == "T":
            x, y = (float(v) for v in op["pt"])
            width = float(op["width"])
            x0 = (
                x - width
                if op.get("align") == "r"
                else x - width / 2
                if op.get("align") == "c"
                else x
            )
            rects.append((x0, y - _DESCENT * size, x0 + width, y + _ASCENT * size))
    return rects


def _port_tangent(pos: str, *, head: bool) -> tuple[Point, Point]:
    """Return Graphviz's ``place_portlabel`` end point and inner point for one end."""

    pieces = pos.split(";")
    tokens = (pieces[-1] if head else pieces[0]).split()
    flag = "e," if head else "s,"
    tip = next((_point(t[2:]) for t in tokens if t.startswith(flag)), None)
    ctrl = [_point(t) for t in tokens if not t.startswith(("e,", "s,"))]
    if head:
        if tip is not None:
            return tip, ctrl[-1]
        return ctrl[-1], _bezier_at(ctrl[-4:], 0.9)
    if tip is not None:
        return tip, ctrl[0]
    return ctrl[0], _bezier_at(ctrl[:4], 0.1)


def _bezier_at(ctrl: Sequence[Point], t: float) -> Point:
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = ctrl
    m = 1.0 - t
    return (
        m**3 * x0 + 3 * m * m * t * x1 + 3 * m * t * t * x2 + t**3 * x3,
        m**3 * y0 + 3 * m * m * t * y1 + 3 * m * t * t * y2 + t**3 * y3,
    )


def _sample_bezier(ctrl: Sequence[Point]) -> list[Point]:
    points = [ctrl[0]]
    for start in range(0, len(ctrl) - 3, 3):
        points += [_bezier_at(ctrl[start : start + 4], k / 16) for k in range(1, 17)]
    return _densify(points)


def _bounds(points: Sequence[Point]) -> Rect:
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _grow(rect: Rect, margin: float) -> Rect:
    return (rect[0] - margin, rect[1] - margin, rect[2] + margin, rect[3] + margin)


def _overlaps(a: Rect, b: Rect) -> bool:
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def _depth(point: Point, rect: Rect) -> float:
    """How far ``point`` lies inside ``rect`` (negative outside)."""

    return min(point[0] - rect[0], rect[2] - point[0], point[1] - rect[1], rect[3] - point[1])


def _point_rect_distance(point: Point, rect: Rect) -> float:
    dx = max(rect[0] - point[0], 0.0, point[0] - rect[2])
    dy = max(rect[1] - point[1], 0.0, point[1] - rect[3])
    return math.hypot(dx, dy)


def _rect_rect_gap(a: Rect, b: Rect) -> float:
    dx = max(b[0] - a[2], a[0] - b[2])
    dy = max(b[1] - a[3], a[1] - b[3])
    if dx > 0 or dy > 0:
        return math.hypot(max(dx, 0.0), max(dy, 0.0))
    return max(dx, dy)


def _rect_line_gap(rect: Rect, points: Sequence[Point]) -> float:
    """Gap from ``rect`` to densely sampled points; negative is the deepest sample inside."""

    inside = max((_depth(p, rect) for p in points), default=-math.inf)
    if inside >= 0.0:
        return -inside
    return min((_point_rect_distance(p, rect) for p in points), default=_REACH)


def _rect_poly_gap(rect: Rect, ring: Sequence[Point]) -> float:
    """Gap from ``rect`` to a closed, densely sampled outline; negative when they overlap."""

    gap = _rect_line_gap(rect, ring)
    corners = ((rect[0], rect[1]), (rect[2], rect[1]), (rect[2], rect[3]), (rect[0], rect[3]))
    inner = [c for c in corners if _inside(c, ring)]
    if not inner:
        return gap
    # A corner inside the shape: the penetration is how far that corner lies from
    # the outline, which can exceed how far the outline reaches into the box.
    depth = max(min(math.hypot(c[0] - p[0], c[1] - p[1]) for p in ring) for c in inner)
    return -max(depth, -gap)


def _inside(point: Point, poly: Sequence[Point]) -> bool:
    x, y = point
    hit = False
    for (x1, y1), (x2, y2) in zip(poly, [*poly[1:], poly[0]], strict=True):
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            hit = not hit
    return hit
