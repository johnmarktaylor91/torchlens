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
statements of the text the image is rendered from only: they depend on the installed
Graphviz build, so the DOT a drawing returns or saves never carries them.

An argument label (``arg 1``, the head label that names which input of a multi-input
op an edge feeds) gets the same treatment only when Graphviz's own spot for it lands
on something: dot puts a head label a fixed distance below its arrowhead, and when the
edge leaves a module box from the rank below, that spot is the box's title line (a
ResNet block's residual add over its ``downsample`` box).  Module titles count as
obstacles here; an argument label whose default spot is clear keeps it, so drawings
without that collision render exactly as before.
"""

from __future__ import annotations

import math
import re
import subprocess
from collections.abc import Iterable, Sequence
from typing import Any, NamedTuple

from .._io._json import loads_bounded
from ..errors import GraphvizUnavailableError
from . import _render_utils
from ._label_geometry import (
    Point,
    Rect,
    _bb,
    _beziers,
    _bounds,
    _grow,
    _label_lines,
    _outlines,
    _overlaps,
    _point,
    _point_rect_distance,
    _port_tangent,
    _rect_line_gap,
    _rect_poly_gap,
    _rect_rect_gap,
    _ring,
    _sample_bezier,
    _shifted,
    _union,
)
from ._loop_edges import reports_route_failure, without_compass_ports

#: a head or tail label that carries a rolled pass range (``In 2-3``, ``Out 1``)
_PASS_TEXT = re.compile(r"(?<![a-z])(?:headlabel|taillabel)=<<TABLE[^\n]*?>(?:In|Out) [0-9]")
#: an argument-position head label (``arg 1``, ``arg (0, 2)``); placed only when its
#: default spot collides, and only in drawings with a titled module box to collide with
_ARG_TEXT = re.compile(r"(?<![a-z])headlabel=<<TABLE[^\n]*?>arg [0-9(]")
_CLUSTER_LINE = re.compile(r"^\s*subgraph\s+\"?cluster_")
_EDGE_LINE = re.compile(r'^\s*("(?:[^"\\]|\\.)*"|[^\s"\[]+)\s*->\s*("(?:[^"\\]|\\.)*"|[^\s"\[]+)')
_MIN_GAP = 0.5  # (pt) a label nearer than this to anything collides (the gate allows 0.25 overlap)
#: (pt) extra room text keeps from other text: two labels each nudged by padding can
#: land a point or so off their predicted spots on older Graphviz
_TEXT_MARGIN = 1.5
_CLEAR_TARGET = 3.0  # clearance (pt) after which closeness to the node wins
_PORT_LABEL_DISTANCE = 10.0  # Graphviz's PORT_LABEL_DISTANCE
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
        layout = loads_bounded(layout_json)
        scene = _Scene(layout)
    except (ValueError, KeyError, TypeError, IndexError):
        return list(body)
    scene.add_fixed_labels(set(targets))
    new_body = list(body)
    for edge in scene.edges:
        target = targets.get((edge["tail"], edge["head"], edge["occurrence"]))
        if target is None or not edge["labels"]:
            continue
        line_index, kind = target
        if kind == "arg" and not scene.default_collides(edge):
            scene.keep_default(edge)  # an argument label moves only off a module title
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

    if not has_pass_labels(source.splitlines()):
        return source
    routed = _routed_layout(source, engine=engine, timeout=timeout, cwd=cwd)
    if routed is None:
        return source
    source, layout = routed
    return "".join(choose_pass_label_placements(source.splitlines(keepends=True), layout))


def _routed_layout(
    source: str, *, engine: str, timeout: float, cwd: str | None
) -> tuple[str, str] | None:
    """Return the text to render and its ``-Tjson`` layout, or ``None`` when layout fails.

    When the installed Graphviz reports a spline-router failure for ``source`` (2.43 on
    the README loop's corner ports, :mod:`._loop_edges`), the text without edge compass
    ports is laid out instead and used only if that layout routes cleanly. When Graphviz
    aborted on the ported text, any portless layout it completes is used, since the
    ported text draws nothing; ``None`` when neither lays out.
    """

    completed = _layout_run(source, engine=engine, timeout=timeout, cwd=cwd)
    if completed is None:
        return None
    aborted = completed.returncode != 0
    if reports_route_failure(completed.stderr):
        portless = without_compass_ports(source)
        retry = _layout_run(portless, engine=engine, timeout=timeout, cwd=cwd)
        if retry is not None and retry.returncode == 0 and portless != source:
            if aborted or not reports_route_failure(retry.stderr):
                return portless, _stdout_text(retry)
    return None if aborted else (source, _stdout_text(completed))


def forward_sources(
    dot: Any, source_override: str | None, trace: Any, timeout: float
) -> tuple[str, str]:
    """Return the forward render's DOT and the text its image is rendered from.

    The DOT is ``source_override`` (the sibling-ordering verifier's source when it
    rewrote the graph) or ``dot.source``; it is what a drawing returns, identical under
    every Graphviz build or none.  Only the image text carries the pass-label
    placement, which reads the installed Graphviz's layout.  Graphviz runs in the
    trace's visualizer directory, so the extra layout resolves relative image paths
    the same way the render does.
    """

    source = source_override if source_override is not None else dot.source
    cwd = trace._visualizer_dir
    placed = place_pass_labels(
        source, engine=dot.engine, timeout=timeout, cwd=str(cwd) if cwd else None
    )
    return source, placed


def _layout_run(
    source: str, *, engine: str, timeout: float, cwd: str | None
) -> subprocess.CompletedProcess[Any] | None:
    """Run ``engine -Tjson`` on ``source``; ``None`` when Graphviz fails or is missing.

    A run Graphviz aborts on a spline-router failure (2.43 can exit 1 with ``lost ...
    edge`` after ``Pshortestpath failed``) comes back failed (nonzero ``returncode``, no
    layout) so the caller can lay the text out again without compass ports.
    """

    try:
        return _render_utils.run_bounded_subprocess(
            [engine, "-Tjson"], input=source.encode("utf-8"), timeout=timeout, cwd=cwd
        )
    except subprocess.CalledProcessError as error:
        if not reports_route_failure(error.stderr):
            return None
        return subprocess.CompletedProcess(error.cmd, error.returncode, b"", error.stderr)
    except (subprocess.TimeoutExpired, GraphvizUnavailableError):
        return None


def _stdout_text(completed: subprocess.CompletedProcess[Any]) -> str:
    """Return a Graphviz child's stdout as text."""

    stdout = completed.stdout
    return stdout.decode("utf-8", "replace") if isinstance(stdout, bytes) else str(stdout)


def has_pass_labels(body: Iterable[str]) -> bool:
    """Return whether the statements carry a label the placement pass may move.

    A rolled ``In``/``Out`` head or tail label anywhere, or an ``arg`` head label in a
    drawing that also has a module box (``subgraph cluster_...``) whose title it could
    land on.  A drawing with argument labels and no module box keeps Graphviz's own
    spots without the extra layout.
    """

    lines = list(body)
    if any(_PASS_TEXT.search(line) for line in lines):
        return True
    return any(_CLUSTER_LINE.match(line) for line in lines) and any(
        _ARG_TEXT.search(line) for line in lines
    )


def _pass_label_lines(body: Sequence[str]) -> dict[tuple[str, str, int], tuple[int, str]]:
    """Map ``(tail, head, occurrence)`` of each labelled edge line to its index and kind.

    The kind is ``"pass"`` for a rolled ``In``/``Out`` label (always placed) and
    ``"arg"`` for an argument label (placed only when its default spot collides);
    argument labels are collected only when the body has a module box.
    """

    lines = list(body)
    with_clusters = any(_CLUSTER_LINE.match(line) for line in lines)
    seen: dict[tuple[str, str], int] = {}
    targets: dict[tuple[str, str, int], tuple[int, str]] = {}
    for index, line in enumerate(lines):
        match = _EDGE_LINE.match(line)
        if match is None:
            continue
        pair = (_unquote(match.group(1)), _unquote(match.group(2)))
        occurrence = seen.get(pair, 0)
        seen[pair] = occurrence + 1
        if _PASS_TEXT.search(line):
            targets[(*pair, occurrence)] = (index, "pass")
        elif with_clusters and _ARG_TEXT.search(line):
            targets[(*pair, occurrence)] = (index, "arg")
    return targets


def _unquote(name: str) -> str:
    """Return a DOT node name without its port suffix and surrounding quotes."""
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
        """Wrap the matched HTML label in a table padded on the chosen sides."""
        return (
            f'{kind}=<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0">'
            f"{above}<TR>{left}<TD>{match.group(1)}</TD>{right}</TR>{below}</TABLE>>"
        )

    return re.sub(_LABEL_HTML.format(kind=kind), wrap, line, count=1)


def _with_placement(line: str, choice: tuple[float, float]) -> str:
    """Append the chosen ``labelangle`` and ``labeldistance`` to an edge statement."""
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
        # a module box's title line is text a label must not cover
        self.titles: list[Rect] = [
            r for obj in objects[:sub_count] for r in _label_lines(obj.get("_ldraw_"))
        ]
        self.rects: list[Rect] = list(self.titles)
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
        """Record one laid-out edge: its spline, arrowheads, label boxes and end-label candidates."""
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

    def default_collides(self, edge: dict[str, Any]) -> bool:
        """Return whether Graphviz's own spot for one of the edge's labels covers a module title.

        Only the title collision moves an argument label: the trigger stays narrow so
        every drawing without it keeps Graphviz's own label spots.
        """

        return any(
            _overlaps(_grow(rect, _MIN_GAP), title)
            for label in edge["labels"]
            for rect in label["default"]
            for title in self.titles
        )

    def keep_default(self, edge: dict[str, Any]) -> None:
        """Record the edge's labels at Graphviz's own spots as obstacles for later edges."""

        self.rects.extend(r for label in edge["labels"] for r in label["default"])

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
