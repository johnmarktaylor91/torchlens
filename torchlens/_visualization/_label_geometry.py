"""Plane geometry of a laid-out Graphviz drawing (points, y up), for label placement.

Parsers for ``dot -Tjson`` draw ops (node outlines, edge splines, label text boxes,
port tangents) and the box, line and outline gap measures the placement pass scores
candidate label spots with.  Pure functions on tuples; nothing here reads the drawing.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

Rect = tuple[float, float, float, float]
Point = tuple[float, float]

_ASCENT, _DESCENT = 0.78, 0.22  # glyph box model of tests/support/label_geometry.py
_REACH = 90.0  # (pt) the gap reported when there is nothing to measure against
_SAMPLE_STEP = 1.5  # (pt) spacing of outline and spline samples


def _shifted(lines: list[Rect], shift: Point) -> list[Rect]:
    """Return ``lines`` moved by ``shift``."""
    dx, dy = shift
    return [(r[0] + dx, r[1] + dy, r[2] + dx, r[3] + dy) for r in lines]


def _union(rects: Sequence[Rect]) -> Rect:
    """Return the smallest rectangle that covers every rectangle in ``rects``."""
    return (
        min(r[0] for r in rects),
        min(r[1] for r in rects),
        max(r[2] for r in rects),
        max(r[3] for r in rects),
    )


def _bb(text: str) -> Rect:
    """Parse a Graphviz bounding box string (left, bottom, right, top)."""
    x0, y0, x1, y1 = (float(v) for v in text.split(","))
    return (x0, y0, x1, y1)


def _point(text: str) -> Point:
    """Parse a Graphviz ``x,y`` point, ignoring any trailing fields."""
    x, y = (float(v) for v in text.split(",")[:2])
    return (x, y)


def _outlines(ops: Any) -> list[list[Point]]:
    """Return closed outlines for the ellipses, polygons and curves in Graphviz draw ops."""
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
    """Return the densified outline of ``rect``."""
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
    """Return the control points of every Bezier draw op."""
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
    """Evaluate a cubic Bezier segment at parameter ``t``."""
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = ctrl
    m = 1.0 - t
    return (
        m**3 * x0 + 3 * m * m * t * x1 + 3 * m * t * t * x2 + t**3 * x3,
        m**3 * y0 + 3 * m * m * t * y1 + 3 * m * t * t * y2 + t**3 * y3,
    )


def _sample_bezier(ctrl: Sequence[Point]) -> list[Point]:
    """Sample a piecewise cubic Bezier into a densified polyline."""
    points = [ctrl[0]]
    for start in range(0, len(ctrl) - 3, 3):
        points += [_bezier_at(ctrl[start : start + 4], k / 16) for k in range(1, 17)]
    return _densify(points)


def _bounds(points: Sequence[Point]) -> Rect:
    """Return the bounding rectangle of ``points``."""
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _grow(rect: Rect, margin: float) -> Rect:
    """Return ``rect`` enlarged by ``margin`` on every side."""
    return (rect[0] - margin, rect[1] - margin, rect[2] + margin, rect[3] + margin)


def _overlaps(a: Rect, b: Rect) -> bool:
    """Whether rectangles ``a`` and ``b`` touch or overlap."""
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def _depth(point: Point, rect: Rect) -> float:
    """How far ``point`` lies inside ``rect`` (negative outside)."""

    return min(point[0] - rect[0], rect[2] - point[0], point[1] - rect[1], rect[3] - point[1])


def _point_rect_distance(point: Point, rect: Rect) -> float:
    """Distance from ``point`` to ``rect`` (zero inside)."""
    dx = max(rect[0] - point[0], 0.0, point[0] - rect[2])
    dy = max(rect[1] - point[1], 0.0, point[1] - rect[3])
    return math.hypot(dx, dy)


def _rect_rect_gap(a: Rect, b: Rect) -> float:
    """Gap between rectangles ``a`` and ``b`` (negative when they overlap)."""
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
    """Whether ``point`` lies inside polygon ``poly`` (even-odd rule)."""
    x, y = point
    hit = False
    for (x1, y1), (x2, y2) in zip(poly, [*poly[1:], poly[0]], strict=True):
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            hit = not hit
    return hit
