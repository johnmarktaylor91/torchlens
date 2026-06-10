"""Programmatic geometry audit for graphviz edge-label placement.

Library API (used by ``tests/test_label_geometry.py`` as a regression gate)::

    from support.label_geometry import audit_gv_source
    result = audit_gv_source(gv_source)
    assert result.hard_violation_count == 0, result.describe_violations()

Manual runs over a directory of ``*.gv`` files::

    python tests/support/label_geometry.py <dir_with_gv_files>

The audit runs ``dot -Tjson`` (verified against graphviz 7.0.5, xdot JSON
flavor, xdotversion 1.7) and parses the computed layout draw ops:

  - node shapes:     node ``_draw_`` 'e'/'E' (ellipse: [cx,cy,rx,ry]) and
                     'p'/'P' (polygon points)
  - cluster borders: subgraph objects with a ``bb`` rect (names cluster_*)
  - edge splines:    edge ``_draw_`` 'b'/'B' bezier control points
                     (3n+1 points), sampled densely
  - arrowheads:      'p'/'P' polygons in edge ``_hdraw_``/``_tdraw_``
                     (graphviz 7 emits arrowheads there, NOT in ``_draw_``)
  - labels:          edge ``_ldraw_`` (midpoint), ``_hldraw_`` (head),
                     ``_tldraw_`` (tail).  All 'T' ops of one block form ONE
                     label (multi-line midpoint labels).  Each 'T' gives the
                     baseline anchor point, justification ('l'/'c'/'r') and
                     the EXACT computed glyph-run width; the preceding 'F'
                     op gives the font size.

Coordinates are points, y-UP.  Per-line glyph bbox:
    x-extent: from justification + width
    y-extent: [baseline - DESCENT*fontsize, baseline + ASCENT*fontsize]
              (ascent extends UP = +y in the y-up frame)

All gaps are exact signed values in points: positive = clearance, negative =
penetration depth.  A hard violation is classified when penetration exceeds
PEN_EPS (default 0.25 pt).  The calibrated parameters (glyph bbox model and
penetration threshold) are LOCKED to the values used by the offline placement
sweep that tuned the rendering constants -- do not weaken them.

Stdlib only.  Deterministic.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional

# ----------------------------------------------------------------------------
# calibrated constants (LOCKED -- keep in sync with the offline audit sweep)
# ----------------------------------------------------------------------------
ASCENT = 0.78  # fraction of font size above baseline (y-up: +y)
DESCENT = 0.22  # fraction of font size below baseline (y-up: -y)
PEN_EPS = 0.25  # penetration (pt) beyond which an overlap is a violation
SPLINE_SAMPLES = 64  # sample points per cubic bezier segment
ELLIPSE_SAMPLES = 256  # polygon approximation of ellipse outlines
POLY_EDGE_SAMPLES = 64  # samples per polygon edge for penetration depth
BORDER_SAMPLES = 512  # samples per cluster-border side for depth

# ----------------------------------------------------------------------------
# geometry primitives (rects are (x0, y0, x1, y1), y-up)
# ----------------------------------------------------------------------------


def rect_corners(r):
    x0, y0, x1, y1 = r
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def rect_edges(r):
    c = rect_corners(r)
    return [(c[i], c[(i + 1) % 4]) for i in range(4)]


def point_in_rect(p, r):
    return r[0] <= p[0] <= r[2] and r[1] <= p[1] <= r[3]


def depth_in_rect(p, r):
    """Distance from p to the rect boundary, positive iff p is inside."""
    return min(p[0] - r[0], r[2] - p[0], p[1] - r[1], r[3] - p[1])


def point_rect_dist(p, r):
    dx = max(r[0] - p[0], 0.0, p[0] - r[2])
    dy = max(r[1] - p[1], 0.0, p[1] - r[3])
    return math.hypot(dx, dy)


def point_seg_dist(p, a, b):
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 == 0.0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / L2
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _orient(a, b, c):
    v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if v > 1e-12:
        return 1
    if v < -1e-12:
        return -1
    return 0


def _on_seg(a, b, c):
    return (
        min(a[0], b[0]) - 1e-12 <= c[0] <= max(a[0], b[0]) + 1e-12
        and min(a[1], b[1]) - 1e-12 <= c[1] <= max(a[1], b[1]) + 1e-12
    )


def segs_intersect(p1, p2, p3, p4):
    d1 = _orient(p3, p4, p1)
    d2 = _orient(p3, p4, p2)
    d3 = _orient(p1, p2, p3)
    d4 = _orient(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and (
        (d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)
    ):
        return True
    if d1 == 0 and _on_seg(p3, p4, p1):
        return True
    if d2 == 0 and _on_seg(p3, p4, p2):
        return True
    if d3 == 0 and _on_seg(p1, p2, p3):
        return True
    if d4 == 0 and _on_seg(p1, p2, p4):
        return True
    return False


def seg_rect_dist(a, b, r):
    """Min distance between segment ab and rect r (0 if touching/crossing)."""
    if point_in_rect(a, r) or point_in_rect(b, r):
        return 0.0
    for e0, e1 in rect_edges(r):
        if segs_intersect(a, b, e0, e1):
            return 0.0
    d = min(point_rect_dist(a, r), point_rect_dist(b, r))
    for c in rect_corners(r):
        d = min(d, point_seg_dist(c, a, b))
    return d


def point_in_poly(p, poly):
    x, y = p
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            xc = xi + (y - yi) / (yj - yi) * (xj - xi)
            if x < xc:
                inside = not inside
        j = i
    return inside


def poly_boundary_dist(p, poly):
    n = len(poly)
    return min(point_seg_dist(p, poly[i], poly[(i + 1) % n]) for i in range(n))


def rect_rect_signed(a, b):
    """Signed separation of two AABBs.

    > 0: Euclidean gap; <= 0: -(penetration depth) where penetration is the
    minimum axis overlap (minimum translation to separate).
    """
    ox = min(a[2], b[2]) - max(a[0], b[0])
    oy = min(a[3], b[3]) - max(a[1], b[1])
    if ox > 0 and oy > 0:
        return -min(ox, oy)
    return math.hypot(max(-ox, 0.0), max(-oy, 0.0))


def rect_poly_signed(r, poly):
    """Signed distance between rect r and the REGION bounded by poly.

    > 0: gap to the poly outline; <= 0: -(estimated penetration depth).
    Uses the true outline so a label near an ellipse "corner" (where the
    AABB of the ellipse would overlap but the ellipse does not) reports a
    positive gap.
    """
    n = len(poly)
    edges = [(poly[i], poly[(i + 1) % n]) for i in range(n)]
    crossing = False
    gap = float("inf")
    for a, b in edges:
        d = seg_rect_dist(a, b, r)
        if d == 0.0:
            crossing = True
            break
        gap = min(gap, d)
    rect_in_poly = all(point_in_poly(c, poly) for c in rect_corners(r))
    poly_in_rect = all(point_in_rect(p, r) for p in poly)
    if not crossing and not rect_in_poly and not poly_in_rect:
        return gap
    # overlap: estimate penetration depth
    pen = 0.0
    for a, b in edges:
        for i in range(POLY_EDGE_SAMPLES + 1):
            t = i / POLY_EDGE_SAMPLES
            pt = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
            d = depth_in_rect(pt, r)
            if d > pen:
                pen = d
    for c in rect_corners(r):
        if point_in_poly(c, poly):
            d = poly_boundary_dist(c, poly)
            if d > pen:
                pen = d
    return -pen


def sample_ellipse(cx, cy, rx, ry, n=ELLIPSE_SAMPLES):
    return [
        (cx + rx * math.cos(2 * math.pi * i / n), cy + ry * math.sin(2 * math.pi * i / n))
        for i in range(n)
    ]


def sample_beziers(ctrl, per_seg=SPLINE_SAMPLES):
    """ctrl: 3n+1 control points of a piecewise cubic bezier."""
    pts = []
    nseg = (len(ctrl) - 1) // 3
    for s in range(nseg):
        p0, p1, p2, p3 = ctrl[3 * s : 3 * s + 4]
        for i in range(per_seg + 1):
            if s > 0 and i == 0:
                continue  # avoid duplicating segment joints
            t = i / per_seg
            mt = 1.0 - t
            x = mt**3 * p0[0] + 3 * mt * mt * t * p1[0] + 3 * mt * t * t * p2[0] + t**3 * p3[0]
            y = mt**3 * p0[1] + 3 * mt * mt * t * p1[1] + 3 * mt * t * t * p2[1] + t**3 * p3[1]
            pts.append((x, y))
    return pts


def rect_polyline_signed(r, pts):
    """Signed distance of rect to a sampled polyline (e.g. edge spline).

    > 0: min gap; <= 0: -(max inside-depth of any sampled point in the rect).
    """
    pen = 0.0
    crossed = False
    for p in pts:
        d = depth_in_rect(p, r)
        if d >= 0.0:
            crossed = True
            if d > pen:
                pen = d
    if crossed:
        return -pen
    gap = float("inf")
    for i in range(len(pts) - 1):
        d = seg_rect_dist(pts[i], pts[i + 1], r)
        if d < gap:
            gap = d
    return gap


def rect_border_signed(r, cb):
    """Signed distance of label rect r to the OUTLINE of cluster rect cb.

    Fully inside or fully outside the cluster is fine (positive gap to the
    border line).  Crossing/touching the border returns -(max depth of any
    border point inside r).
    """
    sides = rect_edges(cb)
    gap = min(seg_rect_dist(a, b, r) for a, b in sides)
    if gap > 0.0:
        return gap
    pen = 0.0
    for a, b in sides:
        for i in range(BORDER_SAMPLES + 1):
            t = i / BORDER_SAMPLES
            pt = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
            d = depth_in_rect(pt, r)
            if d > pen:
                pen = d
    return -pen


# ----------------------------------------------------------------------------
# dot -Tjson parsing
# ----------------------------------------------------------------------------


def parse_labels(edge, edge_name, edge_gvid):
    """Group T ops per ldraw block: one block = ONE label (multi-line ok)."""
    kinds = (("_hldraw_", "head"), ("_tldraw_", "tail"), ("_ldraw_", "midpoint"))
    labels = []
    for key, kind in kinds:
        ops = edge.get(key)
        if not ops:
            continue
        size = 14.0
        lines = []
        for op in ops:
            if op["op"] == "F":
                size = float(op["size"])
            elif op["op"] == "T":
                x, y = float(op["pt"][0]), float(op["pt"][1])
                w = float(op["width"])
                al = op.get("align", "l")
                if al == "r":
                    x0 = x - w
                elif al == "c":
                    x0 = x - w / 2.0
                else:
                    x0 = x
                rect = (x0, y - DESCENT * size, x0 + w, y + ASCENT * size)
                lines.append(
                    {
                        "text": op["text"],
                        "baseline": [x, y],
                        "align": al,
                        "width": w,
                        "fontsize": size,
                        "bbox": list(rect),
                    }
                )
        if lines:
            ub = (
                min(line["bbox"][0] for line in lines),
                min(line["bbox"][1] for line in lines),
                max(line["bbox"][2] for line in lines),
                max(line["bbox"][3] for line in lines),
            )
            labels.append(
                {
                    "kind": kind,
                    "edge": edge_name,
                    "edge_gvid": edge_gvid,
                    "text": " / ".join(line["text"] for line in lines),
                    "lines": lines,
                    "bbox": list(ub),
                }
            )
    return labels


def shape_outlines(draw_ops):
    """Extract closed outlines (point lists) from a _draw_ op array."""
    outlines = []
    for op in draw_ops or []:
        o = op["op"]
        if o in ("e", "E"):
            cx, cy, rx, ry = [float(v) for v in op["rect"]]
            outlines.append(sample_ellipse(cx, cy, rx, ry))
        elif o in ("p", "P"):
            outlines.append([(float(p[0]), float(p[1])) for p in op["points"]])
    return outlines


def bezier_controls(draw_ops):
    out = []
    for op in draw_ops or []:
        if op["op"] in ("b", "B"):
            out.append([(float(p[0]), float(p[1])) for p in op["points"]])
    return out


def parse_graph(json_doc):
    sub_cnt = json_doc.get("_subgraph_cnt", 0)
    objects = json_doc.get("objects", [])
    gvid_name = {}
    nodes = []  # (name, [outline, ...])
    clusters = []  # (name, (x0,y0,x1,y1))
    for obj in objects:
        gvid_name[obj["_gvid"]] = obj.get("name", "obj%d" % obj["_gvid"])
    for obj in objects[:sub_cnt]:
        bb = obj.get("bb")
        if bb:
            x0, y0, x1, y1 = [float(v) for v in bb.split(",")]
            clusters.append((obj.get("name", "?"), (x0, y0, x1, y1)))
    for obj in objects[sub_cnt:]:
        outs = shape_outlines(obj.get("_draw_"))
        if outs:
            nodes.append((obj["name"], outs))
    edges = []
    for idx, e in enumerate(json_doc.get("edges", [])):
        tail = gvid_name.get(e["tail"], str(e["tail"]))
        head = gvid_name.get(e["head"], str(e["head"]))
        gvid = e.get("_gvid", idx)
        name = "%s->%s" % (tail, head)
        splines = [sample_beziers(c) for c in bezier_controls(e.get("_draw_"))]
        arrows = []
        for key in ("_hdraw_", "_tdraw_"):
            for outline in shape_outlines(e.get(key)):
                arrows.append(outline)
        labels = parse_labels(e, name, gvid)
        edges.append(
            {
                "gvid": gvid,
                "name": name,
                "tail": tail,
                "head": head,
                "splines": splines,
                "arrowheads": arrows,
                "labels": labels,
            }
        )
    return {"nodes": nodes, "clusters": clusters, "edges": edges, "bb": json_doc.get("bb", "")}


# ----------------------------------------------------------------------------
# per-graph audit
# ----------------------------------------------------------------------------


def label_signed_to_outline(label, outline):
    return min(rect_poly_signed(tuple(line["bbox"]), outline) for line in label["lines"])


def label_signed_to_polyline(label, pts):
    return min(rect_polyline_signed(tuple(line["bbox"]), pts) for line in label["lines"])


def label_signed_to_label(a, b):
    return min(
        rect_rect_signed(tuple(la["bbox"]), tuple(lb["bbox"]))
        for la in a["lines"]
        for lb in b["lines"]
    )


def label_signed_to_border(label, cb):
    return min(rect_border_signed(tuple(line["bbox"]), cb) for line in label["lines"])


def audit_graph(g, pen_eps):
    edges = g["edges"]
    nodes = g["nodes"]
    clusters = g["clusters"]
    all_labels = []
    for e in edges:
        for lab in e["labels"]:
            all_labels.append((e, lab))

    results = []
    for e, lab in all_labels:
        violations = []

        # (a) label vs label (reported on BOTH labels of a pair; the
        # graph-level unique count de-dupes the pair)
        for j2, (e2, lab2) in enumerate(all_labels):
            if lab2 is lab:
                continue
            sd = label_signed_to_label(lab, lab2)
            if sd < -pen_eps:
                violations.append(
                    {
                        "type": "label-label",
                        "other": "%s %s '%s'" % (lab2["edge"], lab2["kind"], lab2["text"]),
                        "pair_key": "label-label:%s"
                        % "+".join(
                            sorted(
                                [
                                    "%s/%s" % (lab["edge_gvid"], lab["kind"]),
                                    "%s/%s" % (lab2["edge_gvid"], lab2["kind"]),
                                ]
                            )
                        ),
                        "signed_gap": round(sd, 3),
                    }
                )

        # (b) label vs node outline (true outline, not bbox)
        node_gaps = {}
        for nname, outlines in nodes:
            sd = min(label_signed_to_outline(lab, o) for o in outlines)
            node_gaps[nname] = sd
            if sd < -pen_eps:
                violations.append(
                    {
                        "type": "label-node",
                        "other": nname,
                        "signed_gap": round(sd, 3),
                    }
                )

        # (c) label vs any edge spline
        own_spline_sd = None
        for e2 in edges:
            if not e2["splines"]:
                continue
            sd = min(label_signed_to_polyline(lab, pts) for pts in e2["splines"])
            if e2 is e:
                own_spline_sd = sd
            if sd < -pen_eps:
                violations.append(
                    {
                        "type": "label-spline",
                        "other": e2["name"],
                        "signed_gap": round(sd, 3),
                    }
                )

        # (d) label vs arrowhead polygons
        arrow_gaps = []
        for e2 in edges:
            for ai, poly in enumerate(e2["arrowheads"]):
                sd = label_signed_to_outline(lab, poly)
                arrow_gaps.append((sd, e2["name"], ai, e2 is e))
                if sd < -pen_eps:
                    violations.append(
                        {
                            "type": "label-arrowhead",
                            "other": "%s arrow#%d" % (e2["name"], ai),
                            "signed_gap": round(sd, 3),
                        }
                    )

        # (e) label vs cluster boundary rectangle EDGE
        for cname, cb in clusters:
            sd = label_signed_to_border(lab, cb)
            if sd < -pen_eps:
                violations.append(
                    {
                        "type": "label-clusterborder",
                        "other": cname,
                        "signed_gap": round(sd, 3),
                    }
                )

        # ---- soft metrics ----
        endpoint_names = {e["tail"], e["head"]}
        own_endpoint_gap = None
        own_endpoint_node = None
        for nname in sorted(endpoint_names):
            if nname in node_gaps:
                if own_endpoint_gap is None or node_gaps[nname] < own_endpoint_gap:
                    own_endpoint_gap = node_gaps[nname]
                    own_endpoint_node = nname
        nearest_node = None
        nearest_node_gap = None
        for nname in sorted(node_gaps):
            if nearest_node_gap is None or node_gaps[nname] < nearest_node_gap:
                nearest_node_gap = node_gaps[nname]
                nearest_node = nname
        nearest_arrow = None
        nearest_arrow_gap = None
        for sd, ename, ai, is_own in sorted(arrow_gaps, key=lambda t: (t[0], t[1], t[2])):
            nearest_arrow_gap = sd
            nearest_arrow = "%s arrow#%d%s" % (ename, ai, " (own)" if is_own else "")
            break

        rec = {
            "edge": lab["edge"],
            "edge_gvid": lab["edge_gvid"],
            "kind": lab["kind"],
            "text": lab["text"],
            "bbox": [round(v, 3) for v in lab["bbox"]],
            "lines": lab["lines"],
            "metrics": {
                "gap_to_own_endpoint_node": _r(own_endpoint_gap),
                "own_endpoint_node": own_endpoint_node,
                "gap_to_nearest_node": _r(nearest_node_gap),
                "nearest_node": nearest_node,
                "gap_to_own_spline": _r(own_spline_sd),
                "gap_to_nearest_arrowhead": _r(nearest_arrow_gap),
                "nearest_arrowhead": nearest_arrow,
            },
            "violations": violations,
        }
        results.append(rec)
    # deterministic order
    results.sort(key=lambda r: (r["edge_gvid"], {"head": 0, "tail": 1, "midpoint": 2}[r["kind"]]))
    # unique violation count: label-label pairs count once
    seen = set()
    unique = 0
    for rec in results:
        for v in rec["violations"]:
            key = v.get("pair_key") or (
                "%s:%s/%s:%s" % (v["type"], rec["edge_gvid"], rec["kind"], v["other"])
            )
            if key not in seen:
                seen.add(key)
                unique += 1
    return results, unique


def _r(v):
    return None if v is None else round(v, 3)


# ----------------------------------------------------------------------------
# library API
# ----------------------------------------------------------------------------


@dataclasses.dataclass
class AuditResult:
    """Audit outcome for one rendered graph.

    Attributes:
        labels: Per-label records (edge, kind, text, bbox, soft metrics, and
            any hard violations), in deterministic order.
        hard_violation_count: Unique hard violations (label-label pairs are
            counted once even though they are recorded on both labels).
    """

    labels: List[Dict[str, Any]]
    hard_violation_count: int

    @property
    def violations(self) -> List[Dict[str, Any]]:
        """Flattened violation records with their label's context attached."""
        flat = []
        for rec in self.labels:
            for v in rec["violations"]:
                flat.append(
                    {
                        "edge": rec["edge"],
                        "kind": rec["kind"],
                        "text": rec["text"],
                        "type": v["type"],
                        "other": v["other"],
                        "signed_gap": v["signed_gap"],
                    }
                )
        return flat

    def describe_violations(self, graph_name: str = "graph") -> str:
        """One self-explaining line per violation, for assertion messages."""
        lines = [
            "%d hard label-geometry violation(s) in %s:" % (self.hard_violation_count, graph_name)
        ]
        for v in self.violations:
            lines.append(
                "  [%s] %s %s label '%s' vs %s: penetration %.2fpt"
                % (v["type"], v["edge"], v["kind"], v["text"], v["other"], -v["signed_gap"])
            )
        return "\n".join(lines)


def audit_gv_source(
    gv_source: str,
    *,
    dot: str = "dot",
    pen_eps: float = PEN_EPS,
) -> AuditResult:
    """Audit one graphviz source string for hard edge-label collisions.

    Lays the graph out with ``dot -Tjson`` and checks every edge label
    (head, tail, midpoint) against labels, node outlines, edge splines,
    arrowhead polygons, and cluster borders.

    Args:
        gv_source: The graphviz source text (DOT language).
        dot: The dot executable to invoke.
        pen_eps: Penetration depth (pt) classifying a hard violation.

    Returns:
        An :class:`AuditResult` with per-label records and the unique hard
        violation count.

    Raises:
        RuntimeError: If ``dot`` fails to lay out the graph.
    """
    proc = subprocess.run([dot, "-Tjson"], input=gv_source, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError("dot -Tjson failed: %s" % proc.stderr.strip())
    graph = parse_graph(json.loads(proc.stdout))
    labels, unique = audit_graph(graph, pen_eps)
    return AuditResult(labels=labels, hard_violation_count=unique)


# ----------------------------------------------------------------------------
# manual runs: python tests/support/label_geometry.py <dir_with_gv_files>
# ----------------------------------------------------------------------------


def _main(argv: Optional[List[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: python label_geometry.py <dir_with_gv_files>", file=sys.stderr)
        return 2
    gv_dir = args[0]
    gv_files = sorted(f for f in os.listdir(gv_dir) if f.endswith(".gv"))
    if not gv_files:
        print("no .gv files in %s" % gv_dir, file=sys.stderr)
        return 2
    total = 0
    for fn in gv_files:
        with open(os.path.join(gv_dir, fn)) as fh:
            result = audit_gv_source(fh.read())
        total += result.hard_violation_count
        verdict = "CLEAN" if result.hard_violation_count == 0 else "VIOLATIONS"
        print(
            "%-40s %d labels, %d hard violations  %s"
            % (fn, len(result.labels), result.hard_violation_count, verdict)
        )
        if result.hard_violation_count:
            print(result.describe_violations(fn))
    print("total hard violations: %d" % total)
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(_main())
