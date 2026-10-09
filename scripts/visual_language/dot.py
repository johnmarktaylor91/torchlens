"""Read a rendered panel: its DOT objects (from Graphviz ``-Tjson``) and SVG geometry.

A key entry names its mark with a selector over the panel's DOT, never a Graphviz node
name: ``node(shape=box3d)``, ``edge(text~"In ")``, ``cluster(penwidth=5)``,
``graph(label~stacked)``. Conditions are comma-separated; ``attr=value`` is exact and
case-insensitive, ``attr~text`` matches a substring of the tag-stripped value (``text``
searches every label attribute; ``head`` and ``tail`` search the endpoint names and
labels) and ``attr^token`` matches one token of a comma-separated list such as ``style``.
"""

from __future__ import annotations

import html
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)

_LABEL_KEYS = ("label", "xlabel", "headlabel", "taillabel")
_SELECTOR_RE = re.compile(r"^\s*(node|edge|cluster|graph)\s*\((.*)\)\s*$", re.DOTALL)
_CONDITION_RE = re.compile(r'^\s*([a-z_]+)\s*([=~^])\s*(?:"(.*)"|(.*?))\s*$', re.DOTALL)
_TAG_RE = re.compile(r"<[^>]*>")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:e-?\d+)?")


def strip_tags(value: str) -> str:
    """Visible text of a Graphviz label: tags removed, entities decoded, spaces collapsed."""

    text = _TAG_RE.sub(" ", value.replace("<BR/>", "\n").replace("<br/>", "\n"))
    return " ".join(html.unescape(text).split())


@dataclass
class Box:
    """An axis-aligned box in some coordinate system."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def w(self) -> float:
        return self.x1 - self.x0

    @property
    def h(self) -> float:
        return self.y1 - self.y0

    def union(self, other: Box) -> Box:
        return Box(
            min(self.x0, other.x0),
            min(self.y0, other.y0),
            max(self.x1, other.x1),
            max(self.y1, other.y1),
        )


@dataclass
class Mark:
    """One DOT object: a node, an edge, a cluster or the graph itself."""

    kind: str
    name: str
    attrs: dict[str, str]
    index: int = 0
    tail: str = ""
    head: str = ""

    def text(self) -> str:
        return " ".join(strip_tags(self.attrs.get(key, "")) for key in _LABEL_KEYS).strip()


@dataclass
class PanelDot:
    """Every mark of one rendered panel."""

    graph: Mark
    nodes: list[Mark] = field(default_factory=list)
    edges: list[Mark] = field(default_factory=list)
    clusters: list[Mark] = field(default_factory=list)

    def marks(self, kind: str) -> list[Mark]:
        return {
            "node": self.nodes,
            "edge": self.edges,
            "cluster": self.clusters,
            "graph": [self.graph],
        }[kind]


def parse_json(text: str) -> PanelDot:
    """Parse Graphviz ``-Tjson`` output into marks."""

    data = json.loads(text)
    scalar = {k: str(v) for k, v in data.items() if not isinstance(v, (list, dict))}
    panel = PanelDot(graph=Mark("graph", str(data.get("name", "")), scalar))
    names: dict[int, str] = {}
    for obj in data.get("objects", []):
        attrs = {k: str(v) for k, v in obj.items() if not isinstance(v, (list, dict))}
        name = str(obj.get("name", ""))
        if "nodes" in obj or "subgraphs" in obj:
            if name.startswith("cluster"):
                panel.clusters.append(Mark("cluster", name, attrs))
            continue
        names[int(obj["_gvid"])] = name
        panel.nodes.append(Mark("node", name, attrs))
    for index, edge in enumerate(data.get("edges", [])):
        attrs = {k: str(v) for k, v in edge.items() if not isinstance(v, (list, dict))}
        tail, head = names.get(int(edge["tail"]), ""), names.get(int(edge["head"]), "")
        panel.edges.append(Mark("edge", f"{tail}->{head}", attrs, index, tail, head))
    return panel


def parse_selector(selector: str) -> tuple[str, list[tuple[str, str, str]]]:
    """Split a selector into its kind and ``(attr, op, value)`` conditions."""

    match = _SELECTOR_RE.match(selector)
    if match is None:
        raise ValueError(f"bad selector {selector!r}")
    kind, body = match.groups()
    conditions = []
    for part in _split_conditions(body):
        cond = _CONDITION_RE.match(part)
        if cond is None:
            raise ValueError(f"bad condition {part!r} in selector {selector!r}")
        attr, op, quoted, bare = cond.groups()
        conditions.append((attr, op, quoted if quoted is not None else bare))
    return kind, conditions


def _split_conditions(body: str) -> list[str]:
    parts, depth, current = [], False, []
    for char in body:
        if char == '"':
            depth = not depth
        if char == "," and not depth:
            parts.append("".join(current))
            current = []
            continue
        current.append(char)
    if "".join(current).strip():
        parts.append("".join(current))
    return parts


def _value(mark: Mark, attr: str, panel: PanelDot) -> str:
    if attr == "text":
        return mark.text()
    if attr in ("head", "tail"):
        end = mark.head if attr == "head" else mark.tail
        node = next((n for n in panel.nodes if n.name == end), None)
        return f"{end} {node.text() if node else ''}"
    return mark.attrs.get(attr, "")


def _holds(mark: Mark, condition: tuple[str, str, str], panel: PanelDot) -> bool:
    attr, op, expected = condition
    actual = _value(mark, attr, panel)
    if op == "=":
        return actual.strip().lower() == expected.strip().lower()
    if op == "~":
        return expected.lower() in strip_tags(actual).lower() or expected.lower() in actual.lower()
    tokens = [token.strip().lower() for token in actual.split(",")]
    return expected.strip().lower() in tokens


def select(panel: PanelDot, selector: str) -> list[Mark]:
    """Every mark of ``panel`` the selector matches."""

    kind, conditions = parse_selector(selector)
    return [m for m in panel.marks(kind) if all(_holds(m, c, panel) for c in conditions)]


# ---------------------------------------------------------------- SVG geometry


@dataclass
class SvgPanel:
    """A Graphviz SVG: its root element, viewBox and the graph-to-viewBox transform."""

    root: ET.Element
    view: Box
    scale: float
    translate: tuple[float, float]
    geometry: dict[str, list[Box]]
    font_sizes: list[float]


def _points(element: ET.Element) -> list[tuple[float, float]]:
    tag = element.tag.split("}")[-1]
    if tag == "ellipse":
        cx, cy = float(element.get("cx", 0)), float(element.get("cy", 0))
        rx, ry = float(element.get("rx", 0)), float(element.get("ry", 0))
        return [(cx - rx, cy - ry), (cx + rx, cy + ry)]
    if tag in ("polygon", "polyline"):
        nums = [float(n) for n in _NUMBER_RE.findall(element.get("points", ""))]
        return list(zip(nums[0::2], nums[1::2], strict=False))
    if tag == "path":
        nums = [float(n) for n in _NUMBER_RE.findall(element.get("d", ""))]
        return list(zip(nums[0::2], nums[1::2], strict=False))
    if tag == "text":
        x, y = float(element.get("x", 0)), float(element.get("y", 0))
        size = float(element.get("font-size", 14))
        width = 0.55 * size * len("".join(element.itertext()))
        anchor = element.get("text-anchor", "start")
        left = x - width / 2 if anchor == "middle" else (x - width if anchor == "end" else x)
        return [(left, y - size), (left + width, y + 0.25 * size)]
    if tag == "image":
        x, y = float(element.get("x", 0)), float(element.get("y", 0))
        w = float(str(element.get("width", "0")).rstrip("ptx"))
        h = float(str(element.get("height", "0")).rstrip("ptx"))
        return [(x, y), (x + w, y + h)]
    return []


def _box(points: list[tuple[float, float]]) -> Box | None:
    if not points:
        return None
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return Box(min(xs), min(ys), max(xs), max(ys))


def parse_svg(text: str) -> SvgPanel:
    """Parse a Graphviz SVG: viewBox, the graph transform and each mark's box by title."""

    root = ET.fromstring(text)
    view_nums = [float(n) for n in _NUMBER_RE.findall(root.get("viewBox", "0 0 100 100"))]
    view = Box(view_nums[0], view_nums[1], view_nums[0] + view_nums[2], view_nums[1] + view_nums[3])
    scale, translate = 1.0, (0.0, 0.0)
    geometry: dict[str, list[Box]] = {}
    sizes: list[float] = []
    for group in root.iter(f"{{{SVG_NS}}}g"):
        if group.get("class") == "graph":
            transform = group.get("transform", "")
            sc = re.search(r"scale\(([-\d.]+)", transform)
            tr = re.search(r"translate\(([-\d.]+)[ ,]+([-\d.]+)", transform)
            scale = float(sc.group(1)) if sc else 1.0
            translate = (float(tr.group(1)), float(tr.group(2))) if tr else (0.0, 0.0)
            continue
        title = group.find(f"{{{SVG_NS}}}title")
        if title is None or group.get("class") not in ("node", "edge", "cluster"):
            continue
        name = html.unescape("".join(title.itertext())).replace("&#45;", "-")
        points = [p for child in group.iter() for p in _points(child)]
        box = _box(points)
        if box is not None:
            geometry.setdefault(f"{group.get('class')}:{name}", []).append(box)
    for element in root.iter(f"{{{SVG_NS}}}text"):
        if element.get("font-size") and "".join(element.itertext()).strip():
            sizes.append(float(element.get("font-size", "14")))
    return SvgPanel(root, view, scale, translate, geometry, sizes)


def to_view(svg: SvgPanel, box: Box) -> Box:
    """Map a box from graph coordinates to the SVG's viewBox coordinates."""

    tx, ty = svg.translate
    s = svg.scale
    return Box((box.x0 + tx) * s, (box.y0 + ty) * s, (box.x1 + tx) * s, (box.y1 + ty) * s)


def mark_box(svg: SvgPanel, mark: Mark, occurrence: int = 0) -> Box | None:
    """The viewBox-space box of a selected mark, or None when the SVG has no such title."""

    if mark.kind == "graph":
        return Box(svg.view.x0, svg.view.y0, svg.view.x0 + svg.view.w, svg.view.y0 + 30)
    key = f"{mark.kind}:{mark.name}"
    boxes = svg.geometry.get(key, [])
    if not boxes:
        return None
    return to_view(svg, boxes[min(occurrence, len(boxes) - 1)])


def edge_occurrence(panel: PanelDot, mark: Mark) -> int:
    """Which of several same-named edges this is (Graphviz writes them in order)."""

    same = [e for e in panel.edges if e.name == mark.name]
    return next((i for i, e in enumerate(same) if e.index == mark.index), 0)


def attrs_of(mark: Mark, keys: tuple[str, ...]) -> dict[str, Any]:
    """The subset of a mark's attributes named by ``keys`` that it sets."""

    return {k: mark.attrs[k] for k in keys if k in mark.attrs and mark.attrs[k] != ""}
