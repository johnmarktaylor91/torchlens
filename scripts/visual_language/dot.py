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

    def overlaps(self, other: Box) -> bool:
        return (
            self.x0 < other.x1 and other.x0 < self.x1 and self.y0 < other.y1 and other.y0 < self.y1
        )

    def pad(self, by: float) -> Box:
        return Box(self.x0 - by, self.y0 - by, self.x1 + by, self.y1 + by)


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
    if attr == "name":
        return mark.name
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
        return _same(actual, expected)
    if op == "~":
        return expected.lower() in strip_tags(actual).lower() or expected.lower() in actual.lower()
    tokens = [token.strip().lower() for token in actual.split(",")]
    return expected.strip().lower() in tokens


def _same(actual: str, expected: str) -> bool:
    """Exact, case-insensitive match; numbers compare by value (``5.0`` equals ``5``)."""

    a, b = actual.strip().lower(), expected.strip().lower()
    try:
        return a == b or float(a) == float(b)
    except ValueError:
        return False


def select(panel: PanelDot, selector: str) -> list[Mark]:
    """Every mark of ``panel`` the selector matches."""

    kind, conditions = parse_selector(selector)
    return [m for m in panel.marks(kind) if all(_holds(m, c, panel) for c in conditions)]


# ---------------------------------------------------------------- SVG geometry


@dataclass
class SvgPanel:
    """A Graphviz SVG: its root element, viewBox, mark boxes and texts, in viewBox units."""

    root: ET.Element
    view: Box
    geometry: dict[str, list[Box]]
    texts: list[tuple[float, Box, str]]

    @property
    def font_sizes(self) -> list[float]:
        """Every non-empty text's font size, in viewBox units."""

        return [size for size, _box, _words in self.texts]

    def font_sizes_in(self, region: Box) -> list[float]:
        """Font sizes of the texts whose box overlaps ``region`` (viewBox coordinates)."""

        return [size for size, box, _words in self.texts if box.overlaps(region)]

    def text_boxes(self, needle: str) -> list[Box]:
        """Boxes of the texts containing ``needle`` (viewBox coordinates)."""

        return [box for _size, box, words in self.texts if needle in words]


def parse_svg(text: str) -> SvgPanel:
    """Parse a Graphviz SVG: viewBox, each mark's box by title, and every text.

    Every box is placed through the transforms and nested viewports above it (a code panel
    arrives as two graphs in nested ``<svg>`` elements). Marks come from the first (main)
    graph only; texts come from the whole file.
    """

    # Imported here: svgflat builds on this module's Box.
    from scripts.visual_language.svgflat import Affine, bounds, local, points, walk

    root = ET.fromstring(text)
    view_nums = [float(n) for n in _NUMBER_RE.findall(root.get("viewBox", "0 0 100 100"))]
    view = Box(view_nums[0], view_nums[1], view_nums[0] + view_nums[2], view_nums[1] + view_nums[3])
    frames = {id(el): frame for el, frame in walk(root, Affine())}

    def placed(elements: list[ET.Element]) -> Box | None:
        boxes = [frames[id(el)].box(b) for el in elements if (b := bounds(points(el)))]
        return bounds([(b.x0, b.y0) for b in boxes] + [(b.x1, b.y1) for b in boxes])

    main = next((g for g in root.iter(f"{{{SVG_NS}}}g") if g.get("class") == "graph"), root)
    geometry: dict[str, list[Box]] = {}
    for group in main.iter(f"{{{SVG_NS}}}g"):
        title = group.find(f"{{{SVG_NS}}}title")
        if title is None or group.get("class") not in ("node", "edge", "cluster"):
            continue
        name = html.unescape("".join(title.itertext())).replace("&#45;", "-")
        kind = group.get("class")
        box = placed(list(group.iter()))
        if box is not None:
            geometry.setdefault(f"{kind}:{name}", []).append(box)
        words = placed(list(group.iter(f"{{{SVG_NS}}}text")))
        if words is not None:
            geometry.setdefault(f"{kind}-words:{name}", []).append(words)
    texts = []
    for element, frame in walk(root, Affine()):
        words_text = "".join(element.itertext()).strip()
        if local(element) == "text" and element.get("font-size") and words_text:
            box = bounds(points(element)) or Box(0, 0, 0, 0)
            texts.append(
                (float(element.get("font-size", "14")) * frame.s, frame.box(box), words_text)
            )
    return SvgPanel(root, view, geometry, texts)


def mark_box(svg: SvgPanel, mark: Mark, occurrence: int = 0, words: bool = False) -> Box | None:
    """The viewBox-space box of a selected mark, or None when the SVG has no such title.

    ``words`` bounds only the mark's text (an edge's label without its whole path).
    """

    if mark.kind == "graph":
        return Box(svg.view.x0, svg.view.y0, svg.view.x0 + svg.view.w, svg.view.y0 + 30)
    key = f"{mark.kind}{'-words' if words else ''}:{mark.name}"
    boxes = svg.geometry.get(key, [])
    if not boxes:
        return None
    return boxes[min(occurrence, len(boxes) - 1)]


def edge_occurrence(panel: PanelDot, mark: Mark) -> int:
    """Which of several same-named edges this is (Graphviz writes them in order)."""

    same = [e for e in panel.edges if e.name == mark.name]
    return next((i for i, e in enumerate(same) if e.index == mark.index), 0)


def attrs_of(mark: Mark, keys: tuple[str, ...]) -> dict[str, Any]:
    """The subset of a mark's attributes named by ``keys`` that it sets."""

    return {k: mark.attrs[k] for k in keys if k in mark.attrs and mark.attrs[k] != ""}
