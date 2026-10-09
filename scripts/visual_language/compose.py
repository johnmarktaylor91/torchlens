"""Compose slide pictures from real TorchLens renders: badges, key column, grids, tables.

Each slide picture is one SVG whose user units equal the deck card's design pixels (the
card is 960 by 540 with 48 px padding), sized to the room the slide's text leaves, so the
legibility check can compute the size every text is shown at. Renders are embedded as
nested ``<svg>`` elements with their ids prefixed; nothing references outside the file and
no scripts are written. Badges are near-black filled circles with white numerals, a mark
TorchLens never draws.
"""

from __future__ import annotations

import copy
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from scripts.visual_language.dot import (
    SVG_NS,
    XLINK_NS,
    Box,
    PanelDot,
    SvgPanel,
    edge_occurrence,
    mark_box,
    parse_json,
    parse_svg,
    select,
)
from scripts.visual_language.slides import Key, Slide, fill

DESIGN_W, DESIGN_H, PAD, GAP = 960.0, 540.0, 48.0, 19.2
TITLE_PX, BODY_PX = 46.0, 26.0
ROOM_W = DESIGN_W - 2 * PAD
#: CSS px per design px at the pane target: the room is about 720 CSS px wide.
CARD_SCALE = 720.0 / ROOM_W
MIN_CSS_PX = 9.0
MAX_ASPECT_MISMATCH = 2.2
#: Largest enlargement of a render: a three-node graph is not blown up to poster type.
MAX_SCALE = 1.45
CROP_PAD = 10.0
KEY_PX, LABEL_PX, NOTE_PX, BADGE_R = 15.0, 13.0, 13.0, 11.0
CHAR_EM = 0.55
INK, FAINT, BADGE_FILL = "#222222", "#555555", "#222222"
FONT = "Helvetica, Arial, sans-serif"


def _lines(text: str, px: float, width: float) -> int:
    return max(1, math.ceil(len(text) * CHAR_EM * px / width)) if text else 0


def room_height(slide: Slide) -> float:
    """Design-px height the deck card leaves for the picture under the title and rule."""

    title = _lines(slide.title, TITLE_PX, ROOM_W) * TITLE_PX * 1.15
    body = _lines(fill(slide.rule), BODY_PX, ROOM_W) * BODY_PX * 1.4
    return DESIGN_H - 2 * PAD - title - GAP - body - GAP


def wrap(text: str, px: float, width: float) -> list[str]:
    """Greedy word wrap by estimated Helvetica advance."""

    per_line = max(8, int(width / (CHAR_EM * px)))
    lines: list[str] = []
    current = ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        if len(trial) > per_line and current:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    return lines


def _el(
    tag: str, attrs: dict[str, str | float] | None = None, text: str | None = None
) -> ET.Element:
    element = ET.Element(f"{{{SVG_NS}}}{tag}", {k: str(v) for k, v in (attrs or {}).items()})
    if text is not None:
        element.text = text
    return element


def text_block(
    parent: ET.Element,
    lines: list[str],
    x: float,
    y: float,
    px: float,
    color: str = INK,
    weight: str = "normal",
    anchor: str = "start",
) -> float:
    """Write lines of text starting at baseline ``y``; return the y after the block."""

    for line in lines:
        node = _el(
            "text",
            {
                "x": round(x, 2),
                "y": round(y, 2),
                "font-family": FONT,
                "font-size": px,
                "fill": color,
                "font-weight": weight,
                "text-anchor": anchor,
            },
            line,
        )
        parent.append(node)
        y += px * 1.25
    return y


@dataclass
class Render:
    """One rendered panel: its DOT marks, SVG geometry, and the part of it shown."""

    stem: str
    dot: PanelDot
    svg: SvgPanel
    label: str = ""
    view: Box | None = None
    problems: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, folder: Path, stem: str, label: str = "", crop: tuple[str, ...] = ()) -> Render:
        """Load a panel; ``crop`` selectors narrow the shown part to their marks' union."""

        dot = parse_json((folder / f"{stem}.json").read_text())
        svg = parse_svg((folder / f"{stem}.svg").read_text())
        render = cls(stem, dot, svg, label)
        if crop:
            render.view = render.region(crop)
        return render

    @property
    def shown(self) -> Box:
        return self.view or self.svg.view

    def region(self, selectors: tuple[str, ...]) -> Box | None:
        """The padded union of every mark the selectors find, clipped to the viewBox."""

        boxes = []
        for selector in selectors:
            found = [
                mark_box(self.svg, m, edge_occurrence(self.dot, m) if m.kind == "edge" else 0)
                for m in select(self.dot, fill(selector))
            ]
            found = [b for b in found if b is not None]
            if not found:
                self.problems.append(f"crop {selector} found no mark on {self.stem}")
            boxes.extend(found)
        if not boxes:
            return None
        union = boxes[0]
        for box in boxes[1:]:
            union = union.union(box)
        view = self.svg.view
        union = union.pad(CROP_PAD)
        return Box(
            max(union.x0, view.x0),
            max(union.y0, view.y0),
            min(union.x1, view.x1),
            min(union.y1, view.y1),
        )


@dataclass
class Placed:
    """A render placed on the canvas: where its viewBox landed and at what scale."""

    render: Render
    x: float
    y: float
    scale: float
    crop: Box

    def point(self, box: Box) -> Box:
        s, c = self.scale, self.crop
        return Box(
            self.x + (box.x0 - c.x0) * s,
            self.y + (box.y0 - c.y0) * s,
            self.x + (box.x1 - c.x0) * s,
            self.y + (box.y1 - c.y0) * s,
        )


@dataclass
class Canvas:
    """The slide picture under construction plus its legibility record."""

    width: float
    height: float
    root: ET.Element = field(init=False)
    placed: list[Placed] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    min_css_px: float = 99.0
    witnesses: dict[str, list[str]] = field(default_factory=dict)
    badges: list[tuple[float, float]] = field(default_factory=list)
    defs: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.root = _el(
            "svg",
            {
                "width": f"{self.width:.0f}",
                "height": f"{self.height:.0f}",
                "viewBox": f"0 0 {self.width:.2f} {self.height:.2f}",
            },
        )
        self.root.append(
            _el(
                "rect",
                {"x": 0, "y": 0, "width": self.width, "height": self.height, "fill": "white"},
            )
        )

    def note_text(self, px: float, what: str) -> None:
        css = px * CARD_SCALE
        self.min_css_px = min(self.min_css_px, css)
        if css < MIN_CSS_PX:
            self.problems.append(f"{what}: text shown at {css:.1f} CSS px, under {MIN_CSS_PX}")

    def text(self, lines: list[str], x: float, y: float, px: float, **kw: str) -> float:
        self.note_text(px, "slide text")
        return text_block(self.root, lines, x, y, px, **kw)

    def embed(
        self, render: Render, area: Box, crop: Box | None = None, shared: bool = False
    ) -> Placed:
        """Fit a render (or a crop of it) into ``area``, centred, aspect kept.

        ``shared`` writes the render's content once into ``<defs>`` and places it with
        ``<use>``, so several crops of one render cost one copy.
        """

        view = crop or render.shown
        scale = min(area.w / view.w, area.h / view.h, MAX_SCALE)
        w, h = view.w * scale, view.h * scale
        x, y = area.x0 + (area.w - w) / 2, area.y0 + (area.h - h) / 2
        frame = {
            "x": f"{x:.2f}",
            "y": f"{y:.2f}",
            "width": f"{w:.2f}",
            "height": f"{h:.2f}",
            "viewBox": f"{view.x0:.2f} {view.y0:.2f} {view.w:.2f} {view.h:.2f}",
            "preserveAspectRatio": "xMidYMid meet",
            "overflow": "hidden",
        }
        if shared:
            nested = _el("svg", frame)
            nested.append(_el("use", {"href": f"#{self._define(render)}"}))
        else:
            nested = copy.deepcopy(render.svg.root)
            _strip_metadata(nested)
            _prefix_ids(nested, render.stem)
            nested.attrib.update(frame)
        self.root.append(nested)
        placed = Placed(render, x, y, scale, view)
        self.placed.append(placed)
        sizes = render.svg.font_sizes_in(view)
        if sizes:
            self.note_text(min(sizes) * scale, f"render {render.stem}")
        mismatch = (view.w / view.h) / (area.w / area.h)
        if max(mismatch, 1 / mismatch) > MAX_ASPECT_MISMATCH:
            self.problems.append(
                f"render {render.stem}: aspect {view.w / view.h:.2f} against room "
                f"{area.w / area.h:.2f} (over {MAX_ASPECT_MISMATCH}x)"
            )
        return placed

    def _define(self, render: Render) -> str:
        """Write the render's drawing once into ``<defs>``; return its id."""

        if render.stem in self.defs:
            return self.defs[render.stem]
        source = copy.deepcopy(render.svg.root)
        _strip_metadata(source)
        _prefix_ids(source, render.stem)
        ident = re.sub(r"[^A-Za-z0-9_-]", "_", f"def-{render.stem}")
        group = _el("g", {"id": ident})
        group.extend(list(source))
        defs = self.root.find(f"{{{SVG_NS}}}defs")
        if defs is None:
            defs = _el("defs")
            self.root.insert(0, defs)
        defs.append(group)
        self.defs[render.stem] = ident
        return ident

    def badge(self, number: int, placed: Placed, selector: str, kind_hint: str) -> bool:
        """Place badge ``number`` beside the first mark the selector finds; False if none."""

        matches = select(placed.render.dot, fill(selector))
        self.witnesses[selector] = [m.name for m in matches]
        for mark in matches:
            occurrence = edge_occurrence(placed.render.dot, mark) if mark.kind == "edge" else 0
            box = mark_box(placed.render.svg, mark, occurrence)
            if box is None or not box.overlaps(placed.crop):
                continue
            spot = placed.point(box)
            cx, cy = _badge_spot(spot, mark.kind)
            cx, cy = self._nudge(cx, cy)
            self.badges.append((cx, cy))
            self.root.append(
                _el(
                    "circle",
                    {
                        "cx": round(cx, 2),
                        "cy": round(cy, 2),
                        "r": BADGE_R,
                        "fill": BADGE_FILL,
                        "stroke": "white",
                        "stroke-width": 1.5,
                    },
                )
            )
            self.root.append(
                _el(
                    "text",
                    {
                        "x": round(cx, 2),
                        "y": round(cy + 4.6, 2),
                        "font-family": FONT,
                        "font-size": LABEL_PX,
                        "font-weight": "bold",
                        "fill": "white",
                        "text-anchor": "middle",
                    },
                    str(number),
                )
            )
            return True
        return False

    def _nudge(self, cx: float, cy: float) -> tuple[float, float]:
        cx = min(max(cx, BADGE_R + 1), self.width - BADGE_R - 1)
        cy = min(max(cy, BADGE_R + 1), self.height - BADGE_R - 1)
        for bx, by in self.badges:
            if abs(bx - cx) < 2.2 * BADGE_R and abs(by - cy) < 2.2 * BADGE_R:
                cx += 2.4 * BADGE_R
        return cx, cy

    def write(self, path: Path) -> None:
        tree = ET.ElementTree(self.root)
        ET.indent(tree, space=" ")
        tree.write(path, encoding="unicode", xml_declaration=False)


def _badge_spot(box: Box, kind: str) -> tuple[float, float]:
    if kind == "edge":
        return box.x0 + box.w / 2, box.y0 + box.h / 2 - BADGE_R
    if kind == "cluster":
        return box.x0 + BADGE_R * 0.4, box.y0 + BADGE_R * 0.4
    return box.x0 + BADGE_R * 0.2, box.y0 + BADGE_R * 0.2


def _strip_metadata(root: ET.Element) -> None:
    """Drop invisible metadata (titles, comments) that a viewer never draws."""

    for parent in list(root.iter()):
        for child in list(parent):
            if child.tag == f"{{{SVG_NS}}}title" or not isinstance(child.tag, str):
                parent.remove(child)


_URL_RE = re.compile(r"url\(#([^)]+)\)")


def _prefix_ids(root: ET.Element, prefix: str) -> None:
    """Prefix every id and in-file reference so several renders share one file safely."""

    clean = re.sub(r"[^A-Za-z0-9_-]", "_", prefix)
    href = f"{{{XLINK_NS}}}href"
    for element in root.iter():
        if "id" in element.attrib:
            element.attrib["id"] = f"{clean}-{element.attrib['id']}"
        for attr, value in list(element.attrib.items()):
            if "url(#" in value:
                element.attrib[attr] = _URL_RE.sub(lambda m: f"url(#{clean}-{m.group(1)})", value)
            if attr in (href, "href") and value.startswith("#"):
                element.attrib[attr] = f"#{clean}-{value[1:]}"


# ------------------------------------------------------------------ layouts


def _need(render: Render) -> float:
    """The smallest scale at which every text shown in ``render`` stays legible."""

    sizes = render.svg.font_sizes_in(render.shown)
    return MIN_CSS_PX / CARD_SCALE / min(sizes) if sizes else 0.1


def _split_area(area: Box, renders: list[Render], label_h: float) -> list[Box]:
    """Areas for 1 to 3 renders, side by side or stacked, sized by what each needs.

    Each render's share of the width (side by side) or height (stacked) is proportional
    to its extent times the scale its smallest text needs; the arrangement whose worst
    render keeps the larger legibility margin wins.
    """

    n = len(renders)
    if n == 1:
        return [Box(area.x0, area.y0 + label_h, area.x1, area.y1)]

    def margin(boxes: list[Box]) -> float:
        return min(
            min(b.w / r.shown.w, b.h / r.shown.h, MAX_SCALE) / _need(r)
            for b, r in zip(boxes, renders, strict=True)
        )

    gap = 10.0
    side, stack = [], []
    weights = [r.shown.w * _need(r) for r in renders]
    x = area.x0
    for weight in weights:
        w = (area.w - gap * (n - 1)) * weight / sum(weights)
        side.append(Box(x, area.y0 + label_h, x + w, area.y1))
        x += w + gap
    weights = [r.shown.h * _need(r) + label_h for r in renders]
    y = area.y0
    for weight in weights:
        h = (area.h - gap * (n - 1)) * weight / sum(weights)
        stack.append(Box(area.x0, y + label_h, area.x1, y + h))
        y += h + gap
    return side if margin(side) >= margin(stack) else stack


def _panel_labels(canvas: Canvas, areas: list[Box], renders: list[Render], label_h: float) -> None:
    for area, render in zip(areas, renders, strict=True):
        if render.label:
            canvas.text(
                [render.label], area.x0 + 2, area.y0 - label_h + LABEL_PX, LABEL_PX, color=FAINT
            )


def _key_lines(keys: tuple[Key, ...], width: float) -> list[list[str]]:
    return [wrap(fill(key.text), KEY_PX, width - 2.6 * BADGE_R) for key in keys]


def _place_badges(canvas: Canvas, slide: Slide, placed: dict[str, Placed]) -> list[str]:
    missing = []
    for number, key in enumerate(slide.keys, start=1):
        if key.select is None:
            continue
        target = placed.get(key.panel)
        if target is None or not canvas.badge(number, target, key.select, ""):
            missing.append(f"key {number} ({key.select}) found no mark on panel {key.panel}")
    return missing


def _draw_key_entry(canvas: Canvas, number: int, lines: list[str], x: float, y: float) -> float:
    canvas.root.append(
        _el(
            "circle",
            {"cx": x + BADGE_R, "cy": y - 5, "r": BADGE_R * 0.85, "fill": BADGE_FILL},
        )
    )
    canvas.root.append(
        _el(
            "text",
            {
                "x": x + BADGE_R,
                "y": y - 0.6,
                "font-family": FONT,
                "font-size": 12,
                "font-weight": "bold",
                "fill": "white",
                "text-anchor": "middle",
            },
            str(number),
        )
    )
    return canvas.text(lines, x + 2.6 * BADGE_R, y, KEY_PX)


def compose_auto(slide: Slide, renders: list[Render], height: float) -> Canvas:
    """Key beside or key below, whichever shows the smallest text larger with no failure."""

    tries = [compose_wide(slide, renders, height)]
    if slide.keys:
        tries.append(compose_key(slide, renders, height))

    def rank(canvas: Canvas) -> tuple[int, float]:
        return (-len(canvas.problems), canvas.min_css_px)

    best = max(tries, key=rank)
    best.problems.extend(p for r in renders for p in r.problems)
    return best


def compose_key(slide: Slide, renders: list[Render], height: float) -> Canvas:
    """Render left (about 68 percent), numbered key column right."""

    canvas = Canvas(ROOM_W, height)
    key_x = ROOM_W * 0.70
    note = fill(slide.footnote)
    note_lines = wrap(note, NOTE_PX, ROOM_W * 0.30) if note else []
    label_h = LABEL_PX + 6 if len(renders) > 1 or renders[0].label else 0.0
    area = Box(0, 0, ROOM_W * 0.68, height)
    areas = _split_area(area, renders, label_h)
    _panel_labels(canvas, areas, renders, label_h)
    placed = {}
    for render, box in zip(renders, areas, strict=True):
        placed[render.stem.rsplit("-", 1)[-1]] = canvas.embed(render, box)
    y = KEY_PX + 4
    for number, lines in enumerate(_key_lines(slide.keys, ROOM_W - key_x), start=1):
        y = _draw_key_entry(canvas, number, lines, key_x, y) + 8
    tail = [*note_lines]
    if not slide.caption_kept and any(p.render.dot.graph for p in canvas.placed):
        tail.append("Caption hidden on this slide.")
    if tail:
        canvas.text(
            tail, key_x, max(y + 6, height - len(tail) * NOTE_PX * 1.25), NOTE_PX, color=FAINT
        )
    canvas.problems.extend(_place_badges(canvas, slide, placed))
    return canvas


def compose_wide(slide: Slide, renders: list[Render], height: float) -> Canvas:
    """Render across the full width; the key as a compact two-column block underneath."""

    canvas = Canvas(ROOM_W, height)
    col_w = ROOM_W / 2 - 12
    key_lines = _key_lines(slide.keys, col_w)
    rows = [max(len(a), len(b)) for a, b in _pairs(key_lines)]
    key_h = sum(r * KEY_PX * 1.25 + 6 for r in rows) + 8
    note = fill(slide.footnote)
    note_lines = wrap(note, NOTE_PX, ROOM_W) if note else []
    key_h += len(note_lines) * NOTE_PX * 1.25
    label_h = LABEL_PX + 6 if len(renders) > 1 else 0.0
    area = Box(0, 0, ROOM_W, height - key_h)
    areas = _split_area(area, renders, label_h)
    _panel_labels(canvas, areas, renders, label_h)
    placed = {}
    for render, box in zip(renders, areas, strict=True):
        placed[render.stem.rsplit("-", 1)[-1]] = canvas.embed(render, box)
    y = height - key_h + KEY_PX + 6
    for pair_index, ((left, right), row) in enumerate(zip(_pairs(key_lines), rows, strict=True)):
        index = 2 * pair_index + 1
        _draw_key_entry(canvas, index, left, 0, y)
        if right:
            _draw_key_entry(canvas, index + 1, right, ROOM_W / 2, y)
        y += row * KEY_PX * 1.25 + 6
    if note_lines:
        canvas.text(note_lines, 0, y + 2, NOTE_PX, color=FAINT)
    canvas.problems.extend(_place_badges(canvas, slide, placed))
    return canvas


def _pairs(items: list[list[str]]) -> list[tuple[list[str], list[str]]]:
    return [(items[i], items[i + 1] if i + 1 < len(items) else []) for i in range(0, len(items), 2)]


def compose_grid(slide: Slide, renders: dict[str, Render], height: float) -> Canvas:
    """Several renders (or crops of one render) in a labelled grid."""

    canvas = Canvas(ROOM_W, height)
    note = fill(slide.footnote)
    note_lines = wrap(note, NOTE_PX, ROOM_W) if note else []
    key_lines = _key_lines(slide.keys, ROOM_W)
    bottom = len(note_lines) * NOTE_PX * 1.25 + sum(len(k) * KEY_PX * 1.25 + 4 for k in key_lines)
    if slide.cells:
        items = [(renders[c.panel], c.select, c.label) for c in slide.cells]
    else:
        items = [(r, None, r.label) for r in renders.values()]
    crops = [
        _crop(render, selector, canvas) if selector else render.shown
        for render, selector, _ in items
    ]
    shown = [(item, crop) for item, crop in zip(items, crops, strict=True) if crop is not None]
    cols = _grid_columns([(item[0], crop) for item, crop in shown], ROOM_W, height - bottom)
    rows = max(1, math.ceil(len(shown) / cols))
    cell_w = ROOM_W / cols
    cell_h = (height - bottom) / rows
    placed: dict[str, Placed] = {}
    for i, ((render, _selector, label), crop) in enumerate(shown):
        r, c = divmod(i, cols)
        box = Box(
            c * cell_w + 4, r * cell_h + 4, (c + 1) * cell_w - 4, (r + 1) * cell_h - LABEL_PX - 8
        )
        placed[render.stem.rsplit("-", 1)[-1]] = canvas.embed(
            render, box, crop, shared=bool(slide.cells)
        )
        canvas.text([label], box.x0 + box.w / 2, box.y1 + LABEL_PX + 2, LABEL_PX, anchor="middle")
    y = height - bottom + KEY_PX
    for number, lines in enumerate(key_lines, start=1):
        y = _draw_key_entry(canvas, number, lines, 0, y) + 4
    if note_lines:
        canvas.text(note_lines, 0, y, NOTE_PX, color=FAINT)
    canvas.problems.extend(_place_badges(canvas, slide, placed))
    canvas.problems.extend(p for r in renders.values() for p in r.problems)
    return canvas


def _grid_columns(items: list[tuple[Render, Box]], width: float, height: float) -> int:
    """The column count whose worst cell keeps the largest legibility margin."""

    def margin(cols: int) -> float:
        rows = math.ceil(len(items) / cols)
        cw, ch = width / cols - 8, height / rows - LABEL_PX - 12
        worst = 99.0
        for render, view in items:
            sizes = render.svg.font_sizes_in(view)
            need = MIN_CSS_PX / CARD_SCALE / min(sizes) if sizes else 0.1
            worst = min(worst, min(cw / view.w, ch / view.h, MAX_SCALE) / need)
        return worst

    return max(range(1, len(items) + 1), key=margin) if items else 1


def _crop(render: Render, selector: str, canvas: Canvas) -> Box | None:
    from scripts.visual_language.dot import mark_box as _mark_box

    matches = select(render.dot, fill(selector))
    canvas.witnesses[selector] = [m.name for m in matches]
    for mark in matches:
        box = _mark_box(render.svg, mark)
        if box is not None:
            return box.pad(12.0)
    canvas.problems.append(f"cell {selector} found no mark")
    return None


def compose_table(
    rows: list[list[str]], header: list[str], widths: list[float], height: float, footer: str = ""
) -> Canvas:
    """A plain table drawn as SVG text (fits the card where a Markdown table would clip)."""

    canvas = Canvas(ROOM_W, height)
    xs = [sum(widths[:i]) * ROOM_W for i in range(len(widths))]
    y = LABEL_PX + 2
    for x, word in zip(xs, header, strict=True):
        canvas.text([word], x, y, LABEL_PX, weight="bold")
    y += 6
    canvas.root.append(_el("line", {"x1": 0, "x2": ROOM_W, "y1": y, "y2": y, "stroke": "#999999"}))
    y += LABEL_PX + 4
    for row in rows:
        heights = []
        for x, w, cell in zip(xs, widths, row, strict=True):
            lines = wrap(cell, LABEL_PX, w * ROOM_W - 10)
            text_block(canvas.root, lines, x, y, LABEL_PX)
            heights.append(len(lines))
        y += max(heights) * LABEL_PX * 1.25 + 4
    canvas.note_text(LABEL_PX, "table text")
    if footer:
        canvas.text([footer], 0, height - 4, NOTE_PX, color=FAINT)
    if y > height - (NOTE_PX + 6 if footer else 0):
        canvas.problems.append(f"table runs to {y:.0f} px in a {height:.0f} px room")
    return canvas
