"""Flatten Graphviz SVG into plain coordinates: no transforms, nested viewports or ``<use>``.

A slide places several renders, scaled and cropped, on one canvas. Transforms and nested
viewports draw that correctly, but a checker that reads text positions and sizes from the
attributes alone then measures every label at its unscaled size and unplaced position, and
counts the text a crop hides. Baking each element's placement into its own coordinates, and
dropping what lies outside the crop, makes the file state what it shows.
"""

from __future__ import annotations

import copy
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from dataclasses import dataclass

from scripts.visual_language.dot import SVG_NS, Box

_NUMBER_RE = re.compile(r"-?\d*\.?\d+(?:[eE][-+]?\d+)?")
_OP_RE = re.compile(r"(\w+)\s*\(([^)]*)\)")
_URL_RE = re.compile(r"url\(#([^)]+)\)")
_STRUCTURE = {"g", "a", "svg", "title", "desc", "metadata"}
_SHAPES = {"path", "polygon", "polyline", "ellipse", "circle", "rect", "line"}
_GRADIENTS = {"linearGradient", "radialGradient"}


@dataclass(frozen=True)
class Affine:
    """A uniform scale then a shift: ``p -> s * p + (tx, ty)``."""

    s: float = 1.0
    tx: float = 0.0
    ty: float = 0.0

    def then(self, outer: Affine) -> Affine:
        """This map followed by ``outer``."""

        return Affine(outer.s * self.s, outer.s * self.tx + outer.tx, outer.s * self.ty + outer.ty)

    def point(self, x: float, y: float) -> tuple[float, float]:
        return self.s * x + self.tx, self.s * y + self.ty

    def box(self, box: Box) -> Box:
        x0, y0 = self.point(box.x0, box.y0)
        x1, y1 = self.point(box.x1, box.y1)
        return Box(x0, y0, x1, y1)


def local(element: ET.Element) -> str:
    return element.tag.split("}")[-1] if isinstance(element.tag, str) else ""


def _num(value: str | None, default: float = 0.0) -> float:
    if value is None:
        return default
    found = _NUMBER_RE.match(value.strip())
    return float(found.group(0)) if found else default


def parse_transform(text: str) -> Affine:
    """The map of an SVG transform list made of translate, uniform scale and rotate(0)."""

    result = Affine()
    for op, args in _OP_RE.findall(text):
        nums = [float(n) for n in _NUMBER_RE.findall(args)]
        if op == "translate":
            step = Affine(1.0, nums[0], nums[1] if len(nums) > 1 else 0.0)
        elif op == "scale" and (len(nums) == 1 or nums[0] == nums[1]):
            step = Affine(nums[0])
        elif op == "rotate" and nums[0] == 0:
            continue
        else:
            raise ValueError(f"unsupported SVG transform {op}({args})")
        result = step.then(result)
    return result


def viewport(element: ET.Element) -> Affine:
    """The map of a nested ``<svg x y width height viewBox>`` into its parent."""

    x, y = _num(element.get("x")), _num(element.get("y"))
    nums = [float(n) for n in _NUMBER_RE.findall(element.get("viewBox", ""))]
    if len(nums) != 4 or not nums[2] or not nums[3]:
        return Affine(1.0, x, y)
    vx, vy, vw, vh = nums
    s = min(_num(element.get("width"), vw) / vw, _num(element.get("height"), vh) / vh)
    return Affine(s, x - vx * s, y - vy * s)


def walk(element: ET.Element, frame: Affine) -> Iterator[tuple[ET.Element, Affine]]:
    """Every descendant with the map from its own coordinates to the root's."""

    for child in element:
        if not isinstance(child.tag, str):
            continue
        here = frame
        if child.get("transform"):
            here = parse_transform(child.get("transform", "")).then(frame)
        if local(child) == "svg":
            here = viewport(child).then(here)
        yield child, here
        if local(child) in ("g", "a", "svg", "defs"):
            yield from walk(child, here)


def points(element: ET.Element) -> list[tuple[float, float]]:
    """Extreme points of a drawable element in its own coordinates (text is estimated)."""

    tag = local(element)
    if tag in ("ellipse", "circle"):
        cx, cy = _num(element.get("cx")), _num(element.get("cy"))
        rx = _num(element.get("rx"), _num(element.get("r")))
        ry = _num(element.get("ry"), _num(element.get("r")))
        return [(cx - rx, cy - ry), (cx + rx, cy + ry)]
    if tag in ("polygon", "polyline", "path"):
        nums = [float(n) for n in _NUMBER_RE.findall(element.get("points", element.get("d", "")))]
        return list(zip(nums[0::2], nums[1::2], strict=False))
    if tag == "text":
        x, y = _num(element.get("x")), _num(element.get("y"))
        size = _num(element.get("font-size"), 14.0)
        family = element.get("font-family", "").lower()
        em = 0.6 if "courier" in family or "mono" in family else 0.55
        width = em * size * len("".join(element.itertext()))
        anchor = element.get("text-anchor", "start")
        left = x - width / 2 if anchor == "middle" else (x - width if anchor == "end" else x)
        return [(left, y - size), (left + width, y + 0.25 * size)]
    if tag in ("image", "rect"):
        x, y = _num(element.get("x")), _num(element.get("y"))
        return [(x, y), (x + _num(element.get("width")), y + _num(element.get("height")))]
    if tag == "line":
        return [
            (_num(element.get("x1")), _num(element.get("y1"))),
            (_num(element.get("x2")), _num(element.get("y2"))),
        ]
    return []


def bounds(pts: list[tuple[float, float]]) -> Box | None:
    if not pts:
        return None
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return Box(min(xs), min(ys), max(xs), max(ys))


def _pairs(text: str, frame: Affine) -> str:
    nums = [float(n) for n in _NUMBER_RE.findall(text)]
    moved = [frame.point(x, y) for x, y in zip(nums[0::2], nums[1::2], strict=True)]
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in moved)


def _path(d: str, frame: Affine) -> str:
    """Move an absolute path made of M, C, L and Z commands (all Graphviz writes)."""

    out = []
    for command, args in re.findall(r"([A-Za-z])([^A-Za-z]*)", d):
        if command not in "MCLZ":
            raise ValueError(f"unsupported path command {command!r}")
        out.append(command + (_pairs(args, frame) if args.strip() else ""))
    return " ".join(out)


def _scaled(value: str, s: float) -> str:
    return ",".join(f"{float(n) * s:.2f}" for n in _NUMBER_RE.findall(value))


def _place(element: ET.Element, frame: Affine, prefix: str) -> ET.Element:
    """A copy of one drawable leaf (or gradient) with its placement baked in."""

    # Essential complexity: one branch per SVG element kind Graphviz writes, each with its
    # own coordinate attributes; splitting them apart would only scatter one mapping.

    out = copy.deepcopy(element)
    out.attrib.pop("transform", None)
    if local(out) not in _GRADIENTS:
        out.attrib.pop("id", None)
    s, tag = frame.s, local(out)
    get = out.get
    if tag == "text":
        out.set("x", f"{frame.point(_num(get('x')), 0)[0]:.2f}")
        out.set("y", f"{frame.point(0, _num(get('y')))[1]:.2f}")
        out.set("font-size", f"{_num(get('font-size'), 14.0) * s:.2f}")
    elif tag in ("polygon", "polyline"):
        out.set("points", _pairs(get("points", ""), frame))
    elif tag == "path":
        out.set("d", _path(get("d", ""), frame))
    elif tag in ("ellipse", "circle"):
        cx, cy = frame.point(_num(get("cx")), _num(get("cy")))
        out.set("cx", f"{cx:.2f}")
        out.set("cy", f"{cy:.2f}")
        for key in ("rx", "ry", "r"):
            if get(key) is not None:
                out.set(key, f"{_num(get(key)) * s:.2f}")
    elif tag in ("rect", "image"):
        x, y = frame.point(_num(get("x")), _num(get("y")))
        out.set("x", f"{x:.2f}")
        out.set("y", f"{y:.2f}")
        out.set("width", f"{_num(get('width')) * s:.2f}")
        out.set("height", f"{_num(get('height')) * s:.2f}")
    elif tag == "line":
        for a, b in (("x1", "y1"), ("x2", "y2")):
            x, y = frame.point(_num(get(a)), _num(get(b)))
            out.set(a, f"{x:.2f}")
            out.set(b, f"{y:.2f}")
    elif tag in _GRADIENTS:
        out.set("id", f"{prefix}-{get('id')}")
        if get("gradientUnits") == "userSpaceOnUse":
            for a, b in (("x1", "y1"), ("x2", "y2"), ("cx", "cy"), ("fx", "fy")):
                if get(a) is not None:
                    x, y = frame.point(_num(get(a)), _num(get(b)))
                    out.set(a, f"{x:.2f}")
                    out.set(b, f"{y:.2f}")
            if get("r") is not None:
                out.set("r", f"{_num(get('r')) * s:.2f}")
    if tag in _SHAPES and get("stroke") not in (None, "none", "transparent"):
        out.set("stroke-width", f"{_num(get('stroke-width'), 1.0) * s:.2f}")
        if get("stroke-dasharray"):
            out.set("stroke-dasharray", _scaled(get("stroke-dasharray", ""), s))
    for key, value in list(out.attrib.items()):
        if "url(#" in value:
            out.set(key, _URL_RE.sub(lambda m: f"url(#{prefix}-{m.group(1)})", value))
    return out


def flatten(root: ET.Element, frame: Affine, clip: Box, prefix: str) -> ET.Element:
    """One ``<g>`` holding the drawable leaves of ``root`` placed by ``frame``, clipped.

    ``clip`` is in canvas coordinates; leaves wholly outside it are dropped, so no hidden
    text remains in the file. Structure (groups, links, titles, nested viewports) is not kept.
    """

    group = ET.Element(f"{{{SVG_NS}}}g", {"clip-path": f"url(#{prefix}-clip)"})
    defs = ET.SubElement(group, f"{{{SVG_NS}}}defs")
    clip_path = ET.SubElement(defs, f"{{{SVG_NS}}}clipPath", {"id": f"{prefix}-clip"})
    ET.SubElement(
        clip_path,
        f"{{{SVG_NS}}}rect",
        {
            "x": f"{clip.x0:.2f}",
            "y": f"{clip.y0:.2f}",
            "width": f"{clip.w:.2f}",
            "height": f"{clip.h:.2f}",
        },
    )
    for element, here in walk(root, frame):
        tag = local(element)
        if tag in _GRADIENTS:
            defs.append(_place(element, here, prefix))
            continue
        if tag in _STRUCTURE or tag in ("defs", "stop", "clipPath") or not tag:
            continue
        if tag not in _SHAPES and tag not in ("text", "image"):
            raise ValueError(f"unsupported SVG element <{tag}>")
        if tag == "text" and not "".join(element.itertext()).strip():
            continue
        box = bounds(points(element))
        if box is not None and not here.box(box).overlaps(clip):
            continue
        group.append(_place(element, here, prefix))
    return group
