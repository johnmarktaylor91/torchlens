"""The minimal deterministic SVG composer the watch renderers draw with.

A leaf: vector elements are appended in call order and serialized with no
timestamps, so the same view renders byte-identical SVG.
"""

from __future__ import annotations

import base64

__tl_layer__ = "L5"

__all__ = ("_C_TEXT", "_Svg")

#: Default text color (also the renderers' body-text color).
_C_TEXT = "#222222"


class _Svg:
    """Minimal deterministic SVG composer (vector text, no timestamps)."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self._parts: list[str] = []

    def add(self, element: str) -> None:
        """Append one raw SVG element."""

        self._parts.append(element)

    def line(  # noqa: PLR0913 -- flat coordinates mirror the SVG attribute grammar (x1/y1/x2/y2); packing them into point tuples reads worse at every call site
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: str,
        w: float = 1.0,
        dash: str | None = None,
    ) -> None:
        """Append a line element."""

        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{color}" stroke-width="{w:.2f}"{dash_attr}/>'
        )

    def rect(  # noqa: PLR0913 -- flat geometry mirrors the SVG rect attribute grammar (x/y/width/height)
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str,
        opacity: float = 1.0,
        stroke: str = "none",
    ) -> None:
        """Append a rectangle element."""

        self.add(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
            f'fill="{fill}" fill-opacity="{opacity:.2f}" stroke="{stroke}"/>'
        )

    def text(  # noqa: PLR0913 -- flat coordinates mirror the SVG text attribute grammar (x/y/text-anchor)
        self,
        x: float,
        y: float,
        content: str,
        size: int = 10,
        color: str = _C_TEXT,
        anchor: str = "start",
    ) -> None:
        """Append a vector text element (escaped)."""

        escaped = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.add(
            f'<text x="{x:.2f}" y="{y:.2f}" font-family="monospace" '
            f'font-size="{size}" fill="{color}" text-anchor="{anchor}">{escaped}</text>'
        )

    def polyline(
        self, pts: list[tuple[float, float]], color: str, w: float = 1.5, dash: str | None = None
    ) -> None:
        """Append an open polyline."""

        if len(pts) < 2:
            return
        coords = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<polyline points="{coords}" fill="none" stroke="{color}" '
            f'stroke-width="{w:.2f}"{dash_attr}/>'
        )

    def polygon(self, pts: list[tuple[float, float]], fill: str, opacity: float) -> None:
        """Append a filled polygon (the fan bands)."""

        if len(pts) < 3:
            return
        coords = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
        self.add(f'<polygon points="{coords}" fill="{fill}" fill-opacity="{opacity:.2f}"/>')

    def image_png(self, x: float, y: float, w: float, h: float, png_bytes: bytes) -> None:
        """Embed one PNG (the declared-resolution raster interior)."""

        encoded = base64.b64encode(png_bytes).decode("ascii")
        self.add(
            f'<image x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
            f'preserveAspectRatio="none" '
            f'xlink:href="data:image/png;base64,{encoded}"/>'
        )

    def to_string(self) -> str:
        """Serialize the document (root declares xmlns:xlink; cairosvg gate)."""

        body = "\n".join(self._parts)
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}">\n'
            f'<rect x="0" y="0" width="{self.width}" height="{self.height}" fill="white"/>\n'
            f"{body}\n</svg>\n"
        )
