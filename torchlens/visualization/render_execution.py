"""Shared Graphviz layout-execution honesty seam (vizmech wave 1, lane C05).

Three defects shared one root: the render subprocess seam threw away
evidence. (1) A failed or timed-out layout left a PARTIAL artifact -- a
233-byte stub PNG -- at the user's requested path, indistinguishable from a
successful render (vizmech defect 1). (2) Graphviz prints its "graph is too
large ... scaling by N" clamp warning to stderr and EXITS 0, and every call
site discarded stderr on success, so the one signal that raster dimensions
are fiction never reached the user or the test suite (vizmech D24). (3) No
structured record of what the layout actually produced existed, so geometry
tests re-derived page facts by hand per test.

This module is the one seam all layout executions route through:

- :func:`atomic_render_target` renders to a sibling temp path and publishes
  with ``os.replace`` -- no stub ever survives a failed render.
- :func:`surface_layout_stderr` warns (``TorchLensWarning``) whenever a
  ZERO-exit layout still wrote to stderr, quoting the message.
- :class:`RenderGeometryRecord` + :func:`build_render_geometry_record` give
  every render a cheap structured disclosure: declared page geometry, raster
  pixel dimensions, the parsed clamp scale factor, the format pixel ceiling,
  the layout path, and the engine.
"""

from __future__ import annotations

import os
import re
import struct
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from ..errors._base import TorchLensWarning
from ..utils.display import user_stacklevel

__all__ = [
    "CAIRO_RASTER_PIXEL_CEILING",
    "RenderGeometryRecord",
    "atomic_render_target",
    "build_render_geometry_record",
    "is_raster_format",
    "parse_layout_scale",
    "surface_layout_stderr",
]

# cairo (graphviz's default raster backend) clamps any output dimension to
# this many pixels; a raster whose declared geometry exceeds it is silently
# scaled or cropped, which is exactly the case the stderr warning discloses.
CAIRO_RASTER_PIXEL_CEILING = 32767

_RASTER_FORMATS = frozenset({"png", "jpg", "jpeg", "gif", "bmp", "tif", "tiff", "webp"})

_SCALE_PATTERN = re.compile(r"[Ss]caling by ([0-9.eE+-]+)")
_SVG_ATTR_PATTERN = re.compile(
    r'<svg[^>]*?\bwidth="([0-9.]+)(pt|px)?"[^>]*?\bheight="([0-9.]+)(pt|px)?"',
    re.DOTALL,
)
_PDF_MEDIABOX_PATTERN = re.compile(
    rb"/MediaBox\s*\[\s*([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*\]"
)


def is_raster_format(fileformat: str) -> bool:
    """Return whether an output format is pixel-based (dpi-meaningful).

    ``dpi`` is a physical-resolution knob: it belongs to raster outputs
    only. On vector formats graphviz multiplies the coordinate space with
    it instead (vizmech D23), which is why the callers gate on this.
    """

    return fileformat.lower() in _RASTER_FORMATS


@dataclass(frozen=True)
class RenderGeometryRecord:
    """Structured disclosure of one executed layout (vizmech D24).

    Attributes
    ----------
    engine:
        Layout engine binary actually invoked (``"dot"``, ``"neato"``, ...).
    layout_path:
        Which TorchLens layout pipeline executed: ``"dot"`` or ``"rank"``.
    fileformat:
        Requested output format (``"pdf"``, ``"svg"``, ``"png"``, ...).
    output_path:
        Final published artifact path.
    stderr_text:
        Everything the engine wrote to stderr, even on exit 0. Empty string
        when the engine was silent.
    scaled_by:
        Scale factor parsed from the engine's "scaling by N" clamp warning,
        or ``None`` when no clamp fired. When set, raster pixel dimensions
        do NOT reflect the declared coordinate space.
    declared_size_points:
        ``(width, height)`` of the declared page in points (SVG width/height
        or PDF MediaBox), or ``None`` when unavailable for the format.
    raster_size_pixels:
        ``(width, height)`` pixels of a raster artifact (PNG IHDR), or
        ``None`` for vector formats.
    format_pixel_ceiling:
        The backend's hard per-dimension pixel ceiling for raster formats
        (cairo: 32767), ``None`` for vector formats.
    effective_text_scale:
        ``scaled_by`` when a clamp fired (every glyph shrank by it), else
        ``1.0``. ``None`` when no artifact was produced.
    """

    engine: str
    layout_path: str
    fileformat: str
    output_path: str
    stderr_text: str = ""
    scaled_by: float | None = None
    declared_size_points: tuple[float, float] | None = None
    raster_size_pixels: tuple[int, int] | None = None
    format_pixel_ceiling: int | None = None
    effective_text_scale: float | None = None


@contextmanager
def atomic_render_target(output_path: str) -> Iterator[str]:
    """Yield a sibling temp path; publish it to ``output_path`` on success.

    The layout engine writes to the temp path. If the body completes, the
    artifact is published with ``os.replace`` (atomic on one filesystem). If
    it raises, the temp file is removed and ``output_path`` is left exactly
    as it was -- a failed render can never leave a partial stub at the
    user's requested path.

    Parameters
    ----------
    output_path:
        Final artifact path the caller promised the user.
    """

    temp_path = f"{output_path}.tl-partial-{os.getpid()}"
    try:
        yield temp_path
    except BaseException:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise
    if os.path.exists(temp_path):
        os.replace(temp_path, output_path)


def surface_layout_stderr(stderr: str | bytes | None, *, engine: str) -> str:
    """Return decoded stderr, warning when a successful layout wrote any.

    Graphviz reports real usability failures -- the cairo "graph is too
    large ... scaling by N" clamp above all -- on stderr while exiting 0.
    Discarding it was both a usability hole and a test-integrity hole: the
    clamp message is the signal that raster pixel numbers are fiction.

    Parameters
    ----------
    stderr:
        Raw stderr from the layout subprocess (bytes, str, or ``None``).
    engine:
        Engine name for the warning text.

    Returns
    -------
    str
        Decoded stderr text (empty string when the engine was silent).
    """

    if stderr is None:
        text = ""
    elif isinstance(stderr, bytes):
        text = stderr.decode("utf-8", errors="replace")
    else:
        text = str(stderr)
    text = text.strip()
    if text:
        warnings.warn(
            TorchLensWarning(
                f"Graphviz engine {engine!r} completed but reported: {text} "
                "Remedy: read trace._last_render_geometry for the structured "
                "record; if the message names a size clamp, render direct SVG "
                "or reduce the graph with module= focus or vis_call_depth",
                code="layout_engine_stderr",
            ),
            stacklevel=user_stacklevel(),
        )
    return text


def parse_layout_scale(stderr_text: str) -> float | None:
    """Return the clamp scale factor from a "scaling by N" stderr message."""

    match = _SCALE_PATTERN.search(stderr_text)
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _read_png_dimensions(path: str) -> tuple[int, int] | None:
    """Return a PNG's IHDR pixel dimensions, or ``None`` if unreadable."""

    try:
        with open(path, "rb") as artifact:
            header = artifact.read(24)
    except OSError:
        return None
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        return None
    width, height = struct.unpack(">II", header[16:24])
    return int(width), int(height)


def _read_svg_declared_size(path: str) -> tuple[float, float] | None:
    """Return an SVG's declared width/height in points, or ``None``."""

    try:
        with open(path, encoding="utf-8", errors="replace") as artifact:
            head = artifact.read(4096)
    except OSError:
        return None
    match = _SVG_ATTR_PATTERN.search(head)
    if match is None:
        return None
    width, _w_unit, height, _h_unit = match.groups()
    try:
        return float(width), float(height)
    except ValueError:
        return None


def _read_pdf_media_box(path: str) -> tuple[float, float] | None:
    """Return a PDF's first-page MediaBox size in points, or ``None``."""

    try:
        with open(path, "rb") as artifact:
            head = artifact.read(65536)
    except OSError:
        return None
    match = _PDF_MEDIABOX_PATTERN.search(head)
    if match is None:
        return None
    try:
        llx, lly, urx, ury = (float(group) for group in match.groups())
    except ValueError:
        return None
    return urx - llx, ury - lly


def build_render_geometry_record(
    *,
    engine: str,
    layout_path: str,
    fileformat: str,
    output_path: str,
    stderr_text: str = "",
) -> RenderGeometryRecord:
    """Return the structured geometry record for one published artifact.

    Reads at most the artifact's header bytes (PNG IHDR, SVG root element,
    PDF MediaBox), so the record costs microseconds, never a re-layout.

    Parameters
    ----------
    engine:
        Layout engine binary invoked.
    layout_path:
        ``"dot"`` or ``"rank"``.
    fileformat:
        Requested output format.
    output_path:
        Published artifact path.
    stderr_text:
        Decoded stderr from the layout run (may be empty).
    """

    fmt = fileformat.lower()
    scaled_by = parse_layout_scale(stderr_text)
    declared: tuple[float, float] | None = None
    raster: tuple[int, int] | None = None
    ceiling: int | None = None
    if fmt == "svg":
        declared = _read_svg_declared_size(output_path)
    elif fmt == "pdf":
        declared = _read_pdf_media_box(output_path)
    elif fmt in _RASTER_FORMATS:
        ceiling = CAIRO_RASTER_PIXEL_CEILING
        if fmt == "png":
            raster = _read_png_dimensions(output_path)
    produced = declared is not None or raster is not None or os.path.exists(output_path)
    effective_text_scale: float | None
    if not produced:
        effective_text_scale = None
    else:
        effective_text_scale = scaled_by if scaled_by is not None else 1.0
    return RenderGeometryRecord(
        engine=engine,
        layout_path=layout_path,
        fileformat=fileformat,
        output_path=output_path,
        stderr_text=stderr_text,
        scaled_by=scaled_by,
        declared_size_points=declared,
        raster_size_pixels=raster,
        format_pixel_ceiling=ceiling,
        effective_text_scale=effective_text_scale,
    )
