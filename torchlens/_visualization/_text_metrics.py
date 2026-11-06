"""Text widths in the bundled face: as a browser draws them, and as Graphviz measures them.

Graphviz lays text out through pango at 96 dpi with hinted metrics, so every
glyph advance is rounded to a whole pixel before it is summed. At 14 pt the
common 0.556 em glyphs (lower case, digits) lose about 0.4 px each, and a long
row measures about 3 % narrower than the face's real advances. Graphviz sizes
nodes, tables and the page from that measure; a browser draws the real advances
(Helvetica, Arial and the bundled Liberation Sans share them), so a long row ran
past its box and a long header line past the page (measured on Graphviz 2.43:
the GPT-2 ``embedding_1_2`` row measures 390 pt in Graphviz, 390.75 by this
rounding, and draws 401.25 pt in a browser).

:func:`text_width` gives the drawn width; :func:`graphviz_text_width` the
width Graphviz lays out with, so callers can size for the difference.

Leaf module: imports nothing from the visualization package but the font folder.
"""

from __future__ import annotations

import functools

from ._fonts import _FONT_DIR

__all__ = ("graphviz_text_width", "text_width")

#: Graphviz's pango layout resolution, in dots per inch.
_LAYOUT_DPI = 96.0
#: Pixel size the per-glyph advances are read at (large enough that hinting is negligible).
_ADVANCE_PX = 1000


@functools.lru_cache(maxsize=4)
def _face(bold: bool, px: int) -> object | None:
    """Return the bundled face at ``px`` pixels, or ``None`` when it cannot be loaded."""

    from PIL import ImageFont

    name = "LiberationSans-Bold.ttf" if bold else "LiberationSans-Regular.ttf"
    try:
        return ImageFont.truetype(str(_FONT_DIR / name), size=px)
    except OSError:
        return None


@functools.lru_cache(maxsize=512)
def text_width(text: str, size: float, bold: bool = False) -> float:
    """Return ``text``'s drawn width in points: the bundled face's real advances.

    Parameters
    ----------
    text:
        The text as drawn.
    size:
        Point size.
    bold:
        Whether the bold face draws it.

    Returns
    -------
    float
        The advance width in points (a generous average when the face is missing).

    Examples
    --------
    >>> from torchlens._visualization._text_metrics import text_width
    >>> text_width("ab", 14.0) < text_width("ab", 14.0, bold=True)
    True
    """

    face = _face(bold, max(1, round(size * 10)))
    if face is None:
        # A stripped install without the bundled face: a generous average width.
        return 0.62 * size * len(text)
    return float(face.getlength(text)) / 10.0  # type: ignore[attr-defined]


@functools.lru_cache(maxsize=1024)
def _advance_em(char: str, bold: bool) -> float | None:
    """Return one character's advance in ems, or ``None`` when the face is missing."""

    face = _face(bold, _ADVANCE_PX)
    if face is None:
        return None
    return float(face.getlength(char)) / _ADVANCE_PX  # type: ignore[attr-defined]


def graphviz_text_width(text: str, size: float, bold: bool = False) -> float:
    """Return the width Graphviz lays ``text`` out with: advances rounded to 96 dpi pixels.

    Parameters
    ----------
    text:
        The text as drawn.
    size:
        Point size.
    bold:
        Whether the bold face draws it.

    Returns
    -------
    float
        The width in points (the drawn width when the face is missing).

    Examples
    --------
    >>> from torchlens._visualization._text_metrics import graphviz_text_width, text_width
    >>> row = "num_embeddings=50257, embedding_dim=768, padding_idx=-1"
    >>> text_width(row, 14.0) - graphviz_text_width(row, 14.0) > 10.0
    True
    """

    px_per_pt = _LAYOUT_DPI / 72.0
    total_px = 0
    for char in text:
        advance = _advance_em(char, bold)
        if advance is None:
            return text_width(text, size, bold)
        total_px += round(advance * size * px_per_pt)
    return total_px / px_per_pt
