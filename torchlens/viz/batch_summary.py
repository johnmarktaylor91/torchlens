"""Reusable batch summary renderers for TorchLens visualizations."""

from __future__ import annotations

import html
import math
from collections.abc import Sequence

from PIL import Image


def montage(images: Sequence[Image.Image], max_n: int, max_size: int = 600) -> Image.Image:
    """Render a PIL image batch as a thumbnail montage.

    Parameters
    ----------
    images:
        Input images to render.
    max_n:
        Maximum number of images to include.
    max_size:
        Maximum montage width or height.

    Returns
    -------
    Image.Image
        RGB montage image.

    Raises
    ------
    ValueError
        If ``max_n`` is less than one or ``images`` is empty.
    """

    if max_n < 1:
        raise ValueError("max_n must be at least 1.")
    shown = list(images[:max_n])
    if not shown:
        raise ValueError("montage requires at least one image.")

    cols = int(math.ceil(math.sqrt(len(shown))))
    rows = int(math.ceil(len(shown) / cols))
    cell_size = max(1, max_size // max(cols, rows))
    canvas = Image.new("RGB", (cols * cell_size, rows * cell_size), "white")
    for index, image in enumerate(shown):
        tile = image.convert("RGB")
        tile.thumbnail((cell_size, cell_size), Image.Resampling.LANCZOS)
        x = (index % cols) * cell_size + (cell_size - tile.width) // 2
        y = (index // cols) * cell_size + (cell_size - tile.height) // 2
        canvas.paste(tile, (x, y))
    return canvas


def text_table(strings: Sequence[str], max_n: int) -> str:
    """Render a text batch as a Graphviz HTML-like table label.

    Parameters
    ----------
    strings:
        Text items to render.
    max_n:
        Maximum number of strings to include before a ``+N more`` row.

    Returns
    -------
    str
        Graphviz HTML-like label string.

    Raises
    ------
    ValueError
        If ``max_n`` is less than one.
    """

    if max_n < 1:
        raise ValueError("max_n must be at least 1.")

    shown = list(strings[:max_n])
    rows = ["<TR><TD><B>input</B></TD></TR>"]
    rows.extend(
        f"<TR><TD ALIGN='LEFT'>{html.escape(_truncate_text(text, limit=60))}</TD></TR>"
        for text in shown
    )
    more_count = len(strings) - len(shown)
    if more_count > 0:
        rows.append(f"<TR><TD ALIGN='LEFT'>+{more_count} more</TD></TR>")
    return f"<<TABLE BORDER='0' CELLBORDER='0' CELLSPACING='0'>{''.join(rows)}</TABLE>>"


def _truncate_text(text: str, *, limit: int) -> str:
    """Return ``text`` truncated to ``limit`` characters.

    Parameters
    ----------
    text:
        Text to truncate.
    limit:
        Maximum displayed character count including the ellipsis.

    Returns
    -------
    str
        Original text or a shortened form ending in ``...``.
    """

    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."
