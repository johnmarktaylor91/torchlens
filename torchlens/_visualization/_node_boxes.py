"""Box layout for nodes that carry a picture or a list of labels.

A node that carries a picture (the input photo, an annotation image, a layer
visualizer, a thumbnail overlay, an input montage) draws as a rounded box in its
usual fill: the picture in its own table cell on top and the node's rows below,
padded so neither overlaps the other or touches the border. A decoded output
list (top-k labels, a text answer) draws as the same rounded box without the
picture. Plain nodes keep their shapes.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

from ._render_utils import html_escape, relativize_visualizer_image

if TYPE_CHECKING:
    from .._data_classes.trace import Trace

#: Longest edge, in points, of a picture drawn inside a node (2 inches).
IMAGE_MAX_POINTS = 144.0
#: Padding, in points, around the picture inside its cell.
IMAGE_CELL_PADDING = 4
#: Graphviz node margin (inches, horizontal and vertical) between the table and the border.
BOX_MARGIN = "0.14,0.08"
#: Graphviz reads a PNG without a resolution record at 96 pixels per inch.
_POINTS_PER_PIXEL = 72.0 / 96.0
#: Attributes of the old picture-behind-the-label layout, dropped once the picture is a cell.
_OVERLAY_ATTRS = frozenset({"image", "imagescale", "imagepos", "labelloc", "fixedsize"})
_SIZE_ATTRS = frozenset({"width", "height"})
_TABLE_OPEN = "<<TABLE"

__all__ = ["BOX_MARGIN", "IMAGE_MAX_POINTS", "image_box_args", "list_box_attrs"]


def _rounded_style(node_args: Mapping[str, str]) -> str:
    """Return the node's style with ``rounded`` added (and ``filled`` when it has a fill)."""

    parts = [part for part in str(node_args.get("style", "")).split(",") if part]
    parts = [part for part in parts if part != "rounded"]
    if node_args.get("fillcolor") and "filled" not in parts:
        parts.insert(0, "filled")
    return ",".join([*parts, "rounded"])


def list_box_attrs(node_args: Mapping[str, str]) -> dict[str, str]:
    """Return the rounded-box attributes for a node whose rows are a list of labels."""

    return {"shape": "box", "style": _rounded_style(node_args), "margin": BOX_MARGIN}


def _picture_points(path: Path) -> tuple[float, float] | None:
    """Return the picture's drawn size in points, capped at :data:`IMAGE_MAX_POINTS`."""

    try:
        with Image.open(path) as picture:
            width_px, height_px = picture.size
    except OSError:
        return None
    width, height = width_px * _POINTS_PER_PIXEL, height_px * _POINTS_PER_PIXEL
    scale = min(1.0, IMAGE_MAX_POINTS / max(width, height, 1.0))
    return width * scale, height * scale


def image_box_args(trace: Trace, node_args: dict[str, str]) -> dict[str, str]:
    """Move a node's picture into its own top cell and draw the node as a rounded box.

    Parameters
    ----------
    trace:
        Trace being drawn; relative picture paths resolve against its
        visualizer scratch folder (the Graphviz working directory).
    node_args:
        The node's Graphviz attributes.

    Returns
    -------
    dict[str, str]
        ``node_args`` unchanged when the node has no picture, its label is not
        an HTML table, or the picture cannot be read; otherwise new attributes
        with the picture as the label's first row (scaled to fit), the old
        overlay attributes dropped, and the rounded box in the node's fill.
    """

    from .._utils.display import ensure_trace_visualizer_dir

    picture = node_args.get("image")
    label = str(node_args.get("label", ""))
    if not picture or not label.startswith(_TABLE_OPEN):
        return node_args
    root = ensure_trace_visualizer_dir(trace)
    size = _picture_points(Path(picture) if os.path.isabs(picture) else root / picture)
    if size is None:
        return node_args
    pad = IMAGE_CELL_PADDING
    width, height = (round(side + 2 * pad) for side in size)
    cell = (
        f'<TR><TD FIXEDSIZE="TRUE" WIDTH="{width}" HEIGHT="{height}" CELLPADDING="{pad}">'
        f'<IMG SRC="{html_escape(relativize_visualizer_image(picture))}" SCALE="TRUE"/></TD></TR>'
    )
    split = label.index(">") + 1
    dropped = _OVERLAY_ATTRS | _SIZE_ATTRS
    boxed = {key: value for key, value in node_args.items() if key not in dropped}
    boxed.update(list_box_attrs(node_args), label=label[:split] + cell + label[split:])
    return boxed
