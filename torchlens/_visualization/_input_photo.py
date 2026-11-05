"""The input photo: an image a trace was captured from, drawn on its input node.

``tl.trace(model, "cat.jpg")`` (or a PIL image) keeps the photo as
``trace.input``; ``trace.draw()`` puts a small PNG thumbnail of it above the
input node's own rows, the way ``trace.annotate("input_1", image=...)`` shows
a picture. The thumbnail is written as PNG whatever the source format, since
some Graphviz builds cannot read JPEG node images. ``trace.batch_render =
"shape_only"`` (or ``CaptureOptions(batch_render="shape_only")``) turns it off.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image

from ._render_utils import html_escape, relativize_visualizer_image

if TYPE_CHECKING:
    from .._data_classes.trace import Trace

#: Longest edge, in pixels, of the photo drawn on the input node.
INPUT_PHOTO_MAX_EDGE = 160
#: File name of the thumbnail inside the trace's raw-input scratch folder.
INPUT_PHOTO_FILE = "input_photo.png"
_TABLE_OPEN = "<<TABLE"

__all__ = ["INPUT_PHOTO_FILE", "INPUT_PHOTO_MAX_EDGE", "input_photo_attrs", "single_photo"]


def _raw_input_visualizer_dir(trace: Trace) -> Path:
    """Return the scratch folder for raw-input drawing artifacts (created on demand)."""

    from .._utils.display import ensure_trace_visualizer_dir

    input_dir = ensure_trace_visualizer_dir(trace) / "raw_inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    return input_dir


def single_photo(value: Any) -> Image.Image | None:
    """Return the photo when ``value`` is one PIL image or a list or tuple of exactly one."""

    if isinstance(value, Image.Image):
        return value
    is_one_item = isinstance(value, Sequence) and not isinstance(value, str | bytes | Mapping)
    if is_one_item and len(value) == 1 and isinstance(value[0], Image.Image):
        return value[0]
    return None


def _thumbnail(photo: Image.Image) -> Image.Image:
    """Return a PNG-safe copy of ``photo`` no longer than the max edge on either side."""

    thumb = photo.copy()
    thumb.thumbnail((INPUT_PHOTO_MAX_EDGE, INPUT_PHOTO_MAX_EDGE))
    if thumb.mode not in {"RGB", "RGBA", "L"}:
        thumb = thumb.convert("RGBA" if "A" in thumb.getbands() else "RGB")
    return thumb


def input_photo_attrs(trace: Trace, node_args: Mapping[str, str]) -> dict[str, str] | None:
    """Return Graphviz attributes that draw the traced photo above the input node's rows.

    Parameters
    ----------
    trace:
        Trace being drawn; its ``input`` is the candidate photo and its
        ``batch_render`` policy ``"shape_only"`` turns the photo off.
    node_args:
        The input node's Graphviz attributes; the photo becomes the first row
        of its HTML ``label``.

    Returns
    -------
    dict[str, str] | None
        ``label`` (with the photo row) and ``tooltip``, or ``None`` when the
        input is not one photo, the policy is ``"shape_only"``, the model has
        more than one input node, the label is not an HTML table, or the
        thumbnail cannot be written.
    """

    photo = single_photo(getattr(trace, "input", None))
    label = str(node_args.get("label", ""))
    if photo is None or getattr(trace, "batch_render", "auto") == "shape_only":
        return None
    if len(getattr(trace, "input_layers", ()) or ()) != 1:
        return None  # a photo that became several model inputs: which one is it?
    if not label.startswith(_TABLE_OPEN) or ">" not in label:
        return None
    thumb = _thumbnail(photo)
    path = _raw_input_visualizer_dir(trace) / INPUT_PHOTO_FILE
    try:
        thumb.save(path, format="PNG")
    except OSError:
        return None
    source = html_escape(relativize_visualizer_image(str(path)))
    row = f'<TR><TD ALIGN="CENTER"><IMG SRC="{source}"/></TD></TR>'
    split = label.index(">") + 1
    width, height = photo.size
    return {
        "label": label[:split] + row + label[split:],
        "tooltip": f"input photo, {width}x{height} px (thumbnail {thumb.size[0]}x{thumb.size[1]})",
    }
