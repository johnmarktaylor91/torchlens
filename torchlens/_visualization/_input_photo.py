"""The input photo: an image a trace was captured from, drawn on its input node.

``tl.trace(model, "cat.jpg")`` (or a PIL image) keeps the photo as
``trace.input``; ``trace.draw()`` puts a small PNG thumbnail of it above the
input node's own rows, in the rounded-box layout every picture node uses
(``_node_boxes``). The thumbnail is written as PNG whatever the source format, since
some Graphviz builds cannot read JPEG node images. ``trace.batch_render =
"shape_only"`` (or ``CaptureOptions(batch_render="shape_only")``) turns it off.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image

from .._utils.display import ensure_trace_visualizer_dir

if TYPE_CHECKING:
    from .._data_classes.trace import Trace

#: Longest edge, in pixels, of the photo drawn on the input node.
INPUT_PHOTO_MAX_EDGE = 160
#: File name of the thumbnail inside the trace's raw-input scratch folder.
INPUT_PHOTO_FILE = "input_photo.png"

__all__ = ("INPUT_PHOTO_FILE", "INPUT_PHOTO_MAX_EDGE", "input_photo_attrs", "single_photo")


def _raw_input_visualizer_dir(trace: Trace) -> Path:
    """Return the scratch folder for raw-input drawing artifacts (created on demand)."""

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
    """Return the input node's picture attributes for the traced photo.

    Parameters
    ----------
    trace:
        Trace being drawn; its ``input`` is the candidate photo and its
        ``batch_render`` policy ``"shape_only"`` turns the photo off.
    node_args:
        The input node's Graphviz attributes (read for nothing yet; kept so the
        call site stays one line).

    Returns
    -------
    dict[str, str] | None
        ``image`` (the PNG thumbnail) and ``tooltip``, which the node-box layout
        turns into a top cell; ``None`` when the input is not one photo, the
        policy is ``"shape_only"``, the model has more than one input node, or
        the thumbnail cannot be written.
    """

    del node_args
    photo = single_photo(getattr(trace, "input", None))
    if photo is None or getattr(trace, "batch_render", "auto") == "shape_only":
        return None
    if len(getattr(trace, "input_layers", ()) or ()) != 1:
        return None  # a photo that became several model inputs: which one is it?
    thumb = _thumbnail(photo)
    path = _raw_input_visualizer_dir(trace) / INPUT_PHOTO_FILE
    try:
        thumb.save(path, format="PNG")
    except OSError:
        return None
    width, height = photo.size
    return {
        "image": str(path),
        "tooltip": f"input photo, {width}x{height} px (thumbnail {thumb.size[0]}x{thumb.size[1]})",
    }
