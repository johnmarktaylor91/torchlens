"""Annotation pictures: ``trace.annotate(label, image=...)`` made drawable as PNG.

Some Graphviz builds cannot read JPEG (or WebP) node images, so the drawing
hands Graphviz only PNG files. The conversion is split between annotate time
and draw time so that a saved trace stays small and loads as before:

- annotate time opens a path once with Pillow (refusing ``image_path_unreadable``
  when it cannot) and stores the absolute path in ``annotations["user"]["image"]``;
  a saved trace carries that path, never the picture's pixels.
- draw time passes a ``.png`` file through as is and writes any other format as
  a PNG copy into the trace's visualizer scratch folder (a runtime-only field
  that save drops), under ``annotation_images/``.
- a PIL image has no path to store, so annotate writes it to that scratch
  folder as a PNG at once and stores the PNG's path. The file lives as long as
  the trace object does: a trace saved and loaded later keeps the path but not
  the file, so annotate again after loading. A trace drawing a PNG that sits in
  another trace's scratch folder (a fork, a load in the same session) copies it
  into its own, because Graphviz resolves scratch pictures against the drawing
  trace's folder.
- a stored picture that no longer exists at draw time is not drawn (and the
  node is not an image node), instead of failing the whole Graphviz render.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Any

from .._errors import InvalidArgumentError
from .._utils.display import VISUALIZER_DIR_PREFIX, ensure_trace_visualizer_dir

#: Sub-folder of the trace's visualizer scratch folder holding annotation PNGs.
ANNOTATION_IMAGE_DIR = "annotation_images"

__all__ = ("ANNOTATION_IMAGE_DIR", "annotation_image_value", "drawable_annotation_image")


def _annotation_image_dir(trace: Any) -> Path:
    """Return the trace's scratch folder for annotation PNGs, creating it on demand."""

    image_dir = ensure_trace_visualizer_dir(trace) / ANNOTATION_IMAGE_DIR
    image_dir.mkdir(parents=True, exist_ok=True)
    return image_dir


def _png_safe(picture: Any) -> Any:
    """Return ``picture`` in a mode PNG can store (CMYK and YCbCr JPEGs become RGB)."""

    if picture.mode in {"RGB", "RGBA", "L", "LA", "1"}:
        return picture
    keeps_alpha = "A" in picture.getbands() or "transparency" in picture.info
    return picture.convert("RGBA" if keeps_alpha else "RGB")


def _is_pil_image(value: Any) -> bool:
    """Return whether ``value`` is a PIL image, without importing Pillow for other values."""

    image_module = sys.modules.get("PIL.Image")
    return image_module is not None and isinstance(value, image_module.Image)


def _check_readable(path: str) -> None:
    """Open ``path`` with Pillow once, refusing typed when it is not a readable image."""

    try:
        from PIL import Image

        with Image.open(path) as picture:
            picture.load()
    except (ImportError, OSError) as exc:
        if isinstance(exc, ImportError):
            detail = "Pillow is not installed"
        elif not os.path.isfile(path):
            detail = "the file does not exist"
        else:
            detail = f"it could not be decoded ({exc})"
        raise InvalidArgumentError(
            f"annotate(image={path!r}): the image cannot be read: {detail}.",
            code="image_path_unreadable",
            remedy=(
                "pass the path of an existing image file Pillow can open (PNG, JPEG, "
                "WebP, ...), or a PIL image; install Pillow (pip install pillow) if missing"
            ),
            path=path,
        ) from exc


def annotation_image_value(trace: Any, image: Any) -> str:
    """Return the string ``annotate(image=...)`` stores for one image argument.

    Parameters
    ----------
    trace:
        Trace being annotated; a PIL image is written into its visualizer
        scratch folder.
    image:
        A path (``str`` or ``os.PathLike``) to any image file Pillow can open,
        or a PIL image.

    Returns
    -------
    str
        The absolute path of the image file, or the path of the PNG written for
        a PIL image.

    Raises
    ------
    InvalidArgumentError
        ``image_path_unreadable`` when the path does not name an image file
        Pillow can open, or Pillow is not installed.
    """

    if _is_pil_image(image):
        picture = _png_safe(image)
        digest = hashlib.sha256(
            f"{picture.mode}:{picture.size}:".encode() + picture.tobytes()
        ).hexdigest()[:16]
        path = _annotation_image_dir(trace) / f"pil-{digest}.png"
        if not path.is_file():
            picture.save(path, format="PNG")
        return str(path)
    path_text = str(os.fspath(image) if isinstance(image, os.PathLike) else image)
    _check_readable(path_text)
    return os.path.abspath(path_text)


def _in_other_scratch_folder(trace: Any, source: str) -> bool:
    """Return whether ``source`` sits in a visualizer scratch folder not owned by ``trace``."""

    path = Path(source)
    if not any(part.startswith(VISUALIZER_DIR_PREFIX) for part in path.parts):
        return False
    own = getattr(trace, "_visualizer_dir", None)
    return own is None or not path.is_relative_to(own)


def drawable_annotation_image(trace: Any, source: str) -> str | None:
    """Return a PNG path Graphviz can draw for a stored annotation image.

    Parameters
    ----------
    trace:
        Trace being drawn; any PNG copy goes into its visualizer scratch folder.
    source:
        The value stored in ``annotations["user"]["image"]``.

    Returns
    -------
    str | None
        ``source`` itself when it is a ``.png`` file outside other traces'
        scratch folders; otherwise the path of a PNG copy in this trace's
        scratch folder, reused across draws while the source file is
        unchanged. ``None`` when the source cannot be read any more (gone, or
        changed into something Pillow cannot decode), so the node is drawn
        without the picture rather than failing the Graphviz render.
    """

    if not os.path.isfile(source):
        return None
    if source.lower().endswith(".png") and not _in_other_scratch_folder(trace, source):
        return source
    try:
        from PIL import Image

        stat = os.stat(source)
        key = f"{os.path.abspath(source)}:{stat.st_mtime_ns}:{stat.st_size}"
        target = _annotation_image_dir(trace) / (
            f"{hashlib.sha256(key.encode()).hexdigest()[:16]}.png"
        )
        if not target.is_file():
            partial = target.with_name(f"{target.stem}.partial.png")
            with Image.open(source) as picture:
                _png_safe(picture).save(partial, format="PNG")
            os.replace(partial, target)  # a failed write never leaves a reusable half file
    except (ImportError, OSError):
        return None
    return str(target)
