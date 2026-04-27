"""Internal Graphviz rendering helpers shared across rendering paths.

Private module: not part of the public API. Provides ModelLog-agnostic
primitives that the single-trace ``rendering.py`` and the multi-trace
``multi_trace/visualization.py`` both rely on, so we have one canonical
implementation of file-format dispatch instead of two divergent copies.

Keep this module narrow on purpose -- only primitives that take no
ModelLog/Bundle context and can be reasoned about as pure utilities.
"""

from __future__ import annotations

import os
import subprocess
import warnings
from typing import TYPE_CHECKING

import graphviz

if TYPE_CHECKING:  # pragma: no cover - typing-only
    pass


# Recognised file extensions that callers may include on ``vis_outpath``.
# Mirrors the legacy list in ``rendering.render_graph`` (kept as a tuple
# so it stays cheap and immutable).
_KNOWN_EXTS = ("pdf", "png", "jpg", "svg", "jpeg", "bmp", "pic", "tif", "tiff")

# Default subprocess timeout for the dot/sfdp render call. Mirrors the
# legacy literal that lived inside ``rendering.render_graph``.
RENDER_TIMEOUT_SECONDS = 120


def strip_known_extension(outpath: str) -> str:
    """Strip a recognised image extension off ``outpath`` if present.

    The Graphviz Python binding wants the basename without an extension --
    it adds the extension itself based on the requested ``format``. Users
    who pass ``"out.pdf"`` should still get ``"out.pdf"`` rather than
    ``"out.pdf.pdf"``, so we trim a trailing recognised extension.
    """

    parts = outpath.split(".")
    if len(parts) > 1 and parts[-1].lower() in _KNOWN_EXTS:
        return ".".join(parts[:-1])
    return outpath


def render_dot_to_file(
    dot: "graphviz.Digraph",
    outpath: str,
    file_format: str,
    save_only: bool,
    *,
    timeout_seconds: int = RENDER_TIMEOUT_SECONDS,
    timeout_warning: str | None = None,
) -> str:
    """Render ``dot`` to ``outpath.<file_format>``, optionally previewing it.

    Mirrors the dot/save/subprocess/view flow used internally by
    ``rendering.render_graph`` and ``rendering.render_backward_graph``,
    factored out so the multi-trace renderer can share the same plumbing.

    Returns the DOT source string (``dot.source``) regardless of whether
    the subprocess render succeeded -- failures are surfaced via
    ``warnings.warn`` to match existing behaviour.
    """

    parent = os.path.dirname(os.path.abspath(outpath))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

    source_path = dot.save(outpath)
    try:
        rendered_path = f"{outpath}.{file_format}"
        cmd = [dot.engine, f"-T{file_format}", "-o", rendered_path, source_path]
        subprocess.run(
            cmd,
            timeout=timeout_seconds,
            check=True,
            capture_output=True,
        )
        if not save_only:
            graphviz.backend.viewing.view(rendered_path)
    except subprocess.TimeoutExpired:
        warnings.warn(
            timeout_warning
            or (
                f"Graphviz render timed out ({timeout_seconds}s). "
                f"DOT source saved to '{source_path}'."
            )
        )
    except subprocess.CalledProcessError as exc:
        warnings.warn(f"Graphviz render failed: {exc.stderr.decode()}")
    finally:
        if os.path.exists(source_path):
            os.remove(source_path)
    return dot.source
