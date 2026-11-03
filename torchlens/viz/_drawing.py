"""``tl.viz.Drawing``: the result of every TorchLens picture call.

``Trace.draw`` and the one-call ``tl.draw`` return a ``Drawing``. It owns the
detached DOT source (or the HTML of an animation), the rendered bytes when
the picture was rendered to memory, the written path when a file was
written, and for the one-call door the input-provenance record and capture
receipt. ``save()`` re-renders from the detached source, so it works after
the trace is gone; ``show()`` displays inline in a notebook and opens the
system viewer elsewhere; ``close()`` releases a retained trace.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import weakref  # noqa: F401  (weakref.ref objects arrive through _collapse_source)
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .._errors import ArgumentConflictError, CaptureNotActiveError

__tl_layer__ = "FACADE"

#: Formats whose payload is text written as-is by ``save``.
_TEXT_FORMATS = frozenset({"html", "gv", "dot"})
#: Upper bound for one detached Graphviz render, matching a draw's forward render.
_DETACHED_RENDER_TIMEOUT_S = 120


@dataclass
class Drawing:
    """A rendered TorchLens picture, detached from the trace that made it.

    Attributes
    ----------
    dot:
        The DOT source of the graph (empty for an HTML animation).
    file_format:
        The rendered file format (``"pdf"``, ``"svg"``, ``"html"``, ...).
    path:
        Path of the written artifact, or ``None`` when nothing was written.
    _data:
        Rendered artifact bytes when the picture was rendered to memory
        (notebook inline path, HTML animations), else ``None``.
    provenance:
        The input-provenance record of the one-call door, else ``None``.
    receipt:
        Execution receipt of the one-call door's pinned capture, else ``None``.
    num_ops:
        Op count of the drawn graph when known.
    policy:
        Resolved render policy: theme, collapse, module focus, depth.
    trace:
        A retained trace when the caller asked to keep one; ``close()`` (and
        leaving the ``with`` block) releases it.
    collapse_plan:
        The collapse plan a forward draw with ``collapse="auto"``, ``"max"`` or
        a float level rendered from, computed against the drawing's resolved
        render context (property; refuses typed when the drawing has none).
    collapse_schedule:
        The monotone float collapse schedule of the drawn forward graph under
        the drawing's resolved render context (property).

    Examples
    --------
    >>> drawing = Drawing(dot="digraph { a -> b }", file_format="gv")
    >>> drawing.file_format, drawing.path is None, drawing.num_ops is None
    ('gv', True, True)
    """

    dot: str
    file_format: str = "pdf"
    path: Path | None = None
    _data: bytes | None = None
    provenance: Any | None = None
    receipt: Any | None = None
    num_ops: int | None = None
    policy: dict[str, Any] = field(default_factory=dict)
    trace: Any | None = None
    _collapse_source: Any = field(default=None, repr=False)
    _collapse_context: Any = field(default=None, repr=False)
    _collapse_tools: Any = field(default=None, repr=False)

    @property
    def collapse_plan(self) -> Any:
        """Return the collapse plan this forward drawing rendered from.

        The plan's mode is the drawing's ``collapse`` policy (``"auto"``,
        ``"max"`` or a float level) and its context is the render request the
        draw resolved, so the plan is the one the picture shows. Refuses typed
        (``collapse_plan_unavailable``) for a drawing made with
        ``collapse="none"``, a backward or combined graph, an animation, or
        once the source trace has been released.
        """

        trace, context, tools = self._collapse_inputs()
        mode = self.policy.get("collapse", "none")
        if mode == "none":
            raise CaptureNotActiveError(
                "collapse plan unavailable: this drawing was made with collapse='none'",
                code="collapse_plan_unavailable",
                remedy="draw with collapse='auto', 'max' or a float level in [0.0, 1.0]",
                reason="collapse='none'",
            )
        return tools[0](trace, mode, context)

    @property
    def collapse_schedule(self) -> Any:
        """Return the monotone float collapse schedule of the drawn forward graph.

        Steps run from ``t=0.0`` (full graph) to ``t=1.0`` (byte-identical to
        ``collapse="max"``) under the drawing's resolved render context.
        Refuses typed (``collapse_plan_unavailable``) for backward, combined or
        animation drawings and once the source trace has been released.
        """

        trace, context, tools = self._collapse_inputs()
        return tools[1](trace, context)

    def _collapse_inputs(self) -> tuple[Any, Any, Any]:
        """Return (trace, render context, collapse functions) or refuse typed."""

        if self._collapse_tools is None or self._collapse_context is None:
            raise CaptureNotActiveError(
                "collapse diagnostics unavailable: only a forward graph drawing carries "
                "a collapse plan and schedule",
                code="collapse_plan_unavailable",
                remedy="call trace.draw(graph='forward', ...) and read the members of its result",
                reason="not a forward drawing",
            )
        trace = self.trace
        if trace is None and self._collapse_source is not None:
            trace = self._collapse_source()
        if trace is None:
            raise CaptureNotActiveError(
                "collapse diagnostics unavailable: the trace this drawing was made from "
                "has been released",
                code="collapse_plan_unavailable",
                remedy="keep the trace alive while reading collapse_plan or collapse_schedule",
                reason="trace released",
            )
        return trace, self._collapse_context, self._collapse_tools

    def save(self, path: str | os.PathLike[str], *, editable: bool = False) -> Path:
        """Write the picture to ``path`` and return the path.

        The format is the path suffix, else this drawing's format. With
        ``editable=True`` the Graphviz source is written instead of a rendered
        picture, so the graph can be edited in a DOT-aware tool. Works
        without the trace: the stored source is re-rendered directly.
        """

        target = Path(path)
        suffix = target.suffix.lstrip(".").lower()
        if editable:
            payload = self.dot.encode("utf-8")
        elif self.file_format == "html":
            payload = self._data if self._data is not None else self.dot.encode("utf-8")
        elif suffix in _TEXT_FORMATS:
            payload = self.dot.encode("utf-8")
        else:
            fileformat = suffix or self.file_format
            if self._data is not None and fileformat == self.file_format:
                payload = self._data
            else:
                payload = _render_dot_source(self.dot, fileformat, self._image_root())
        tmp = target.with_name(target.name + ".tl-partial")
        tmp.write_bytes(payload)
        os.replace(tmp, target)
        return target

    def show(self) -> None:
        """Display the picture: inline in a notebook, else in the system viewer."""

        if _in_notebook():
            from IPython.display import display

            display(self)
            return
        target = self.path
        if target is None or not Path(target).exists():
            suffix = "html" if self.file_format == "html" else self.file_format
            with tempfile.NamedTemporaryFile(
                prefix="tl-drawing-", suffix=f".{suffix}", delete=False
            ) as handle:
                temp_name = handle.name
            target = self.save(temp_name)
        _open_viewer(Path(target))

    def close(self) -> None:
        """Release the retained trace.

        Idempotent: a second ``close()`` is a no-op. Leaving the ``with`` block
        calls it, so ``with trace.draw(...) as drawing:`` frees the trace at
        the block's end whether or not the body raised.
        """

        self.trace = None

    def _image_root(self) -> str | None:
        """Return the retained trace's image scratch root, if it still has one."""

        return getattr(self.trace, "_visualizer_dir", None)

    def __enter__(self) -> Drawing:
        """Enter the context manager (returns self)."""

        return self

    def __exit__(self, *exc_info: Any) -> None:
        """Release the retained trace on context exit (calls ``close``)."""

        self.close()

    def __str__(self) -> str:
        """Return the detached source: DOT for graphs, HTML for animations."""

        if self.file_format == "html" and self._data is not None:
            return self._data.decode("utf-8", errors="replace")
        return self.dot

    def _repr_html_(self) -> str | None:
        """Inline HTML for notebook display (animations only)."""

        if self.file_format == "html" and self._data is not None:
            return self._data.decode("utf-8", errors="replace")
        return None

    def _repr_svg_(self) -> str | None:
        """Inline SVG for notebook display (rendered on demand from the source)."""

        if self.file_format == "html":
            return None
        if self._data is not None and self.file_format == "svg":
            return self._data.decode("utf-8", errors="replace")
        if not self.dot:
            return None
        try:
            return _render_dot_source(self.dot, "svg", self._image_root()).decode(
                "utf-8", errors="replace"
            )
        except CaptureNotActiveError:
            return None

    def __repr__(self) -> str:
        """Return a one-line summary naming the artifact and its provenance."""

        where = str(self.path) if self.path is not None else f"in-memory {self.file_format}"
        origin = getattr(self.provenance, "origin", None)
        tail = f", input={origin}" if origin is not None else ""
        return (
            f"Drawing({where}, ops={self.num_ops}{tail}, collapse={self.policy.get('collapse')!r})"
        )


def _render_dot_source(dot: str, fileformat: str, image_root: str | None = None) -> bytes:
    """Render a DOT string to bytes the way ``draw`` renders it (typed failure).

    The detached render (notebook display of a drawing, ``save`` to another
    format) runs the same bounded Graphviz child as ``draw``: the bundled font,
    so the key's table, sized to its words in that face, fits them; relative
    node images resolved against ``image_root`` (the trace's scratch root);
    and an SVG's local image references embedded as data URIs, so the picture
    never links a file under the temp folder.
    """

    try:
        import graphviz  # noqa: F401  (the documented render dependency)
    except ImportError as error:
        raise CaptureNotActiveError(
            "Rendering requires the graphviz Python package and the Graphviz system binaries.",
            code="render_engine_unavailable",
            remedy="pip install graphviz, and install Graphviz (apt install graphviz)",
        ) from error
    from .._visualization import _render_utils
    from .._visualization._svg_compose import _inline_svg_local_images
    from .._visualization.render_execution import surface_layout_stderr

    try:
        completed = _render_utils.run_bounded_subprocess(
            ["dot", f"-T{fileformat}"],
            timeout=_DETACHED_RENDER_TIMEOUT_S,
            check=True,
            capture_output=True,
            input=dot.encode("utf-8"),
            cwd=image_root,
        )
    except Exception as error:
        raise CaptureNotActiveError(
            f"Graphviz could not render the graph to {fileformat!r}: {error}",
            code="render_engine_unavailable",
            remedy=(
                "install the Graphviz system binaries (apt install graphviz), or "
                "choose a supported format such as 'pdf' or 'svg'"
            ),
        ) from error
    surface_layout_stderr(completed.stderr, engine="dot")
    payload = bytes(completed.stdout)
    if fileformat != "svg":
        return payload
    root = Path(image_root) if image_root else None
    return _inline_svg_local_images(payload.decode("utf-8"), root).encode("utf-8")


@dataclass
class DrawTarget:
    """Where one ``draw`` call renders: the output path, the format and the inline mode.

    Both draw doors (``Trace.draw`` and ``Bundle.draw``) resolve their target
    with :func:`resolve_draw_target`, so the extension rule and the notebook
    rule are one rule.

    Attributes
    ----------
    path:
        The path to hand the renderer: the caller's path, a scratch path for an
        inline render, or ``None`` for a bare call outside a notebook (the
        caller picks its default file).
    file_format:
        The resolved format.
    stem:
        ``path`` without its recognised extension (``None`` when ``path`` is).
    inline:
        ``True`` for a bare call inside a Jupyter kernel: the picture renders to
        memory and no file is left behind.
    """

    path: str | None
    file_format: str
    stem: str | None = None
    inline: bool = False
    scratch: tempfile.TemporaryDirectory[str] | None = field(default=None, repr=False)

    def finish(self, dot: str, outpath: str, **fields: Any) -> Drawing:
        """Return the call's :class:`Drawing`; an inline render keeps the bytes, not the file.

        ``outpath`` is the path the renderer was given; ``fields`` are the
        remaining :class:`Drawing` fields (``num_ops``, ``policy``).
        """

        stem = self.stem if self.stem is not None else outpath
        written = Path(f"{stem}.{self.file_format}")
        if not self.inline:
            path = written if written.exists() else None
            return Drawing(dot=dot, file_format=self.file_format, path=path, **fields)
        data = written.read_bytes() if written.exists() else None
        if self.scratch is not None:
            self.scratch.cleanup()
        return Drawing(dot=dot, file_format=self.file_format, _data=data, **fields)


def resolve_draw_target(
    path: str | os.PathLike[str] | None,
    file_format: str | None,
    known_extensions: tuple[str, ...],
    default_format: str = "pdf",
) -> DrawTarget:
    """Resolve a draw call's path and format (the rule both draw doors share).

    The path's extension picks the format when it is one of
    ``known_extensions``; ``file_format`` applies only to a path without one
    (and to a bare call); naming a different format than the extension
    refuses. With no path inside a Jupyter kernel (the detection ``tl.draw``
    uses) the picture renders inline to a scratch directory and no file is
    written; elsewhere a bare call leaves ``path`` ``None`` for the caller's
    default file.

    Raises
    ------
    ArgumentConflictError
        ``draw_format_mismatch``: the extension and ``file_format`` disagree.
    """

    text = None if path is None else os.fspath(path)
    stem, _, suffix = (text or "").rpartition(".")
    extension = suffix.lower() if stem and suffix.lower() in known_extensions else None
    if extension is not None and file_format is not None and file_format.lower() != extension:
        raise ArgumentConflictError(
            f"path {text!r} names the format {extension!r} but file_format={file_format!r}",
            code="draw_format_mismatch",
            remedy="drop file_format=, or end the path with the extension of the format you want",
            argument="file_format",
        )
    inline = text is None and _in_notebook()
    fileformat = extension or file_format or ("svg" if inline else default_format)
    if not inline:
        return DrawTarget(text, fileformat, stem=stem if extension is not None else text)
    scratch = tempfile.TemporaryDirectory(prefix="tl-render-")
    graph = str(Path(scratch.name) / "graph")
    return DrawTarget(graph, fileformat, stem=graph, inline=True, scratch=scratch)


def _in_notebook() -> bool:
    """Best-effort notebook detection (IPython kernel with a display hook)."""

    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    return bool(shell is not None and type(shell).__name__ == "ZMQInteractiveShell")


def _open_viewer(path: Path) -> None:
    """Open ``path`` with the system viewer."""

    if sys.platform.startswith("darwin"):
        subprocess.Popen(["open", str(path)])
    elif os.name == "nt":
        os.startfile(str(path))  # type: ignore[attr-defined]
    else:
        subprocess.Popen(
            ["xdg-open", str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


__all__ = ["Drawing"]
