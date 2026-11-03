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
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from .._errors import CaptureNotActiveError, refuse_cleaned_up

if TYPE_CHECKING:
    from .._visualization.collapse_plan import CollapsePlan, CollapseSchedule, RenderContext

__tl_layer__ = "FACADE"

#: Formats whose payload is text written as-is by ``save``.
_TEXT_FORMATS = frozenset({"html", "gv", "dot"})


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
                payload = _render_dot_source(self.dot, fileformat)
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

    def collapse_plan(
        self,
        mode: Literal["auto", "max"] | float = "auto",
        context: RenderContext | None = None,
    ) -> CollapsePlan:
        """Return the diagnostic collapse plan for the trace behind this drawing.

        The plan is computed on the trace the draw retained, so it needs
        ``return_trace=True`` and an open drawing (not yet ``close()``d).

        Parameters
        ----------
        mode:
            Collapse policy to plan: ``"auto"`` for a readable overview,
            ``"max"`` for aggressive condensation, or a float in ``[0.0, 1.0]``
            selecting the monotone collapse schedule (``0.0`` is the full
            graph, ``1.0`` is byte-identical to ``"max"``).
        context:
            Optional ``RenderContext``. When omitted, the default unrolled
            Graphviz context is used.

        Returns
        -------
        CollapsePlan
            Renderer-faithful diagnostic plan. Node kinds are ``ModuleBox`` for
            collapsed module calls, ``RawOp`` for exposed operations,
            ``RepeatFold`` for representative-plus-ellipsis repeated runs,
            ``OpSegment`` and ``ChildSegment`` for condensed segment boxes, and
            ``Boundary`` for input/output or renderer boundary nodes.

        Raises
        ------
        ValueError
            If ``mode`` is not ``"auto"``, ``"max"``, or a float in
            ``[0.0, 1.0]``, or the optimizer declines to produce a plan for the
            provided context.
        TraceCleanedUpError
            If this drawing retains no trace.
        """

        return self._retained_trace("collapse_plan")._collapse_plan(mode, context)

    def collapse_schedule(self, context: RenderContext | None = None) -> CollapseSchedule:
        """Return the monotone float collapse schedule for the trace behind this drawing.

        Parameters
        ----------
        context:
            Optional ``RenderContext``. When omitted, the default unrolled
            Graphviz context is used.

        Returns
        -------
        CollapseSchedule
            Ordered schedule from ``t=0.0`` full graph to ``t=1.0`` max
            collapse. Each step includes the selected plan, visible count, and
            collapsed module-address set.

        Raises
        ------
        TraceCleanedUpError
            If this drawing retains no trace.
        """

        return self._retained_trace("collapse_schedule")._collapse_schedule(context)

    def _retained_trace(self, name: str) -> Any:
        """Return the retained trace, or refuse typed when the draw kept none."""

        if self.trace is None:
            refuse_cleaned_up(
                f"this Drawing retains no trace, so {name}() has nothing to plan on",
                remedy="draw with return_trace=True and call it before close()",
                attribute=name,
            )
        return self.trace

    def close(self) -> None:
        """Release the retained trace.

        Idempotent: a second ``close()`` is a no-op. Leaving the ``with`` block
        calls it, so ``with trace.draw(...) as drawing:`` frees the trace at
        the block's end whether or not the body raised.
        """

        self.trace = None

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
            return _render_dot_source(self.dot, "svg").decode("utf-8", errors="replace")
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


def _render_dot_source(dot: str, fileformat: str) -> bytes:
    """Render a DOT string to bytes via the graphviz pipeline (typed failure)."""

    try:
        import graphviz
    except ImportError as error:
        raise CaptureNotActiveError(
            "Rendering requires the graphviz Python package and the Graphviz system binaries.",
            code="render_engine_unavailable",
            remedy="pip install graphviz, and install Graphviz (apt install graphviz)",
        ) from error
    try:
        return bytes(graphviz.Source(dot).pipe(format=fileformat))
    except Exception as error:
        raise CaptureNotActiveError(
            f"Graphviz could not render the graph to {fileformat!r}: {error}",
            code="render_engine_unavailable",
            remedy=(
                "install the Graphviz system binaries (apt install graphviz), or "
                "choose a supported format such as 'pdf' or 'svg'"
            ),
        ) from error


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
