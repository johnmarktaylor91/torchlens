"""The one-call render facade and its detached result object (B15, memo D12-D14).

``render`` is a curated facade over the input resolver, the concrete capture
primitive, the theme registry, and the EXISTING renderer -- never a second
renderer. It is the ONLY default route to ``collapse="auto"`` anywhere
(decided: here and only here); ``Trace.draw()`` keeps ``collapse="none"``
and the legacy wrapper keeps its own defaults.

Viewer policy (D14): never auto-open, anywhere, by default -- ``view=True``
is explicit opt-in. Notebook policy (settled 3:0): inline display, no
implicit file, no viewer. Script policy is maintainer fork F1; branch A (the panel's
2:1 recommendation) ships as the default, switched by ONE constant below.

Every spelling is DOCUMENTED-UNSTABLE pending the naming sprint.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from torch import nn

from .._errors import CaptureContextError, KeywordConflictError
from ._gate import require_gold
from ._primitive import ExecutionReceipt, capture_concrete
from ._provenance import InputProvenance
from ._resolve import attach_provenance, resolve_inputs

__tl_layer__ = "FACADE"

#: FORK F1 (both branches designed, this one constant switches them).
#: True = branch A: a bare render call in a plain script writes a
#: collision-safe ``<ModelClass>-graph.<format>`` and prints the path on one
#: line. False = branch B: no implicit filesystem action ever; the caller
#: uses ``file=`` or ``RenderResult.save()``.
BARE_RENDER_WRITES_FILE_IN_SCRIPTS = True

#: The curated call surface (D12). Everything else passes through to
#: ``Trace.draw`` with a typed refusal on collision with these names.
_CURATED_KWARGS = frozenset(
    {
        "input_kwargs",
        "input_size",
        "theme",
        "collapse",
        "module",
        "depth",
        "file",
        "format",
        "view",
        "return_trace",
    }
)

#: draw() spellings the curated dials resolve to; passing both the dial and
#: its underlying spelling is a collision, not an override.
_CURATED_TARGETS = frozenset(
    {
        "vis_theme",
        "collapse",
        "vis_call_depth",
        "vis_outpath",
        "vis_fileformat",
        "vis_save_only",
        "module",
    }
)

#: Value-driven encoding sources that are DERIVED-SEMANTICS claims on a
#: synthesized-value trace (memo D7): activations/gradients of random data.
_VALUE_CHANNEL_SOURCES = frozenset({"magnitude", "grad_norm"})


@dataclass
class RenderResult:
    """Detached one-call render result (memo D13). Always returned, never None.

    Owns the detached DOT source, the rendered bytes (when rendered to
    memory), the written path (when a file was written), the provenance
    record, capture facts, and the resolved render policy. Self-displays
    inline in notebooks; ``save()`` re-renders from the detached DOT source,
    so it works after the temporary trace is cleaned.

    Attributes
    ----------
    dot:
        The final DOT source (detached; never references the trace).
    data:
        Rendered artifact bytes when the render targeted memory (notebook
        inline path), else ``None``.
    format:
        The rendered file format (``"pdf"``, ``"svg"``, ...).
    path:
        Path of the written artifact, or ``None`` when nothing was written.
    provenance:
        The input-provenance record (same object the trace carries).
    receipt:
        Execution receipt from the pinned capture policy.
    total_ops:
        Op count of the captured graph (honest total; the collapse plan's
        visible/hidden split is disclosed by the legend in the artifact).
    policy:
        Resolved render policy: theme, collapse, module focus, depth.
    trace:
        The retained metadata-only trace when ``return_trace=True`` was
        requested (the EXACT verified trace on the zero-argument rung);
        ``None`` otherwise. Cleanup transfers to ``close()``.
    """

    dot: str
    data: bytes | None
    format: str
    path: Path | None
    provenance: InputProvenance
    receipt: ExecutionReceipt
    total_ops: int | None
    policy: dict[str, Any] = field(default_factory=dict)
    trace: Any | None = None

    def save(self, path: str | os.PathLike[str], format: str | None = None) -> Path:
        """Render the detached DOT source to ``path`` and return the path.

        Works without the trace: the stored DOT source is re-rendered by
        Graphviz directly. The format defaults to the path suffix, then to
        this result's format.
        """

        target = Path(path)
        fileformat = format or (target.suffix.lstrip(".") or self.format)
        payload = _render_dot_source(self.dot, fileformat)
        tmp = target.with_name(target.name + ".tl-partial")
        tmp.write_bytes(payload)
        os.replace(tmp, target)
        return target

    def close(self) -> None:
        """Release the retained trace (``return_trace=True`` transfers cleanup here)."""

        self.trace = None

    def __enter__(self) -> RenderResult:
        """Enter the context manager (returns self)."""

        return self

    def __exit__(self, *exc_info: Any) -> None:
        """Release the retained trace on context exit."""

        self.close()

    def _repr_svg_(self) -> str | None:
        """Inline SVG for notebook display (None when not rendered as SVG)."""

        if self.data is not None and self.format == "svg":
            return self.data.decode("utf-8", errors="replace")
        return None

    def __repr__(self) -> str:
        """Return a one-line summary naming the artifact and its provenance."""

        where = str(self.path) if self.path is not None else f"in-memory {self.format}"
        return (
            f"RenderResult({where}, ops={self.total_ops}, "
            f"input={self.provenance.origin}, collapse={self.policy.get('collapse')!r})"
        )


def _render_dot_source(dot: str, fileformat: str) -> bytes:
    """Render a DOT string to bytes via the graphviz pipeline (typed failure)."""

    try:
        import graphviz
    except ImportError as error:
        raise CaptureContextError(
            "Rendering requires the graphviz Python package and the Graphviz system binaries.",
            code="render_engine_unavailable",
            remedy="pip install graphviz, and install Graphviz (apt install graphviz)",
        ) from error
    try:
        return bytes(graphviz.Source(dot).pipe(format=fileformat))
    except Exception as error:
        raise CaptureContextError(
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


def _collision_safe_path(stem: str, fileformat: str) -> Path:
    """Return ``<stem>-graph.<format>``, suffixed ``-2``, ``-3``... if taken.

    The shared-generic-filename silent-overwrite trap (two renders, first
    picture gone) dies here either way fork F1 is ruled.
    """

    candidate = Path(f"{stem}-graph.{fileformat}")
    counter = 2
    while candidate.exists():
        candidate = Path(f"{stem}-graph-{counter}.{fileformat}")
        counter += 1
    return candidate


def _check_collisions(draw_kwargs: dict[str, Any]) -> None:
    """Refuse pass-through kwargs that collide with the curated dials."""

    collisions = sorted(set(draw_kwargs) & (_CURATED_KWARGS | _CURATED_TARGETS))
    if collisions:
        raise KeywordConflictError(
            f"render() already owns {collisions}: the curated dials and their "
            "underlying draw() spellings cannot also be passed through.",
            code="render_kwarg_collision",
            remedy=(
                "use the curated dial (theme=/collapse=/module=/depth=/file=/"
                "format=/view=), or drop to tl.trace(...).draw(...) for full "
                "draw() control"
            ),
            collisions=collisions,
        )


def _gate_value_channels(trace: Any, draw_kwargs: dict[str, Any]) -> None:
    """Hard-refuse value-driven encoding channels on synthesized-value traces."""

    for channel in ("color_by", "size_by"):
        source = draw_kwargs.get(channel)
        if isinstance(source, str) and source in _VALUE_CHANNEL_SOURCES:
            require_gold(trace, f"the value-driven {channel}={source!r} render channel")


def render(  # noqa: PLR0913 -- curated public facade (memo D2/D14): the input ladder plus the render dials ARE the spec'd surface
    model: nn.Module,
    input_args: Any = None,
    *,
    input_kwargs: dict[str, Any] | None = None,
    input_size: Any = None,
    theme: str | None = None,
    collapse: Any = None,
    module: Any = None,
    depth: int | None = None,
    file: str | os.PathLike[str] | None = None,
    format: str | None = None,
    view: bool = False,
    return_trace: bool = False,
    **draw_kwargs: Any,
) -> RenderResult:
    """One-call model picture: resolve input, capture metadata-only, render.

    The three input rungs (memo D2/D4)::

        render(model, x)                            # best: your real input
        render(model, input_size=(1, 3, 224, 224))  # your shape, random values; disclosed
        render(model)                               # inferred shape + random values; disclosed, or a teach
        render(lm, "The quick brown fox")           # HF models: a string is a real input

    Parameters
    ----------
    model:
        The model to picture.
    input_args:
        A real positional forward input (rung 1). Leave it and
        ``input_kwargs`` unset to use ``input_size=`` (rung 2) or inference
        (rung 3).
    input_kwargs:
        A real keyword forward input (rung 1).
    input_size:
        Declared shape(s): one flat tuple, a sequence of tuples, or a mapping
        of forward keyword names to shapes (the D4 grammar).
    theme:
        Theme preset name resolved by the visualization theme registry.
        Defaults to the renderer's own default. (The readability-budgeted
        ``"overview"`` default ships once the themes panel's visible-unit
        budget lands.)
    collapse:
        Smart-collapse mode. Defaults to ``"auto"`` -- the one decided
        default route to auto-collapse; ``Trace.draw()`` keeps ``"none"``.
        An explicit value wins and is disclosed in the result policy.
    module:
        Focus dial: restrict the picture to one module subtree.
    depth:
        Focus dial: limit the rendered call depth.
    file:
        Output target. Explicit targets always write and report. With no
        target: notebooks display inline and write NOTHING; plain scripts
        follow fork F1 branch A (collision-safe ``<ModelClass>-graph.pdf``
        + one printed line) while ``BARE_RENDER_WRITES_FILE_IN_SCRIPTS`` is
        True.
    format:
        Output format (``"pdf"``, ``"svg"``, ...). Defaults to ``"svg"``
        for inline notebook display and ``"pdf"`` otherwise.
    view:
        Explicit opt-in to open the rendered file with the system viewer.
        Never the default (memo D14).
    return_trace:
        Retain the metadata-only trace on the result (the EXACT verified
        trace on the zero-argument rung); cleanup transfers to
        ``RenderResult.close()``.
    **draw_kwargs:
        Passed through to ``Trace.draw`` unchanged; colliding with the
        curated dials refuses typed (``render_kwarg_collision``).

    Returns
    -------
    RenderResult
        The detached result (memo D13). Never ``None``.
    """

    _check_collisions(draw_kwargs)
    resolved = resolve_inputs(model, input_args, input_kwargs, input_size, verb="render")
    trace, receipt = capture_concrete(
        model,
        resolved.plan,
        pinned_eval=True,
        metadata_only=True,
        verify_state=True,
    )
    keep_trace = False
    try:
        attach_provenance(trace, resolved.provenance)
        _gate_value_channels(trace, draw_kwargs)
        result = _render_finished_trace(
            trace,
            model,
            resolved.provenance,
            receipt,
            theme=theme,
            collapse=collapse,
            module=module,
            depth=depth,
            file=file,
            format=format,
            view=view,
            draw_kwargs=draw_kwargs,
        )
        if return_trace:
            result.trace = trace
        keep_trace = return_trace
        return result
    finally:
        if not keep_trace:
            del trace


def _render_finished_trace(  # noqa: PLR0913 -- mirrors the facade's spec'd dial set verbatim (one pass-through worker)
    trace: Any,
    model: nn.Module,
    provenance: InputProvenance,
    receipt: ExecutionReceipt,
    *,
    theme: str | None,
    collapse: Any,
    module: Any,
    depth: int | None,
    file: str | os.PathLike[str] | None,
    format: str | None,
    view: bool,
    draw_kwargs: dict[str, Any],
) -> RenderResult:
    """Drive ``Trace.draw`` for the facade and package the detached result."""

    resolved_collapse = "auto" if collapse is None else collapse
    notebook = _in_notebook()
    fileformat = format or ("svg" if (notebook and file is None) else "pdf")
    call_kwargs: dict[str, Any] = dict(draw_kwargs)
    call_kwargs["collapse"] = resolved_collapse
    if theme is not None:
        call_kwargs["vis_theme"] = theme
    if module is not None:
        call_kwargs["module"] = module
    if depth is not None:
        call_kwargs["vis_call_depth"] = depth
    call_kwargs["vis_save_only"] = True  # never auto-open (D14); view= is handled below
    call_kwargs["vis_fileformat"] = fileformat

    target: Path | None
    data: bytes | None = None
    if file is not None:
        target = Path(file)
    elif notebook or not BARE_RENDER_WRITES_FILE_IN_SCRIPTS:
        target = None
    else:
        target = _collision_safe_path(type(model).__name__, fileformat)

    import tempfile

    if target is None:
        with tempfile.TemporaryDirectory(prefix="tl-render-") as tmpdir:
            call_kwargs["vis_outpath"] = str(Path(tmpdir) / "graph")
            dot = str(trace.draw(**call_kwargs))
            rendered = Path(tmpdir) / f"graph.{fileformat}"
            data = rendered.read_bytes() if rendered.exists() else None
    else:
        outpath = str(target)
        if outpath.endswith(f".{fileformat}"):
            outpath = outpath[: -(len(fileformat) + 1)]
        call_kwargs["vis_outpath"] = outpath
        dot = str(trace.draw(**call_kwargs))
        print(f"TorchLens graph written to {target}")
        if view:
            _open_viewer(target)

    result = RenderResult(
        dot=dot,
        data=data,
        format=fileformat,
        path=target,
        provenance=provenance,
        receipt=receipt,
        total_ops=getattr(trace, "num_tensors", None),
        policy={
            "theme": theme,
            "collapse": resolved_collapse,
            "module": module,
            "depth": depth,
        },
    )
    return result


def _open_viewer(path: Path) -> None:
    """Open ``path`` with the system viewer (explicit ``view=True`` only)."""

    import subprocess
    import sys

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
