"""The one-call picture door ``tl.draw``.

``draw`` is a curated facade over the input resolver, the concrete capture
primitive, the theme registry, and the EXISTING renderer -- never a second
renderer. It is the ONLY default route to ``collapse="auto"`` anywhere
(decided: here and only here); ``Trace.draw()`` keeps ``collapse="none"``
and the legacy wrapper keeps its own defaults.

Viewer policy: never auto-open, anywhere, by default -- ``open_viewer=True``
is explicit opt-in. Notebook policy: inline display, no implicit file, no
viewer. Script policy: plain scripts write a file by default, switched by ONE
constant below.

Every spelling is DOCUMENTED-UNSTABLE: the names may still change.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from torch import nn

from ..viz._drawing import Drawing, _in_notebook, _open_viewer
from ._gate import require_gold
from ._primitive import ExecutionReceipt, capture_concrete
from ._provenance import InputProvenance
from ._resolve import attach_provenance, resolve_inputs

__tl_layer__ = "FACADE"

#: Script policy (both branches designed, this one constant switches them).
#: True = branch A: a bare ``tl.draw`` call in a plain script writes a
#: collision-safe ``<ModelClass>-graph.<format>`` and prints the path on one
#: line. False = branch B: no implicit filesystem action ever; the caller
#: uses ``path=`` or ``Drawing.save()``.
BARE_RENDER_WRITES_FILE_IN_SCRIPTS = True

#: Value-driven encoding sources that are DERIVED-SEMANTICS claims on a
#: synthesized-value trace: activations/gradients of random data.
_VALUE_CHANNEL_SOURCES = frozenset({"magnitude", "grad_norm"})


def _collision_safe_path(stem: str, fileformat: str) -> Path:
    """Return ``<stem>-graph.<format>``, suffixed ``-2``, ``-3``... if taken.

    The shared-generic-filename silent-overwrite trap (two renders, first
    picture gone) dies here whichever script-policy branch is selected.
    """

    candidate = Path(f"{stem}-graph.{fileformat}")
    counter = 2
    while candidate.exists():
        candidate = Path(f"{stem}-graph-{counter}.{fileformat}")
        counter += 1
    return candidate


def _gate_value_channels(trace: Any, draw_kwargs: dict[str, Any]) -> None:
    """Hard-refuse value-driven encoding channels on synthesized-value traces."""

    for channel in ("color_by", "size_by"):
        source = draw_kwargs.get(channel)
        if isinstance(source, str) and source in _VALUE_CHANNEL_SOURCES:
            require_gold(trace, f"the value-driven {channel}={source!r} render channel")


def draw(  # noqa: PLR0913 -- curated public facade (memo D2/D14): the input ladder plus the render dials ARE the spec'd surface
    model: nn.Module,
    inputs: Any = None,
    *,
    input_kwargs: dict[str, Any] | None = None,
    input_shape: Any = None,
    theme: str | None = None,
    collapse: Any = None,
    module: Any = None,
    depth: int | None = None,
    path: str | os.PathLike[str] | None = None,
    file_format: str | None = None,
    open_viewer: bool = False,
    **draw_kwargs: Any,
) -> Drawing:
    """One-call model picture: resolve input, capture metadata-only, draw.

    Three ways to give the input (with none, the shape is inferred or an error names the fix)::

        tl.draw(model, x)                            # best: your real input
        tl.draw(model, input_shape=(1, 3, 224, 224))  # your shape, random values; disclosed
        tl.draw(model)                               # inferred shape, random values; disclosed
        tl.draw(lm, "The quick brown fox")           # HF models: a string is a real input

    Parameters
    ----------
    model:
        The model to picture.
    inputs:
        Real input (tensor, tuple of tensors, or an HF string prompt).
    input_kwargs:
        Keyword inputs forwarded to the model.
    input_shape:
        Shape of a synthesized input when no real input is given; disclosed in the provenance.
    theme:
        Skin preset (``Trace.draw(theme=)``).
    collapse:
        Collapse mode; ``tl.draw`` defaults to ``"auto"`` (``Trace.draw`` keeps ``"none"``).
    module:
        Focus the picture on one module.
    depth:
        Maximum module nesting depth shown.
    path:
        Explicit paths write and report. With no path, notebooks display inline without a file;
        scripts write collision-safe ``<ModelClass>-graph.pdf`` plus one printed line when
        ``BARE_RENDER_WRITES_FILE_IN_SCRIPTS`` is True (the default).
    file_format:
        Format for a path with no extension (an extension picks it): ``"svg"`` inline, else ``"pdf"``.
    open_viewer:
        Explicit opt-in to open the written file with the system viewer. Never the default.
    **draw_kwargs:
        Passed unchanged to ``Trace.draw``; the dials above use its spellings and pass once.

    Returns
    -------
    Drawing
        The detached picture. Never ``None``.
    """

    return _capture_and_render(**locals())


def _capture_and_render(
    model: nn.Module,
    inputs: Any,
    input_kwargs: dict[str, Any] | None,
    input_shape: Any,
    draw_kwargs: dict[str, Any],
    **dials: Any,
) -> Drawing:
    """Capture the model's concrete forward and render it with the facade's dials."""

    resolved = resolve_inputs(model, inputs, input_kwargs, input_shape, verb="draw")
    trace, receipt = capture_concrete(
        model,
        resolved.plan,
        pinned_eval=True,
        metadata_only=True,
        verify_state=True,
    )
    try:
        attach_provenance(trace, resolved.provenance)
        _gate_value_channels(trace, draw_kwargs)
        return _render_finished_trace(
            trace, model, resolved.provenance, receipt, draw_kwargs=draw_kwargs, **dials
        )
    finally:
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
    path: str | os.PathLike[str] | None,
    file_format: str | None,
    open_viewer: bool,
    draw_kwargs: dict[str, Any],
) -> Drawing:
    """Drive ``Trace.draw`` for the facade and package the detached result."""

    resolved_collapse = "auto" if collapse is None else collapse
    notebook = _in_notebook()
    fileformat = file_format or ("svg" if (notebook and path is None) else "pdf")
    call_kwargs: dict[str, Any] = dict(draw_kwargs)
    call_kwargs["collapse"] = resolved_collapse
    if theme is not None:
        call_kwargs["theme"] = theme
    if module is not None:
        call_kwargs["module"] = module
    if depth is not None:
        call_kwargs["depth"] = depth
    call_kwargs["open_viewer"] = False  # never auto-open; open_viewer= is handled below
    # An explicit path keeps the caller's file_format: Trace.draw takes the format from its extension.
    call_kwargs["file_format"] = fileformat if path is None else file_format

    target: Path | None
    data: bytes | None = None
    if path is not None:
        target = Path(path)
    elif notebook or not BARE_RENDER_WRITES_FILE_IN_SCRIPTS:
        target = None
    else:
        target = _collision_safe_path(type(model).__name__, fileformat)

    if target is None:
        with tempfile.TemporaryDirectory(prefix="tl-render-") as tmpdir:
            call_kwargs["path"] = str(Path(tmpdir) / "graph")
            drawn = trace.draw(**call_kwargs)
            rendered = Path(tmpdir) / f"graph.{fileformat}"
            data = rendered.read_bytes() if rendered.exists() else None
    else:
        call_kwargs["path"] = str(target)
        drawn = trace.draw(**call_kwargs)
        fileformat = drawn.file_format
        target = drawn.path if drawn.path is not None else target
        print(f"TorchLens graph written to {target}")
        if open_viewer:
            _open_viewer(target)

    result = Drawing(
        dot=str(drawn),
        _data=data,
        file_format=fileformat,
        path=target,
        provenance=provenance,
        receipt=receipt,
        _render=drawn._render,  # re-render (display, save) the way the draw rendered
        num_ops=len(trace) if hasattr(trace, "__len__") else None,
        policy={
            "theme": theme,
            "collapse": resolved_collapse,
            "module": module,
            "depth": depth,
        },
    )
    return result
