"""Internal Graphviz rendering helpers shared across rendering paths.

Private module: not part of the public API. Provides rendering primitives
shared by single-trace graph rendering and any internal bundle renderers, so
we have one canonical implementation of file-format dispatch, direction
translation, module cluster styling, and HTML label escaping.

Keep this module narrow on purpose -- only primitives that take no
Trace/Bundle context and can be reasoned about as pure utilities.
The orchestration that knows WHICH nodes / edges / module paths to use
lives in the per-input-shape callers, such as ``_render_dot.draw`` for
Trace.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import threading
import warnings
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, cast

import graphviz

from .._errors import InvalidArgumentError
from ..utils.display import user_stacklevel

#: Live viewer child handles (r-b6 R40-1). Retained so every launch can reap
#: previously-exited viewers; without this the discarded ``Popen`` handle left
#: one persistent zombie per process (each new spawn reaped the previous
#: corpse, so the census never returned to baseline).
_VIEWER_PROCS: list[subprocess.Popen[bytes]] = []


def _reap_finished_viewers() -> None:
    """Drop (and thereby reap) every viewer child that has already exited."""

    _VIEWER_PROCS[:] = [proc for proc in _VIEWER_PROCS if proc.poll() is None]


def _wait_and_release_viewer(proc: subprocess.Popen[bytes]) -> None:
    """Reap one viewer child the moment it exits.

    r3 b6-opus/sol R40 (carried MED): the registry alone reaped only on the
    NEXT launch, so one ``draw()`` that opened a viewer left one zombie for
    the life of the process — and retaining the ``Popen`` handle disabled
    even the finalizer's opportunistic reap. A per-viewer daemon waiter
    holds no lock, blocks nothing, and removes the handle as soon as the
    child is waited on; the launch-time sweep stays as a belt for waiter
    threads that die abnormally.
    """

    try:
        proc.wait()
    finally:
        with contextlib.suppress(ValueError):  # already swept at next launch
            _VIEWER_PROCS.remove(proc)


def _is_interactive_display_context() -> bool:
    """Return whether launching a GUI viewer is reasonable in this process.

    Returns
    -------
    bool
        ``False`` on Linux shells with no display variables or SSH sessions
        without X/Wayland forwarding; otherwise ``True``.
    """

    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if sys.platform.startswith("linux") and not has_display:
        return False
    return not (os.environ.get("SSH_CONNECTION") and not has_display)


def _open_file_quietly(filepath: str, *, announce_headless: bool = False) -> bool:
    """Open ``filepath`` in the platform default viewer, suppressing viewer noise.

    Replaces ``graphviz.backend.viewing.view`` to (a) silence ``xdg-open``
    stderr on headless Linux boxes that have no registered viewer and
    (b) skip the attempt entirely when no display is detected.

    Parameters
    ----------
    filepath:
        Rendered artifact path to open.
    announce_headless:
        If ``True``, emit one stderr line when the viewer is skipped because
        the current process appears headless.

    Returns
    -------
    bool
        ``True`` if a viewer launch was attempted, otherwise ``False``.
    """

    # In a notebook the figure is rendered inline via IPython ``display()``;
    # launching a desktop viewer is never appropriate there, and a
    # "headless, skipping auto-open" note would be misleading noise. Bail out
    # before any viewer launch or announcement.
    from ..utils.display import in_notebook

    if in_notebook():
        return False

    if not _is_interactive_display_context():
        if announce_headless:
            print(
                "torchlens.draw: headless context detected; "
                f"rendered file at {filepath}, skipping auto-open.",
                file=sys.stderr,
            )
        return False
    try:
        if sys.platform == "win32":
            os.startfile(filepath)  # type: ignore[attr-defined]
        else:
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            # r-b6 R40-1/3a: retain the handle and reap prior viewer children
            # (the discarded Popen left one persistent zombie per process),
            # and detach the viewer into its own session so a later render
            # timeout kill cannot orphan its grandchildren onto us.
            _reap_finished_viewers()
            viewer = subprocess.Popen(
                [opener, filepath],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            _VIEWER_PROCS.append(viewer)
            # Asynchronous wait so the FINAL viewer of a process is reaped
            # too, not only viewers followed by another launch (r3 R40).
            threading.Thread(
                target=_wait_and_release_viewer,
                args=(viewer,),
                name="torchlens-viewer-reaper",
                daemon=True,
            ).start()
        return True
    except (FileNotFoundError, OSError):
        return False  # no viewer available; silently skip


if TYPE_CHECKING:
    pass


# Recognised file extensions that callers may include on ``vis_outpath``.
# Mirrors the legacy list in ``_render_dot.draw`` (kept as a tuple
# so it stays cheap and immutable).
_KNOWN_EXTS = ("pdf", "png", "jpg", "svg", "jpeg", "bmp", "pic", "tif", "tiff", "dot")

# Default subprocess timeout for Graphviz render calls. Mirrors the
# legacy literal that lived inside ``_render_dot.draw``.
RENDER_TIMEOUT_SECONDS = 120

# The bounded-subprocess spawn discipline (process-group teardown, Linux
# PR_SET_PDEATHSIG parent-death binding, kill-grace escalation) lives in
# ``utils/_subprocess`` so non-visualization callers (the doctor ``dot``
# probe, the bundle git-provenance stamp) can share it without this module's
# hard ``graphviz`` import (R40). Render call sites and tests monkeypatch
# ``_render_utils.run_bounded_subprocess``, so the viz-facing wrapper lives
# here and adds the ONE viz-specific behavior on top of the shared seam.
from ..utils._subprocess import (  # noqa: E402
    _HAS_PROCESS_GROUPS,  # noqa: F401  (re-export: tests pin the spawn contract)
    run_bounded_subprocess as _run_bounded_subprocess_shared,
)


def run_bounded_subprocess(
    cmd: list[str],
    *,
    timeout: float,
    check: bool = True,
    capture_output: bool = True,
    input: bytes | str | None = None,
    cwd: str | None = None,
    text: bool = False,
) -> subprocess.CompletedProcess[Any]:
    """Run ``cmd`` through the shared bounded spawn seam, refusing typed.

    Delegates to :func:`torchlens.utils._subprocess.run_bounded_subprocess`
    (the ONE spawn discipline) and adds the viz-specific door: a missing
    Graphviz binary raises the typed install-remedy refusal instead of a raw
    ``FileNotFoundError: 'dot'`` naming neither Graphviz nor the remedy
    (b8 R65). The doctor ``dot`` probe, by contrast, wants the raw signal
    and calls the shared seam directly. Tests monkeypatch this function to
    simulate Graphviz outcomes.
    """

    try:
        return _run_bounded_subprocess_shared(
            cmd,
            timeout=timeout,
            check=check,
            capture_output=capture_output,
            input=input,
            cwd=cwd,
            text=text,
        )
    except FileNotFoundError as exc:
        # Lazy import: _render_common top-imports this module, so the typed
        # class cannot be imported at module level without minting a cycle.
        # The single most common cold-user viz failure: the Graphviz BINARY
        # is not installed (the python 'graphviz' package alone does not
        # ship it); the class carries the install remedy.
        from ._render_common import GraphvizUnavailableError

        raise GraphvizUnavailableError(
            f"TorchLens could not render this graph: the Graphviz executable "
            f"{cmd[0]!r} was not found on PATH",
            executable=cmd[0],
        ) from exc


# -- Module subgraph border widths (shared between Trace and bundle paths)
# Outermost modules get the thickest border; deeper modules thin out by depth
# fraction so visual hierarchy reads at a glance.  These constants are the
# canonical source for both ``_render_dot.py`` and the bundle renderer.
MAX_MODULE_PENWIDTH = 5
MIN_MODULE_PENWIDTH = 2
PENWIDTH_RANGE = MAX_MODULE_PENWIDTH - MIN_MODULE_PENWIDTH


_VISUALIZER_DIR_MARKER = "torchlens_visualizers_"


def relativize_visualizer_image(path: str) -> str:
    """Return an image path relative to the trace visualizer scratch root.

    r-b6 R19-6: node ``image=`` attributes used to embed the absolute
    ``tempfile.mkdtemp`` visualizer path, so every run's DOT differed in
    every image node and byte-comparison/golden hashing was impossible for
    those features. Emitting the path RELATIVE to the scratch root keeps
    per-run bytes out of the source entirely; T9 (grind-p3) supplies the
    root to Graphviz as the render subprocess working directory (the dot
    engine) instead of an in-source ``imagepath`` graph attribute, so
    user-saved DOT stays free of the per-run temp path. Paths outside a
    visualizer scratch dir (user-supplied images) pass through unchanged.
    """

    marker_index = path.find(_VISUALIZER_DIR_MARKER)
    if marker_index == -1:
        return path
    separator_index = path.find(os.sep, marker_index)
    if separator_index == -1:
        return path
    return path[separator_index + 1 :]


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


def direction_to_rankdir(direction: str) -> str:
    """Translate a TorchLens vis-direction literal into a Graphviz ``rankdir``.

    Accepted inputs: ``'bottomup'``, ``'topdown'``, ``'leftright'``.  Raises
    ``ValueError`` on anything else so callers don't silently render with the
    Graphviz default when a typo'd direction sneaks through.
    """

    if direction == "bottomup":
        return "BT"
    if direction == "leftright":
        return "LR"
    if direction == "topdown":
        return "TB"
    raise InvalidArgumentError(
        f"direction must be one of 'bottomup', 'topdown', or 'leftright'; got {direction!r}",
        code="visualization_direction_invalid",
        remedy="pass direction='bottomup', 'topdown', or 'leftright'",
        argument="direction",
    )


def compute_module_penwidth(call_depth: int, max_call_depth: int) -> float:
    """Return the cluster border width for a module at ``call_depth``.

    ``call_depth`` is 0-based (outermost is depth 0).  Outermost modules
    get the maximum penwidth; deepest modules get the minimum.  When the
    overall hierarchy has only one level (``max_call_depth == 0`` or
    ``1``) we still return a sensible value so callers don't have to
    special-case shallow models.
    """

    if max_call_depth <= 0:
        return float(MIN_MODULE_PENWIDTH + PENWIDTH_RANGE)
    nesting_fraction = (max_call_depth - call_depth) / max_call_depth
    nesting_fraction = max(0.0, min(1.0, nesting_fraction))
    return MIN_MODULE_PENWIDTH + nesting_fraction * PENWIDTH_RANGE


def html_escape(value: str) -> str:
    """Escape the three Graphviz HTML-label specials.

    Graphviz HTML-like labels reserve ``<``, ``>``, and ``&``; embedding any
    of those raw breaks the parser.  Mirrors ``html.escape`` minus the
    ``quote`` argument because Graphviz attribute values are themselves
    already inside double quotes (no need to escape ``"``).
    """

    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def make_module_cluster_label(
    title: str,
    module_type: str | None = None,
    *,
    title_already_escaped: bool = False,
) -> str:
    """Return the HTML-style label string for a module cluster.

    Mirrors the legacy format used by ``_render_flow._setup_subgraphs_recurse``:
    ``<<B>@title</B><br align='left'/>(type)<br align='left'/>>``.  The
    ``module_type`` line is omitted when no type information is available
    (which is the case for bundle clusters because the supergraph stores
    the module path string but not the underlying module class).

    ``title_already_escaped`` lets the Trace path keep its existing
    raw-title behaviour (where the title may itself contain ``:`` and is
    fed verbatim) while the bundle path can opt-in to escaping arbitrary
    user-provided strings.
    """

    title_str = title if title_already_escaped else html_escape(title)
    if module_type:
        return (
            f"<<B>@{title_str}</B><br align='left'/>({html_escape(module_type)})<br align='left'/>>"
        )
    return f"<<B>@{title_str}</B><br align='left'/>>"


def make_module_cluster_attrs(
    *,
    title: str,
    module_type: str | None,
    line_style: str,
    penwidth: float,
    fillcolor: str = "white",
    title_already_escaped: bool = False,
) -> dict[str, str]:
    """Return the standard cluster attribute dict used by both renderers.

    Centralises the Graphviz attrs that Trace and bundle clusters share:
    HTML label, bottom labelloc, ``filled,<line_style>`` style, fill colour,
    and depth-aware penwidth.  Module-type information is optional: bundle
    clusters omit it because the supergraph doesn't preserve the module
    class, while Trace clusters always pass it through.
    """

    return {
        "label": make_module_cluster_label(
            title, module_type, title_already_escaped=title_already_escaped
        ),
        "labelloc": "b",
        "style": f"filled,{line_style}",
        "fillcolor": fillcolor,
        "penwidth": str(penwidth),
        # Extra breathing room between the cluster border and the nodes/edge
        # labels inside it (graphviz default is 8pt, which crowds the border).
        "margin": "20",
    }


StyleOverride = Mapping[str, str] | Callable[[Any], Mapping[str, str] | None]


def resolve_style_override(
    override: StyleOverride | None,
    context: Any,
) -> dict[str, str]:
    """Resolve a per-node or per-edge style override.

    Parameters
    ----------
    override:
        Static attribute mapping or callable receiving ``context``.
    context:
        Node or edge context passed to callable overrides.

    Returns
    -------
    dict[str, str]
        Graphviz attributes with string values.
    """

    if override is None:
        return {}
    resolved = override(context) if callable(override) else override
    if resolved is None:
        return {}
    return {str(key): str(value) for key, value in resolved.items()}


def merge_node_style(
    base_style: Mapping[str, str],
    node_overrides: Mapping[str, StyleOverride] | None,
    graph_node_label: str,
    context: Any,
) -> dict[str, str]:
    """Return merged Graphviz node style attributes.

    Parameters
    ----------
    base_style:
        Default node attributes.
    node_overrides:
        Optional mapping from node names to static or callable overrides.
    graph_node_label:
        Node key to look up in ``node_overrides``.
    context:
        Node context passed to callable overrides.

    Returns
    -------
    dict[str, str]
        Merged node attributes.
    """

    merged = {str(key): str(value) for key, value in base_style.items()}
    if node_overrides is not None:
        merged.update(resolve_style_override(node_overrides.get(graph_node_label), context))
    return merged


def merge_edge_style(
    base_style: Mapping[str, str],
    edge_overrides: Mapping[tuple[str, str], StyleOverride] | None,
    edge_key: tuple[str, str],
    context: Any,
) -> dict[str, str]:
    """Return merged Graphviz edge style attributes.

    Parameters
    ----------
    base_style:
        Default edge attributes.
    edge_overrides:
        Optional mapping from ``(source, target)`` keys to overrides.
    edge_key:
        Edge key to look up.
    context:
        Edge context passed to callable overrides.

    Returns
    -------
    dict[str, str]
        Merged edge attributes.
    """

    merged = {str(key): str(value) for key, value in base_style.items()}
    if edge_overrides is not None:
        merged.update(resolve_style_override(edge_overrides.get(edge_key), context))
    return merged


def render_dot_to_file(
    dot: graphviz.Digraph,
    outpath: str,
    file_format: str,
    save_only: bool,
    *,
    timeout_seconds: int = RENDER_TIMEOUT_SECONDS,
    timeout_warning: str | None = None,
) -> str:
    """Render ``dot`` to ``outpath.<file_format>``, optionally previewing it.

    Mirrors the dot/save/subprocess/view flow used internally by
    ``_render_dot.draw`` and ``_render_entrypoints.render_backward_graph``,
    factored out so the multi-trace renderer can share the same plumbing.

    Returns the DOT source string (``dot.source``) regardless of whether
    the subprocess render succeeded -- failures are surfaced via
    ``warnings.warn`` to match existing behaviour.
    """

    parent = os.path.dirname(os.path.abspath(outpath))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

    source_path = dot.save(outpath)
    render_succeeded = False
    try:
        rendered_path = f"{outpath}.{file_format}"
        from .render_execution import atomic_render_target, surface_layout_stderr

        # Atomic publish + exit-0 stderr surfacing (vizmech D20/D24).
        with atomic_render_target(rendered_path) as temp_rendered_path:
            cmd = [dot.engine, f"-T{file_format}", "-o", temp_rendered_path, source_path]
            completed = run_bounded_subprocess(cmd, timeout=timeout_seconds)
        surface_layout_stderr(completed.stderr, engine=dot.engine)
        render_succeeded = True
        if not save_only:
            _open_file_quietly(rendered_path)
    except subprocess.TimeoutExpired:
        warnings.warn(
            timeout_warning
            or (
                f"Graphviz render timed out ({timeout_seconds}s). "
                f"DOT source saved to '{source_path}'."
            ),
            stacklevel=user_stacklevel(),
        )
    except subprocess.CalledProcessError as exc:
        warnings.warn(
            f"Graphviz render failed: {exc.stderr.decode()}",
            stacklevel=user_stacklevel(),
        )
    finally:
        if render_succeeded and os.path.exists(source_path):
            os.remove(source_path)
    return cast(str, dot.source)
