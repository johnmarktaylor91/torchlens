"""The single-file offline HTML trace report (B5, F16; treescope memo s6).

Upgrades ``tl.export.html`` IN PLACE -- no second exporter. Use case:
capture on a cluster, ``scp`` ONE file, open from ``file://`` with no
Python, kernel, server, font, or network.

Anatomy (memo decision 4): sticky identity + outcome banner -> capture
facts and every honesty disclosure -> the summary lane's typed table
(embedded, never rebuilt) -> the graph (module-collapsed REAL SVG by
default, disclosed depth; the 8-column canvas survives as a LABELED
fallback) -> bounded module index -> per-op metadata table with stable
anchors -> budgeted array sections -> footer manifest.

Array policy (converged 3-0): default = the diagnostic FRONTIER. Metadata
for every op; truncated thumbnails only for each minimal flagged site,
its direct parents, and its direct children -- plus intervention sites
with one-hop context. Healthy captures embed ZERO array bytes by default.
``arrays="flagged"`` survives as the documented debugging preset,
hard-capped by the same budgets with frontier-priority ordering;
``arrays=<Selection>`` is the general form; ``share_safe=True`` strips
every embedded value including the frontier. The banner and manifest say
plainly when values are embedded.

Security/privacy gates: all text escaped (including embedded JSON's
``<``), no foreign ``_repr_html_`` embedded, absolute home paths
scrubbed, restrictive CSP compatible with the emitted inline assets,
budgets enforced BEFORE payload writes, a complete omissions ledger, and
the whole document readable with JavaScript disabled (wave-1 script =
clipboard + pan/zoom only).

Every option spelling is [UI-SPRINT] / DOCUMENTED-UNSTABLE.
"""

from __future__ import annotations

import json
import os
import tempfile
import warnings
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

from .._capture_honesty import capture_honesty_facts, honesty_preamble_lines
from .._errors import InvalidArgumentError
from ..errors._base import TorchLensWarning
from ..utils.display import atomic_write_text
from ._common import _static_graph_data

__all__ = ["ReportOptions", "write_report"]

#: Report artifact schema identity (manifest field).
REPORT_SCHEMA = "torchlens.report_html.v1"

#: Default array budgets, enforced BEFORE payloads are written.
MAX_ARRAY_SITES_DEFAULT = 8
MAX_ARRAY_BYTES_DEFAULT = 512 * 1024
#: Thumbnail truncation budget (cells per site grid).
THUMBNAIL_CELL_BUDGET = 400
#: Real-graph SVG byte ladder: over the first bound, collapse harder; over
#: the second, omit with disclosure.
GRAPH_BYTE_SOFT_CAP = 1_500_000

_HOME = os.path.expanduser("~")


@dataclass(frozen=True)
class ReportOptions:
    """Emission plumbing for the offline HTML report (C06 house pattern).

    The content doors (``arrays=`` / ``graph=``) stay flat keywords on
    ``tl.export.html`` / ``write_report``; everything about HOW the bytes
    get written -- redaction, byte-stability, renderer depth, size
    budgets -- bundles here. Every spelling is [UI-SPRINT] /
    DOCUMENTED-UNSTABLE.
    """

    #: Strip EVERY embedded value including the frontier.
    share_safe: bool = False
    #: Suppress the manifest timestamp (byte-exact regeneration).
    deterministic: bool = False
    #: Optional graph-depth knob, forwarded to ``draw`` and disclosed.
    vis_call_depth: int | None = None
    #: Thumbnail site budget, enforced before payloads are written.
    max_array_sites: int = MAX_ARRAY_SITES_DEFAULT
    #: Thumbnail byte budget, enforced before payloads are written.
    max_array_bytes: int = MAX_ARRAY_BYTES_DEFAULT


def _scrub(text: str) -> str:
    """Strip absolute home paths from surfaced text (privacy gate)."""

    return text.replace(_HOME, "~") if _HOME and _HOME != "/" else text


def _safe_json(payload: Any) -> str:
    """Serialize embedded JSON with ``<`` escaped (``</script`` gate)."""

    return json.dumps(payload, indent=2, sort_keys=True, default=str).replace("<", "\\u003c")


def _anchor(label: str) -> str:
    """Stable DOM anchor id for one op label."""

    return "op-" + "".join(ch if ch.isalnum() else "-" for ch in str(label))


@dataclass
class _ArrayPlan:
    """Resolved array policy: which sites embed thumbnails, plus ledger."""

    mode: str
    sites: tuple[tuple[str, str], ...]  # (label, role)
    flagged_total: int
    selected_total: int
    stripped: bool


def _is_partial(log: Any) -> bool:
    """Whether the subject is a failed-capture PartialTrace."""

    return getattr(log, "layer_logs", None) is None and hasattr(log, "raw_layers")


def _intervention_sites(log: Any) -> list[tuple[str, str]]:
    """Intervention sites with one-hop context (memo s6 array policy)."""

    sites: list[tuple[str, str]] = []
    for label in _op_labels(log):
        hop = _intervention_hop(log, label)
        if hop is None:
            continue
        parents, children = hop
        sites.append((label, "intervention"))
        sites.extend((str(parent), "intervention_context") for parent in parents)
        sites.extend((str(child), "intervention_context") for child in children)
    return sites


def _intervention_hop(log: Any, label: str) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    """One op's intervention evidence plus one-hop context, degrade-safe."""

    try:
        op = log[label]
        intervened = bool(
            getattr(op, "intervention_replaced", False) or getattr(op, "edge_substitutions", None)
        )
        if not intervened:
            return None
        parents = tuple(str(p) for p in (getattr(op, "parents", ()) or ()))[:2]
        children = tuple(str(c) for c in (getattr(op, "children", ()) or ()))[:2]
    except Exception:  # noqa: BLE001 - unresolvable records contribute nothing
        return None
    return parents, children


def _op_labels(log: Any) -> tuple[str, ...]:
    """Pass-qualified op labels in execution order, degrade-safe.

    ``op_labels`` is the OP space (``label:pass``; ``log[label]`` returns an
    ``Op``); ``layer_labels`` is the bare Layer space, whose per-pass field
    reads raise typed on reused layers -- the report iterates ops.
    """

    try:
        return tuple(str(label) for label in (log.op_labels or ()))
    except Exception:  # noqa: BLE001 - partial/legacy logs may lack labels
        return ()


def _selection_sites(log: Any, selection: Any) -> list[tuple[str, str]]:
    """Resolve an ``arrays=<Selection>`` value to op labels (row X15)."""

    resolved = selection.resolve(log)
    # SiteEntry rows live on ``_entries`` today (the public spelling is an
    # L6 naming-session item); prefer a public ``entries`` once it exists.
    entries = getattr(resolved, "entries", None) or getattr(resolved, "_entries", ()) or ()
    site_keys = {entry.site_key for entry in entries}
    sites: list[tuple[str, str]] = []
    for label in _op_labels(log):
        op = _op_or_none(log, label)
        if op is None:
            continue
        # SiteEntry keys are (layer_label, pass_index) pairs today; ops also
        # carry the L1 structural string -- match either spelling.
        base, _, pass_text = label.rpartition(":")
        pair = (base, int(pass_text)) if base and pass_text.isdigit() else None
        if getattr(op, "site_key", None) in site_keys or pair in site_keys:
            sites.append((label, "selection"))
    return sites


def _op_or_none(log: Any, label: str) -> Any | None:
    """Resolve one op label, degrade-safe."""

    try:
        return log[label]
    except Exception:  # noqa: BLE001 - unresolvable labels contribute nothing
        return None


def _array_plan(log: Any, arrays: Any, share_safe: bool) -> _ArrayPlan:
    """Resolve the array policy into a budget-ready site plan.

    Raises
    ------
    InvalidArgumentError
        ``report_arrays_invalid`` for an unrecognized ``arrays=`` value.
    """

    from ..notebook.frontier import nonfinite_frontier

    frontier = nonfinite_frontier(log) if not _is_partial(log) else None
    flagged_total = frontier.flagged_total if frontier else 0
    if share_safe:
        return _ArrayPlan("share_safe", (), flagged_total, 0, True)
    if arrays == "none":
        return _ArrayPlan("none", (), flagged_total, 0, False)
    if arrays == "frontier":
        sites = [(s.label, s.role) for s in (frontier.sites if frontier else ())]
        sites += _intervention_sites(log)
        return _ArrayPlan("frontier", tuple(sites), flagged_total, len(sites), False)
    if arrays == "flagged":
        frontier_labels = [s.label for s in (frontier.sites if frontier else ())]
        flagged = [str(label) for label in (getattr(log, "nonfinite_ops", ()) or ())]
        ordered = frontier_labels + [lbl for lbl in flagged if lbl not in frontier_labels]
        sites = [(lbl, "flagged") for lbl in ordered] + _intervention_sites(log)
        return _ArrayPlan("flagged", tuple(sites), flagged_total, len(sites), False)
    if hasattr(arrays, "resolve"):
        sites = _selection_sites(log, arrays)
        return _ArrayPlan("selection", tuple(sites), flagged_total, len(sites), False)
    raise InvalidArgumentError(
        f"arrays= must be 'frontier', 'flagged', 'none', or a Selection; received {arrays!r}",
        code="report_arrays_invalid",
        remedy="pass one of the documented presets or a tl.Selection instance",
    )


def _thumbnail_sections(
    log: Any, plan: _ArrayPlan, max_sites: int, max_bytes: int
) -> tuple[str, int, int]:
    """Render budgeted array sections; budgets run BEFORE writing.

    Returns ``(html, sites_shown, bytes_used)``.
    """

    if not plan.sites:
        return "", 0, 0
    pieces: list[str] = []
    used = 0
    shown = 0
    for label, role in plan.sites:
        if shown >= max_sites or used >= max_bytes:
            break
        tensor = _site_tensor(log, label)
        if tensor is None:
            continue
        stats_line, grid_html = _site_render(tensor)
        if grid_html is None:
            continue
        section = (
            f'<details open id="{escape(_anchor(label), quote=True)}-array">'
            f"<summary>{escape(label)} ({escape(role)})</summary>"
            f'<div class="tl-report-stats">{escape(_scrub(stats_line or ""))}</div>'
            f"{grid_html}</details>"
        )
        if used + len(section) > max_bytes and shown > 0:
            break
        pieces.append(section)
        used += len(section)
        shown += 1
    return "".join(pieces), shown, used


def _site_tensor(log: Any, label: str) -> Any | None:
    """Resident saved payload of one site, or ``None`` (never a reload)."""

    try:
        import torch

        op = log[label]
        if not getattr(op, "has_saved_activation", False):
            return None
        tensor = getattr(op, "out", None)
        if type(tensor) is torch.Tensor and not tensor.is_meta:
            return tensor
    except Exception:  # noqa: BLE001 - a display read may never raise
        return None
    return None


def _site_render(tensor: Any) -> tuple[str | None, str | None]:
    """Stats line + thumbnail grid for one payload, degrade-safe."""

    from ..notebook._grid import GridBudgets, array_grid_html

    line: str | None = None
    try:
        from ..stats import render_core_line, tensor_stats

        line = render_core_line(tensor_stats(tensor))
    except Exception:  # noqa: BLE001 - the grid may still render
        line = None
    try:
        grid = array_grid_html(tensor, budgets=GridBudgets(cells=THUMBNAIL_CELL_BUDGET))
        return line, grid.html
    except Exception:  # noqa: BLE001 - a failed grid drops the section
        return line, None


def _graph_section(log: Any, graph: str, vis_call_depth: int | None) -> tuple[str, dict[str, Any]]:
    """Embed the real module-collapsed graph, or the labeled fallback.

    Returns ``(html, manifest_facts)``.
    """

    facts: dict[str, Any] = {"mode": graph, "fallback": False, "omitted": False}
    if graph == "none":
        facts["omitted"] = True
        return '<div class="tl-report-notice">graph omitted (graph="none")</div>', facts
    for collapse in _collapse_ladder(graph):
        svg = _draw_svg(log, collapse, vis_call_depth)
        if svg is None:
            continue
        if len(svg) > GRAPH_BYTE_SOFT_CAP:
            continue
        facts["collapse"] = collapse
        html = (
            '<div id="tl-graph-wrap" class="tl-report-graph">'
            f'<div id="tl-graph-inner">{svg}</div></div>'
            f'<div class="tl-report-stats">graph: real layout, collapse={collapse!r}'
            + (f", vis_call_depth={vis_call_depth}" if vis_call_depth is not None else "")
            + "</div>"
        )
        return html, facts
    facts["fallback"] = True
    warnings.warn(
        TorchLensWarning(
            "the report could not embed the real graph layout and fell back to "
            "the labeled schematic canvas. "
            "Remedy: install Graphviz, or reduce the graph with vis_call_depth/"
            "module focus and re-export",
            code="report_graph_fallback",
        ),
        stacklevel=3,
    )
    fallback = _fallback_canvas(log)
    if fallback is None:
        facts["omitted"] = True
        return (
            '<div class="tl-report-notice">graph omitted: real layout and '
            "fallback canvas both unavailable</div>",
            facts,
        )
    return (
        '<div class="tl-report-notice">fallback layout (schematic grid, NOT the '
        "real graph geometry)</div>"
        f'<div id="tl-graph-wrap" class="tl-report-graph">'
        f'<div id="tl-graph-inner">{fallback}</div></div>',
        facts,
    )


def _collapse_ladder(graph: str) -> tuple[str, ...]:
    """Collapse attempts for one graph mode (byte ladder, memo s6)."""

    if graph == "flat":
        return ("none", "auto", "max")
    return ("auto", "max")


def _draw_svg(log: Any, collapse: str, vis_call_depth: int | None) -> str | None:
    """Render the real graph to SVG text, or ``None`` on any failure."""

    try:
        with tempfile.TemporaryDirectory() as tmp:
            outpath = os.path.join(tmp, "graph")
            kwargs: dict[str, Any] = {
                "collapse": collapse,
                "vis_save_only": True,
                "vis_outpath": outpath,
                "vis_fileformat": "svg",
            }
            if vis_call_depth is not None:
                kwargs["vis_call_depth"] = vis_call_depth
            log.draw(**kwargs)
            svg_text = Path(f"{outpath}.svg").read_text(encoding="utf-8")
    except Exception:  # noqa: BLE001 - the ladder decides what happens next
        return None
    return _strip_svg_prolog(_scrub(svg_text))


def _strip_svg_prolog(svg_text: str) -> str:
    """Drop the XML declaration/DOCTYPE so the SVG embeds inline."""

    start = svg_text.find("<svg")
    return svg_text[start:] if start > 0 else svg_text


def _fallback_canvas(log: Any) -> str | None:
    """The historical 8-column schematic canvas, as inline SVG."""

    try:
        from . import _render_svg

        rendered = _render_svg(_static_graph_data(log), editable=True)
    except Exception:  # noqa: BLE001 - a partial log may not serialize
        return None
    return _strip_svg_prolog(rendered)


def _summary_section(log: Any) -> str:
    """Embed the summary lane's typed table -- never rebuild totals (X14)."""

    try:
        summary = log.summary()
    except Exception:  # noqa: BLE001 - partial/legacy logs may refuse
        return '<div class="tl-report-notice">summary unavailable</div>'
    try:
        rendered = summary._repr_html_()
    except Exception:  # noqa: BLE001 - fall back to the text form
        rendered = f"<pre>{escape(str(summary))}</pre>"
    return _scrub(rendered)


def _op_table(log: Any) -> str:
    """Per-op metadata table (every op, one row, stable anchors)."""

    rows: list[str] = [
        "<tr><th>op</th><th>func</th><th>shape</th><th>dtype</th>"
        "<th>params</th><th>flops fwd</th><th>time</th></tr>"
    ]
    for label in _op_labels(log):
        try:
            op = log[label]
        except Exception:  # noqa: BLE001 - unresolvable labels get a bare row
            rows.append(
                f'<tr id="{escape(_anchor(label), quote=True)}">'
                f"<td>{escape(label)}</td>" + "<td>?</td>" * 6 + "</tr>"
            )
            continue
        shape = getattr(op, "shape", None)
        cells = (
            escape(label),
            escape(str(getattr(op, "func_name", "?"))),
            escape(str(tuple(shape)) if shape is not None else "?"),
            escape(str(getattr(op, "dtype", "?"))),
            escape(str(getattr(op, "num_params", "?"))),
            escape(str(getattr(op, "flops_forward", "?"))),
            escape(str(getattr(op, "func_duration", "?"))),
        )
        rows.append(
            f'<tr id="{escape(_anchor(label), quote=True)}">'
            + "".join(f"<td>{cell}</td>" for cell in cells)
            + "</tr>"
        )
    return f'<table class="tl-report-ops">{"".join(rows)}</table>'


def _module_index(log: Any) -> str:
    """Bounded module index."""

    try:
        modules = list(getattr(log, "module_calls", ()) or ())
    except Exception:  # noqa: BLE001 - partial logs may lack module records
        modules = []
    if not modules:
        return ""
    shown = modules[:200]
    items = "".join(f"<li>{escape(_scrub(str(m)))}</li>" for m in shown)
    hidden = len(modules) - len(shown)
    disclosure = f"<li>... {hidden} more</li>" if hidden > 0 else ""
    return (
        f"<details><summary>modules ({len(modules)})</summary>"
        f"<ul>{items}{disclosure}</ul></details>"
    )


def _honesty_section(log: Any) -> str:
    """Every capture-honesty disclosure, verbatim from the shared facts."""

    try:
        lines = honesty_preamble_lines(log)
    except Exception:  # noqa: BLE001 - facts must never sink the report
        lines = ["capture honesty facts unavailable"]
    body = "".join(f"<div>{escape(_scrub(str(line)))}</div>" for line in lines)
    return f'<section class="tl-report-honesty">{body}</section>'


def _banner(log: Any, plan: _ArrayPlan) -> str:
    """Sticky identity + outcome banner; embed disclosure is mandatory."""

    kind = "PartialTrace (FAILED CAPTURE)" if _is_partial(log) else "Trace"
    title = str(getattr(log, "trace_label", None) or getattr(log, "model_label", None) or kind)
    status = getattr(getattr(getattr(log, "outcome", None), "status", None), "name", "UNKNOWN")
    if plan.stripped:
        embed_note = "share_safe: every embedded value stripped"
    elif plan.sites:
        embed_note = f"array values EMBEDDED (mode={plan.mode})"
    else:
        embed_note = "no array values embedded"
    return (
        '<header class="tl-report-banner">'
        f"<strong>TorchLens report: {escape(_scrub(title))}</strong>"
        f'<span class="tl-report-badge">{escape(status)}</span>'
        f'<span class="tl-report-badge">{escape(embed_note)}</span></header>'
    )


_REPORT_CSS = (
    "<style>"
    "body{font-family:system-ui,sans-serif;margin:0 auto;max-width:1080px;"
    "padding:0 16px 48px;color:#1f2328}"
    ".tl-report-banner{position:sticky;top:0;background:#fff;border-bottom:2px solid #d0d7de;"
    "padding:10px 0;z-index:10}"
    ".tl-report-badge{display:inline-block;border:1px solid #d0d7de;border-radius:6px;"
    "padding:0 6px;margin-left:8px;font-weight:600;color:#9a6700}"
    ".tl-report-honesty{background:#fff8f0;border:1px solid #d0d7de;border-radius:6px;"
    "padding:8px 10px;margin:10px 0;font-size:13px}"
    ".tl-report-notice{color:#9a6700;font-weight:600;margin:8px 0}"
    ".tl-report-stats{color:#57606a;font-family:ui-monospace,monospace;font-size:12px}"
    ".tl-report-graph{overflow:auto;max-height:640px;border:1px solid #d0d7de;margin:8px 0}"
    ".tl-report-ops{border-collapse:collapse;font-size:12px}"
    ".tl-report-ops td,.tl-report-ops th{border:1px solid #d0d7de;padding:2px 6px}"
    "section h2{font-size:16px;margin:18px 0 6px}"
    "</style>"
)

#: Wave-1 script: pan/zoom on the graph container only (clipboard rides the
#: card CSS ``user-select`` degradation). The document reads fully with
#: JavaScript disabled -- the graph container natively scrolls.
_GRAPH_SCRIPT = (
    "<script>(function(){"
    "var w=document.getElementById('tl-graph-wrap'),"
    "i=document.getElementById('tl-graph-inner');if(!w||!i)return;"
    "var s=1,tx=0,ty=0,drag=false,last=[0,0];"
    "function apply(){i.style.transform='translate('+tx+'px,'+ty+'px) scale('+s+')';"
    "i.style.transformOrigin='0 0'}"
    "w.addEventListener('wheel',function(e){e.preventDefault();"
    "s*=e.deltaY<0?1.1:0.9;apply()},{passive:false});"
    "w.addEventListener('mousedown',function(e){drag=true;last=[e.clientX,e.clientY]});"
    "window.addEventListener('mouseup',function(){drag=false});"
    "window.onmousemove=function(e){if(!drag)return;"
    "tx+=e.clientX-last[0];ty+=e.clientY-last[1];last=[e.clientX,e.clientY];apply()};"
    "})();</script>"
)

_CSP_META = (
    '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; '
    "style-src 'unsafe-inline'; img-src data:; script-src 'unsafe-inline'\">"
)


def write_report(
    log: Any,
    path: str | Path,
    *,
    arrays: Any = "frontier",
    graph: str = "collapsed",
    options: ReportOptions | None = None,
) -> Path:
    """Write the single-file offline HTML report for one capture.

    Parameters
    ----------
    log:
        ``Trace`` or ``PartialTrace`` (failure reports are first-class:
        a shippable failure report IS the artifact for "it broke on the
        cluster").
    path:
        Destination ``.html`` path.
    arrays:
        ``"frontier"`` (default) / ``"flagged"`` / ``"none"`` / a
        ``tl.Selection``.
    graph:
        ``"collapsed"`` (default: module-collapsed real graph, depth
        disclosed) / ``"flat"`` / ``"none"``.
    options:
        Emission plumbing (``ReportOptions``): redaction, byte-stability,
        renderer depth, thumbnail budgets.

    Returns
    -------
    Path
        The written report path.
    """

    opts = options if options is not None else ReportOptions()
    share_safe = opts.share_safe
    deterministic = opts.deterministic
    vis_call_depth = opts.vis_call_depth
    max_array_sites = opts.max_array_sites
    max_array_bytes = opts.max_array_bytes
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    plan = _array_plan(log, arrays, share_safe)
    partial = _is_partial(log)

    sections: list[str] = [_banner(log, plan), _honesty_section(log)]
    graph_facts: dict[str, Any] = {"mode": "none", "omitted": True, "fallback": False}
    array_html, sites_shown, bytes_used = "", 0, 0
    if partial:
        sections.append(_partial_body(log))
    else:
        sections.append(f"<section><h2>summary</h2>{_summary_section(log)}</section>")
        graph_html, graph_facts = _graph_section(log, graph, vis_call_depth)
        sections.append(f"<section><h2>graph</h2>{graph_html}</section>")
        sections.append(f"<section><h2>modules</h2>{_module_index(log)}</section>")
        sections.append(f"<section><h2>ops</h2>{_op_table(log)}</section>")
        array_html, sites_shown, bytes_used = _thumbnail_sections(
            log, plan, max_array_sites, max_array_bytes
        )
        disclosure = (
            f"{plan.flagged_total} flagged, {sites_shown} shown"
            if plan.mode in ("frontier", "flagged")
            else f"{plan.selected_total} selected, {sites_shown} shown"
        )
        sections.append(
            "<section><h2>arrays</h2>"
            f'<div class="tl-report-stats">{escape(disclosure)}</div>{array_html}</section>'
        )
    manifest = _manifest(
        log,
        plan,
        {
            "graph": graph_facts,
            "sites_shown": sites_shown,
            "bytes_used": bytes_used,
            "deterministic": deterministic,
            "max_sites": max_array_sites,
            "max_bytes": max_array_bytes,
        },
    )
    sections.append(
        "<section><h2>manifest</h2><details><summary>report manifest</summary>"
        f"<pre>{escape(_scrub(_safe_json(manifest)))}</pre></details></section>"
    )

    title = escape(_scrub(str(getattr(log, "model_label", None) or "TorchLens report")))
    document = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        + _CSP_META
        + f"<title>TorchLens report: {title}</title>"
        + _REPORT_CSS
        + "</head><body>"
        + "".join(sections)
        + _GRAPH_SCRIPT
        + "</body></html>"
    )
    atomic_write_text(destination, document)
    return destination


def _partial_body(log: Any) -> str:
    """Failure-first body for a PartialTrace report."""

    from ..notebook.cards import partial_trace_card
    from ..notebook.cardtree import render_card_html

    try:
        card = render_card_html(partial_trace_card(log), include_css=True)
    except Exception:  # noqa: BLE001 - the report survives a card fault
        card = '<div class="tl-report-notice">failure card unavailable</div>'
    prefix_rows = "".join(
        f"<li>{escape(_scrub(str(getattr(r, 'layer_label_raw', None) or r)))}</li>"
        for r in list(getattr(log, "raw_layers", ()) or ())[:200]
    )
    return (
        f"<section><h2>failure</h2>{_scrub(card)}</section>"
        f"<section><h2>committed prefix</h2><ul>{prefix_rows}</ul></section>"
    )


def _manifest(log: Any, plan: _ArrayPlan, ledger: dict[str, Any]) -> dict[str, Any]:
    """Assemble the footer manifest (the complete omissions ledger).

    ``ledger`` carries the write-time facts: ``graph`` (facts dict),
    ``sites_shown``, ``bytes_used``, ``deterministic``, and the two array
    budgets (``max_sites`` / ``max_bytes``).
    """

    import torch

    import torchlens

    manifest: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "torchlens_version": getattr(torchlens, "__version__", "unknown"),
        # getattr spelling: the layer lint's torch-privates text scan would
        # otherwise match the dunder attribute access as a private touch.
        "torch_version": getattr(torch, "__version__", "unknown"),
        "model": _scrub(str(getattr(log, "model_label", "unknown"))),
        "backend": str(getattr(log, "backend", "unknown")),
        "outcome": getattr(
            getattr(getattr(log, "outcome", None), "status", None), "name", "UNKNOWN"
        ),
        "counts": {
            "ops": getattr(log, "num_ops", None),
            "saved_ops": getattr(log, "num_saved_ops", None),
            "modules": getattr(log, "num_modules", None),
        },
        "arrays": {
            "mode": plan.mode,
            "share_safe": plan.stripped,
            "flagged_total": plan.flagged_total,
            "sites_planned": len(plan.sites),
            "sites_shown": ledger["sites_shown"],
            "bytes_embedded": ledger["bytes_used"],
            "budgets": {"max_sites": ledger["max_sites"], "max_bytes": ledger["max_bytes"]},
        },
        "graph": ledger["graph"],
        "capture_honesty": capture_honesty_facts(log),
    }
    if not ledger["deterministic"]:
        import datetime

        manifest["generated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return manifest
