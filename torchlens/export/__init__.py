"""Static export helpers for TorchLens logs.

Promoted to a real subpackage (architecture memo rule 4 corollary, C01 items
17-18): tracker sinks live in ``_trackers``, foreign graph-viewer writers in
``_graphs``, shared helpers in ``_common``, and EVERY member -- builtin or
out-of-tree -- registers through the ONE export-target door
(:mod:`torchlens.export._registry`) with a per-member tier row.

Every exporter here carries the shared capture-honesty facts
(:mod:`torchlens._capture_honesty`) in the most format-appropriate slot --
comment preamble, metadata block, ``DataFrame.attrs``, or a dedicated key --
so an exported file never presents a possibly-unverified, poisoned, or
episode capture as clean data (WT1 A-V row 24).
"""

from __future__ import annotations

import json as _json
from html import escape
from pathlib import Path
from typing import Any

from .._capture_honesty import capture_honesty_facts, honesty_preamble_lines
from .._io._json import loads_bounded
from ..capture.structure_only import (
    require_structure_only_capability as _require_structure_only_capability,
)
from ..utils.display import atomic_write_text
from ._common import _iter_layers, _scalarize_cell, _static_graph_data
from ._graphs import NETRON_DISCLAIMER, netron
from ._registry import (
    export_target_info,
    export_targets,
    register_export_target,
    resolve_export_target,
    unregister_export_target,
)
from ._trackers import aim, mlflow, tensorboard, wandb

__tl_layer__ = "L7"

#: Bridge-tier Model Explorer names served lazily (PEP 562): this package is
#: L7 and ``._model_explorer`` is an L8 bridge, so the implementation loads on
#: first attribute access or first exporter call, never at package import.
_MODEL_EXPLORER_LAZY_NAMES = frozenset(
    {"to_model_explorer_dict", "validate_model_explorer_payload"}
)


def __getattr__(name: str) -> Any:
    """Lazily resolve re-exported Model Explorer names from the L8 bridge."""

    if name in _MODEL_EXPLORER_LAZY_NAMES:
        from . import _model_explorer as _me

        value = getattr(_me, name)
        globals()[name] = value
        return value
    if name == "ReportOptions":
        # The offline-report worker stays lazy like the html() door itself.
        from ._report import ReportOptions

        globals()[name] = ReportOptions
        return ReportOptions
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def model_explorer(log: Any, path: str | Path, **kwargs: Any) -> Path:
    """Export one log as a Model Explorer JSON graph collection.

    Thin deferred door: the L8 bridge implementation
    (:func:`torchlens.export._model_explorer.model_explorer`) loads on first
    call, keeping this L7 package free of eager upward imports.

    Parameters
    ----------
    log:
        TorchLens ``Trace`` to export.
    path:
        Destination JSON path.
    **kwargs:
        Options forwarded verbatim to the bridge implementation
        (``overlays=``, ``privacy_profile=``, ``per_step=``, ...).

    Returns
    -------
    Path
        Written JSON path.
    """

    from ._model_explorer import model_explorer as _impl

    return _impl(log, path, **kwargs)


def model_explorer_diff(members: Any, out_dir: str | Path, **kwargs: Any) -> Path:
    """Export a Model Explorer side-by-side diff collection for N members.

    Thin deferred door over
    :func:`torchlens.export._model_explorer.model_explorer_diff` (L8 bridge,
    loaded on first call).

    Parameters
    ----------
    members:
        Bundle or sequence of logs to compare.
    out_dir:
        Destination directory.
    **kwargs:
        Options forwarded verbatim (``label=``, ``privacy_profile=``).

    Returns
    -------
    Path
        Written collection path.
    """

    from ._model_explorer import model_explorer_diff as _impl

    return _impl(members, out_dir, **kwargs)


def model_explorer_serve(source: Any, **kwargs: Any) -> Path:
    """Serve one exported payload in a local Model Explorer instance.

    Thin deferred door over
    :func:`torchlens.export._model_explorer.model_explorer_serve` (L8 bridge,
    loaded on first call; requires the ``ai-edge-model-explorer`` extra).

    Parameters
    ----------
    source:
        Log or exported payload path to serve.
    **kwargs:
        Options forwarded verbatim (``path=`` plus viewer kwargs).

    Returns
    -------
    Path
        Path of the payload handed to the viewer.
    """

    from ._model_explorer import model_explorer_serve as _impl

    return _impl(source, **kwargs)


def _honesty_comment_block(log: Any, prefix: str) -> str:
    """Format the shared honesty preamble as comment lines.

    Parameters
    ----------
    log:
        Capture object being exported.
    prefix:
        Comment marker of the destination format (e.g. ``"# "``).

    Returns
    -------
    str
        Newline-terminated comment block.
    """

    return "".join(f"{prefix}{line}\n" for line in honesty_preamble_lines(log))


def _honesty_xml_comment(log: Any) -> str:
    """Format the shared honesty preamble as one XML/HTML comment.

    Returns
    -------
    str
        Single-line-per-fact comment; ``--`` is collapsed because XML
        comments must not contain double hyphens.
    """

    body = "\n".join(line.replace("--", "-") for line in honesty_preamble_lines(log))
    return f"<!-- {body} -->\n"


def _bundle_member_honesty(bundle: Any) -> dict[str, Any]:
    """Return per-member honesty facts for a Bundle export.

    Parameters
    ----------
    bundle:
        TorchLens ``Bundle``.

    Returns
    -------
    dict[str, Any]
        Member-name-keyed honesty fact blocks (best effort per member).
    """

    members: dict[str, Any] = {}
    for member_name in getattr(bundle, "names", ()) or ():
        # Per-member disclosure fallback: a member the bundle cannot serve (or
        # whose facts cannot be read) gets an explicit error row, never a
        # silent omission. The fact reader is getattr-defensive, so the
        # realistic raise surface is the member lookup itself.
        try:
            members[str(member_name)] = capture_honesty_facts(bundle[member_name])
        except (AttributeError, KeyError, TypeError):
            members[str(member_name)] = {"error": "member honesty facts unavailable"}
    return {"members": members}


def svg(log: Any, path: str | Path, *, editable: bool = True) -> Path:
    """Export a Trace graph as a lightweight SVG file.

    Parameters
    ----------
    log:
        TorchLens ``Trace`` to export.
    path:
        Destination SVG path.
    editable:
        Whether to include stable IDs and semantic CSS classes.

    Returns
    -------
    Path
        Written SVG path.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = _static_graph_data(log)
    rendered = _render_svg(data, editable=editable)
    # The honesty comment must sit AFTER the XML declaration (a comment before
    # it is invalid XML) and before the <svg> root.
    declaration_end = rendered.index("?>\n") + len("?>\n")
    rendered = rendered[:declaration_end] + _honesty_xml_comment(log) + rendered[declaration_end:]
    atomic_write_text(destination, rendered)
    return destination


def html(
    log: Any,
    path: str | Path,
    *,
    arrays: Any = "frontier",
    graph: str = "collapsed",
    options: Any = None,
) -> Path:
    """Export the single-file offline HTML trace report (F16, memo s6).

    Upgraded IN PLACE from the minimal canvas viewer: one file, no
    network, metadata for every op, the real module-collapsed graph with
    pan/zoom (the historical canvas survives as a labeled fallback), the
    summary lane's typed table, capture-honesty disclosures, and -- by
    default -- truncated thumbnails only for the diagnostic FRONTIER of a
    failure. Healthy captures embed zero array bytes by default.

    Parameters
    ----------
    log:
        TorchLens ``Trace`` or ``PartialTrace`` to export.
    path:
        Destination HTML path.
    arrays:
        ``"frontier"`` (default) / ``"flagged"`` (hard-capped debugging
        preset) / ``"none"`` / a ``tl.Selection`` (the general form).
    graph:
        ``"collapsed"`` (default) / ``"flat"`` / ``"none"``.
    options:
        ``tl.export.ReportOptions`` emission plumbing: ``share_safe``
        (strip every embedded value including the frontier),
        ``deterministic`` (suppress the manifest timestamp for byte-exact
        regeneration), ``vis_call_depth`` (graph-depth knob, forwarded
        and disclosed), and the thumbnail budgets.

    Returns
    -------
    Path
        Written HTML path.
    """

    from ._report import write_report

    return write_report(log, path, arrays=arrays, graph=graph, options=options)


#: The clock-basis disclosure every TorchLens host-clock export carries
#: (torchnative W0.1). These artifacts time WRAPPED CALLS on the host wall
#: clock under instrumentation; they are not device kernel time and share no
#: clock basis with a torch.profiler/Kineto trace, so values must never be
#: subtracted across the two.
CLOCK_BASIS_DISCLOSURE = (
    "host wall time around wrapped calls, under TorchLens instrumentation; "
    "not device kernel time -- shares no clock basis with torch.profiler/"
    "Kineto traces"
)


def chrome_trace(log: Any, path: str | Path) -> Path:
    """Export a Chrome tracing JSON timeline for one TorchLens log.

    Timestamps are ``CLOCK_BASIS_DISCLOSURE``: host wall time around wrapped
    calls under instrumentation, never device kernel time; the disclosure
    rides the artifact metadata.

    Parameters
    ----------
    log:
        TorchLens ``Trace`` to export.
    path:
        Destination JSON path.

    Returns
    -------
    Path
        Written JSON path.
    """
    # D15 (weightsfree memo): measurement-shaped output refuses on a
    # structure-only capture -- timings/allocator peaks are measurements a
    # value-free capture never made; rendering meta-dispatch cost as
    # 'measured' inverts real cost rankings. One chokepoint, typed code.
    _require_structure_only_capability(log, "measurement_exports")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "traceEvents": _chrome_trace_events(log),
        "displayTimeUnit": "ms",
        "metadata": {
            "schema": "torchlens.chrome_trace.v1",
            "clock_basis": CLOCK_BASIS_DISCLOSURE,
            "torchlens_capture_honesty": capture_honesty_facts(log),
        },
    }
    atomic_write_text(destination, _json.dumps(payload, indent=2))
    return destination


def chrome_trace_diff(bundle: Any, path: str | Path) -> Path:
    """Export a Chrome trace timeline comparing bundle members.

    Parameters
    ----------
    bundle:
        TorchLens ``Bundle`` with a ``supergraph`` accessor.
    path:
        Destination JSON path.

    Returns
    -------
    Path
        Written JSON path.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "traceEvents": _chrome_trace_diff_events(bundle),
        "displayTimeUnit": "ms",
        "metadata": {
            "schema": "torchlens.chrome_trace_diff.v1",
            "members": list(bundle.names),
            "torchlens_capture_honesty": _bundle_member_honesty(bundle),
        },
    }
    atomic_write_text(destination, _json.dumps(payload, indent=2))
    return destination


def speedscope(log: Any, path: str | Path) -> Path:
    """Export a speedscope evented profile for one TorchLens log.

    Timestamps are host wall time around wrapped calls under TorchLens
    instrumentation (``CLOCK_BASIS_DISCLOSURE``), never device kernel time.

    Parameters
    ----------
    log:
        TorchLens ``Trace`` to export.
    path:
        Destination JSON path.

    Returns
    -------
    Path
        Written JSON path.
    """
    # D15 (weightsfree memo): measurement-shaped output refuses on a
    # structure-only capture -- timings/allocator peaks are measurements a
    # value-free capture never made; rendering meta-dispatch cost as
    # 'measured' inverts real cost rankings. One chokepoint, typed code.
    _require_structure_only_capability(log, "measurement_exports")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frames = [{"name": _layer_display_name(layer)} for layer in _iter_layers(log)]
    events: list[dict[str, int | str]] = []
    cursor_us = 0
    for index, layer in enumerate(_iter_layers(log)):
        duration_us = _duration_us(layer)
        events.append({"type": "O", "frame": index, "at": cursor_us})
        cursor_us += duration_us
        events.append({"type": "C", "frame": index, "at": cursor_us})

    payload = {
        "$schema": "https://www.speedscope.app/file-format-schema.json",
        "clock_basis": CLOCK_BASIS_DISCLOSURE,
        "torchlens_capture_honesty": capture_honesty_facts(log),
        "shared": {"frames": frames},
        "profiles": [
            {
                "type": "evented",
                "name": getattr(log, "model_class_name", "TorchLens forward"),
                "unit": "microseconds",
                "startValue": 0,
                "endValue": cursor_us,
                "events": events,
            }
        ],
        "activeProfileIndex": 0,
    }
    atomic_write_text(destination, _json.dumps(payload, indent=2))
    return destination


def flamegraph(log: Any, path: str | Path) -> Path:
    """Export a folded-stack flamegraph text file for one TorchLens log.

    Frame weights are host wall time around wrapped calls under TorchLens
    instrumentation (``CLOCK_BASIS_DISCLOSURE``), never device kernel time;
    the folded format has no metadata slot, so the disclosure rides a
    zero-weight synthetic frame beside the honesty facts.

    Parameters
    ----------
    log:
        TorchLens ``Trace`` to export.
    path:
        Destination folded-stack text path.

    Returns
    -------
    Path
        Written text path.
    """
    # D15 (weightsfree memo): measurement-shaped output refuses on a
    # structure-only capture -- timings/allocator peaks are measurements a
    # value-free capture never made; rendering meta-dispatch cost as
    # 'measured' inverts real cost rankings. One chokepoint, typed code.
    _require_structure_only_capability(log, "measurement_exports")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    model_class_name = str(getattr(log, "model_class_name", "TorchLens"))
    # The folded-stack format has no comment/metadata slot, so the honesty
    # facts ride a ZERO-WEIGHT synthetic frame: standard flamegraph tools
    # accept the line and render nothing for weight 0, but the exported file
    # still carries the facts.
    facts = capture_honesty_facts(log)
    honesty_stack = ";".join(
        _sanitize_flamegraph_frame(f"{key}={facts[key]}".replace(" ", "_"))
        for key in ("capture_status", "capture_verified", "structure_only", "poisoned")
    )
    lines.append(f"torchlens_capture_honesty;{honesty_stack} 0")
    lines.append(
        f"torchlens_clock_basis;{_sanitize_flamegraph_frame(CLOCK_BASIS_DISCLOSURE.replace(' ', '_'))} 0"
    )
    for layer in _iter_layers(log):
        stack = [model_class_name]
        stack.extend(str(module) for module in (getattr(layer, "modules", None) or []))
        stack.append(_layer_display_name(layer))
        folded_stack = ";".join(_sanitize_flamegraph_frame(frame) for frame in stack)
        lines.append(f"{folded_stack} {_duration_us(layer)}")
    atomic_write_text(destination, "\n".join(lines) + ("\n" if lines else ""))
    return destination


def memory_timeline(log: Any, path: str | Path) -> Path:
    """Export a tensor-scope memory timeline for one TorchLens log.

    This reports bytes for tensors observed and retained by TorchLens. It is
    not an allocator trace and should not be interpreted as CUDA caching
    allocator, CPU allocator, or peak process memory telemetry.

    Parameters
    ----------
    log:
        TorchLens ``Trace`` to export.
    path:
        Destination JSON path.

    Returns
    -------
    Path
        Written JSON path.
    """
    # D15 (weightsfree memo): measurement-shaped output refuses on a
    # structure-only capture -- timings/allocator peaks are measurements a
    # value-free capture never made; rendering meta-dispatch cost as
    # 'measured' inverts real cost rankings. One chokepoint, typed code.
    _require_structure_only_capability(log, "measurement_exports")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    live_bytes = 0
    events: list[dict[str, Any]] = []
    for layer in _iter_layers(log):
        bytes_value = int(getattr(layer, "activation_memory", 0) or 0)
        live_bytes += bytes_value
        events.append(
            {
                "operation": getattr(layer, "step_index", None),
                "layer": getattr(layer, "layer_label", None),
                "tensor_bytes": bytes_value,
                "cumulative_tensor_bytes": live_bytes,
            }
        )
    payload = {
        "schema": "torchlens.memory_timeline.v1",
        "scope": "tensor",
        "disclaimer": "Tensor scope only; not an allocator trace.",
        # One-release compatibility exporter (observe item 11): the
        # cumulative_tensor_bytes meaning is UNCHANGED and its maximum is
        # structurally its last point -- it cannot answer "when is the peak".
        # The categorized v2 artifact is the replacement.
        "deprecation": (
            "torchlens.memory_timeline.v1 is superseded by the categorized "
            "memory_timeline_v2 export; v1 ships for one more release"
        ),
        "torchlens_capture_honesty": capture_honesty_facts(log),
        "events": events,
    }
    atomic_write_text(destination, _json.dumps(payload, indent=2))
    return destination


def memory_timeline_v2(log: Any, path: str | Path) -> Path:
    """Export the categorized module-contained memory timeline v2 artifact.

    The typed replacement for :func:`memory_timeline` (observe item 11):
    closed categories, named-absent facts, per-event module containment, the
    saved-for-backward band decomposed beside the gross figure, and the three
    products kept apart (per-event logical bytes / persistent baseline /
    cumulative produced bytes, never called live memory).

    Parameters
    ----------
    log:
        TorchLens ``Trace`` to export.
    path:
        Destination JSON path.

    Returns
    -------
    Path
        Written JSON path.
    """

    from torchlens.observe import memory_timeline_v2 as _build_v2

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    artifact = _build_v2(log)
    atomic_write_text(destination, _json.dumps(artifact, indent=2))
    return destination


def xarray(log: Any) -> Any:
    """Return a NeuroidAssembly-shaped xarray DataArray of saved outs.

    Parameters
    ----------
    log:
        TorchLens ``Trace`` whose saved tensor outs should be flattened
        into ``presentation`` by ``neuroid`` form.

    Returns
    -------
    Any
        ``xarray.DataArray`` with ``presentation`` and ``neuroid`` dimensions.

    Raises
    ------
    ImportError
        If xarray is not installed.
    ValueError
        If no saved tensor outs are available or presentation counts differ.
    """
    # D15: xarray's product IS saved tensor values -- it refuses through the
    # EXISTING value-payload capability (a marked empty array would be synthetic).
    _require_structure_only_capability(log, "value_payloads")

    try:
        import numpy as np
        import torch
        import xarray as xr
    except ImportError as exc:
        raise ImportError(
            "xarray export requires xarray. Install an environment with xarray available."
        ) from exc

    arrays = []
    layer_coord: list[str] = []
    layer_label_coord: list[str] = []
    index_coord: list[int] = []
    presentation_count: int | None = None
    for layer in _iter_layers(log):
        out = getattr(layer, "out", None)
        if not isinstance(out, torch.Tensor):
            continue
        values = out.detach().cpu().numpy()
        if values.ndim == 0:
            flat = values.reshape(1, 1)
        elif values.ndim == 1:
            flat = values.reshape(1, -1)
        else:
            flat = values.reshape(values.shape[0], -1)
        if presentation_count is None:
            presentation_count = int(flat.shape[0])
        elif flat.shape[0] != presentation_count:
            label = str(getattr(layer, "label", getattr(layer, "layer_label", "<unknown>")))
            raise ValueError(
                "All exported outs must share the same presentation count; "
                f"{label} has {flat.shape[0]}, expected {presentation_count}."
            )
        arrays.append(flat)
        layer_name = str(getattr(layer, "layer_label", ""))
        label = str(getattr(layer, "layer_label", layer_name))
        layer_coord.extend([layer_name] * flat.shape[1])
        layer_label_coord.extend([label] * flat.shape[1])
        index_coord.extend(range(flat.shape[1]))

    if not arrays or presentation_count is None:
        raise ValueError("No saved tensor outs are available for xarray export.")

    data = np.concatenate(arrays, axis=1)
    return xr.DataArray(
        data,
        dims=("presentation", "neuroid"),
        coords={
            "presentation": list(range(presentation_count)),
            "neuroid": list(range(data.shape[1])),
            "layer": ("neuroid", layer_coord),
            "layer_label": ("neuroid", layer_label_coord),
            "neuroid_index": ("neuroid", index_coord),
        },
        name="out",
        attrs={
            "assembly": "NeuroidAssembly",
            "source": "torchlens.export.xarray",
            "torchlens_capture_honesty": capture_honesty_facts(log),
        },
    )


def csv(log: Any, path: str | Path, **kwargs: Any) -> Path:
    """Write ``Trace.to_pandas()`` to CSV.

    Parameters
    ----------
    log:
        TorchLens ``Trace`` to export.
    path:
        Destination CSV path.
    **kwargs:
        Additional keyword arguments forwarded to ``DataFrame.to_csv``.

    Returns
    -------
    Path
        Written CSV path.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # CSV has no metadata slot: the honesty facts ride ``#`` comment lines
    # before the header. Read back with pd.read_csv(path, comment="#").
    table_text = log.to_pandas().to_csv(None, index=False, **kwargs)
    atomic_write_text(destination, _honesty_comment_block(log, "# ") + table_text)
    return destination


def parquet(log: Any, path: str | Path, **kwargs: Any) -> Path:
    """Write ``Trace.to_pandas()`` to Parquet.

    Parameters
    ----------
    log:
        TorchLens ``Trace`` to export.
    path:
        Destination Parquet path.
    **kwargs:
        Additional keyword arguments forwarded to ``DataFrame.to_parquet``.

    Returns
    -------
    Path
        Written Parquet path.

    Raises
    ------
    ImportError
        If pyarrow is unavailable.
    """

    try:
        import pyarrow
        import pyarrow.parquet as pyarrow_parquet
    except ImportError as exc:
        raise ImportError(
            "Parquet export requires pyarrow. Install with: pip install torchlens[tabular]"
        ) from exc
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    table = pyarrow.Table.from_pandas(_parquet_safe_dataframe(log.to_pandas()))
    # Honesty facts ride the parquet file-level schema metadata (readable via
    # pyarrow.parquet.read_schema(path).metadata); read_parquet is unaffected.
    metadata = dict(table.schema.metadata or {})
    metadata[b"torchlens_capture_honesty"] = _json.dumps(capture_honesty_facts(log)).encode()
    table = table.replace_schema_metadata(metadata)
    pyarrow_parquet.write_table(table, destination, **kwargs)
    return destination


def json(
    log: Any,
    path: str | Path,
    *,
    orient: str = "records",
    **kwargs: Any,
) -> Path:
    """Write ``Trace.to_pandas()`` to JSON.

    Parameters
    ----------
    log:
        TorchLens ``Trace`` to export.
    path:
        Destination JSON path.
    orient:
        JSON orientation passed to ``DataFrame.to_json``.
    **kwargs:
        Additional keyword arguments forwarded to ``DataFrame.to_json``.

    Returns
    -------
    Path
        Written JSON path.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows_json = _parquet_safe_dataframe(log.to_pandas()).to_json(None, orient=orient, **kwargs)
    payload = {
        "schema": "torchlens.table_export.v1",
        "capture_honesty": capture_honesty_facts(log),
        "orient": orient,
        "rows": loads_bounded(rows_json),
    }
    atomic_write_text(destination, _json.dumps(payload, indent=2))
    return destination


def _duration_us(layer: Any) -> int:
    """Return a positive microsecond duration for a layer.

    Parameters
    ----------
    layer:
        Layer entry.

    Returns
    -------
    int
        Duration in microseconds.
    """

    duration = float(getattr(layer, "func_duration", 0.0) or 0.0)
    return max(1, int(duration * 1_000_000))


def _layer_display_name(layer: Any) -> str:
    """Return a human-readable layer name.

    Parameters
    ----------
    layer:
        Layer entry.

    Returns
    -------
    str
        Display name.
    """

    label = str(getattr(layer, "layer_label", ""))
    func_name = str(getattr(layer, "func_name", "") or getattr(layer, "layer_type", ""))
    return f"{label} ({func_name})" if func_name and func_name != label else label


def _chrome_trace_events(log: Any) -> list[dict[str, Any]]:
    """Return Chrome trace events for a model log.

    Parameters
    ----------
    log:
        Model log to serialize.

    Returns
    -------
    list[dict[str, Any]]
        Chrome trace event records.
    """

    events: list[dict[str, Any]] = [
        {
            "name": "process_name",
            "ph": "M",
            "pid": 1,
            "tid": 0,
            "args": {"name": str(getattr(log, "model_class_name", "TorchLens forward"))},
        }
    ]
    cursor_us = 0
    for layer in _iter_layers(log):
        duration_us = _duration_us(layer)
        events.append(
            {
                "name": _layer_display_name(layer),
                "cat": "torchlens.forward",
                "ph": "X",
                "pid": 1,
                "tid": 0,
                "ts": cursor_us,
                "dur": duration_us,
                "args": {
                    "layer_label": getattr(layer, "layer_label", None),
                    "op_type": getattr(layer, "func_name", None),
                    "memory": getattr(layer, "activation_memory", None),
                    "module_path": getattr(layer, "module", None),
                },
            }
        )
        cursor_us += duration_us
    return events


def _chrome_trace_diff_events(bundle: Any) -> list[dict[str, Any]]:
    """Return Chrome trace events for a bundle comparison.

    Parameters
    ----------
    bundle:
        Bundle to serialize.

    Returns
    -------
    list[dict[str, Any]]
        Chrome trace event records.
    """

    events: list[dict[str, Any]] = []
    deltas = bundle.norm_delta()
    pid_by_member = {name: index + 1 for index, name in enumerate(bundle.names)}
    for member_name in bundle.names:
        events.append(
            {
                "name": "process_name",
                "ph": "M",
                "pid": pid_by_member[member_name],
                "tid": 0,
                "args": {"name": member_name},
            }
        )
    for node_index, graph_node_label in enumerate(bundle.supergraph.topological_order):
        node = bundle.supergraph.nodes[graph_node_label]
        for member_name in getattr(node, "traces", []):
            layer = node.layer_refs.get(member_name)
            events.append(
                {
                    "name": graph_node_label,
                    "cat": "torchlens.forward",
                    "ph": "X",
                    "pid": pid_by_member[member_name],
                    "tid": 0,
                    "ts": node_index * 1000,
                    "dur": 1000,
                    "args": {
                        "op_type": getattr(node, "op_type", ""),
                        "module_path": getattr(node, "module_path", None),
                        "module_type": getattr(node, "module_type", None),
                        "delta": deltas.get(graph_node_label, {}).get(member_name),
                        "memory": getattr(layer, "activation_memory", None),
                    },
                }
            )
    return events


def _parquet_safe_dataframe(dataframe: Any) -> Any:
    """Return a dataframe whose object columns are pyarrow-compatible.

    Parameters
    ----------
    dataframe:
        Pandas dataframe to sanitize before Parquet serialization.

    Returns
    -------
    Any
        Sanitized pandas dataframe.
    """

    sanitized = dataframe.copy()
    for column_name in sanitized.columns:
        if str(sanitized[column_name].dtype) == "object":
            sanitized[column_name] = sanitized[column_name].map(_parquet_cell)
    return sanitized


def _parquet_cell(value: Any) -> Any:
    """Return a pyarrow-compatible representation of a table cell.

    Parameters
    ----------
    value:
        Original dataframe cell.

    Returns
    -------
    Any
        Primitive value or string representation.
    """

    return _scalarize_cell(value)


def _sanitize_flamegraph_frame(frame: str) -> str:
    """Return a folded-stack-safe frame name.

    Parameters
    ----------
    frame:
        Raw frame name.

    Returns
    -------
    str
        Sanitized frame name.
    """

    return frame.replace(";", "_").replace("\n", " ").strip() or "<unknown>"


def _render_svg(data: dict[str, Any], *, editable: bool) -> str:
    """Render serialized graph data as SVG.

    Parameters
    ----------
    data:
        Static graph data.
    editable:
        Whether to include stable IDs and semantic classes.

    Returns
    -------
    str
        SVG document.
    """

    node_by_id = {node["id"]: node for node in data["nodes"]}
    edge_markup = []
    for edge in data["edges"]:
        source = node_by_id[edge["source"]]
        target = node_by_id[edge["target"]]
        edge_id = f"tl-edge-{_safe_id(edge['source'])}-{_safe_id(edge['target'])}"
        attrs = f' id="{edge_id}" class="tl-edge"' if editable else ""
        edge_markup.append(
            f'<line{attrs} x1="{source["x"] + 60}" y1="{source["y"]}" '
            f'x2="{target["x"] - 60}" y2="{target["y"]}" />'
        )
    node_markup = []
    for node in data["nodes"]:
        node_id = f"tl-node-{_safe_id(node['id'])}"
        attrs = f' id="{node_id}" class="tl-node tl-node-{node["type"]}"' if editable else ""
        title = escape(f"{node['label']} {node['shape']} {node['memory']}".strip())
        node_markup.append(
            f'<g{attrs} transform="translate({node["x"]},{node["y"]})">'
            f"<title>{title}</title><rect x='-65' y='-28' width='130' height='56' rx='6' />"
            f"<text text-anchor='middle' y='-4'>{escape(node['label'])}</text>"
            f"<text text-anchor='middle' y='16'>{escape(node['shape'] or node['memory'])}</text></g>"
        )
    return (
        "<?xml version='1.0' encoding='utf-8'?>\n"
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{data['width']}' height='{data['height']}' "
        f"viewBox='0 0 {data['width']} {data['height']}'>"
        "<style>.tl-edge{stroke:#555;stroke-width:1.4}.tl-node rect{fill:#fff;stroke:#222;stroke-width:1.2}"
        ".tl-node-input rect{fill:#D9F0D3}.tl-node-output rect{fill:#F6D7C3}"
        ".tl-node-parameterized rect{fill:#DDEAF7}.tl-node-buffer rect{fill:#F7E7BA}"
        ".tl-node text{font:12px sans-serif;fill:#111;pointer-events:none}</style>"
        + "".join(edge_markup)
        + "".join(node_markup)
        + "</svg>"
    )


def _render_html(payload: str) -> str:
    """Render a self-contained HTML graph viewer.

    Parameters
    ----------
    payload:
        JSON graph payload.

    Returns
    -------
    str
        HTML document.
    """

    return (
        "<!doctype html><html><head><meta charset='utf-8'><title>TorchLens graph</title>"
        "<style>html,body{margin:0;height:100%;overflow:hidden;font-family:system-ui,sans-serif}"
        "#tip{position:fixed;display:none;background:#111;color:white;padding:6px 8px;border-radius:4px;"
        "font-size:12px;pointer-events:none}.tl-edge{stroke:#666;stroke-width:1.4}.tl-node rect{fill:#fff;"
        "stroke:#222;stroke-width:1.2}.tl-node:hover rect{stroke:#0072B2;stroke-width:3}"
        ".tl-node-input rect{fill:#D9F0D3}.tl-node-output rect{fill:#F6D7C3}"
        ".tl-node-parameterized rect{fill:#DDEAF7}.tl-node-buffer rect{fill:#F7E7BA}"
        ".tl-node text{font:12px sans-serif;fill:#111;pointer-events:none}</style></head>"
        "<body><svg id='graph' width='100%' height='100%'><g id='viewport'></g></svg><div id='tip'></div>"
        f"<script>const graph={payload};"
        "const svg=document.getElementById('graph'),vp=document.getElementById('viewport'),tip=document.getElementById('tip');"
        "let scale=1,tx=20,ty=20,drag=false,last=[0,0];const byId=new Map(graph.nodes.map(n=>[n.id,n]));"
        "function el(n,a){const e=document.createElementNS('http://www.w3.org/2000/svg',n);for(const k in a)e.setAttribute(k,a[k]);return e}"
        "function draw(){graph.edges.forEach(ed=>{const s=byId.get(ed.source),t=byId.get(ed.target);"
        "vp.appendChild(el('line',{class:'tl-edge',x1:s.x+60,y1:s.y,x2:t.x-60,y2:t.y}));});"
        "graph.nodes.forEach(n=>{const g=el('g',{class:'tl-node tl-node-'+n.type,transform:`translate(${n.x},${n.y})`});"
        "g.appendChild(el('rect',{x:-65,y:-28,width:130,height:56,rx:6}));"
        "let a=el('text',{'text-anchor':'middle',y:-4});a.textContent=n.label;g.appendChild(a);"
        "let b=el('text',{'text-anchor':'middle',y:16});b.textContent=n.shape||n.memory;g.appendChild(b);"
        "g.onmousemove=e=>{tip.style.display='block';tip.style.left=e.clientX+12+'px';tip.style.top=e.clientY+12+'px';"
        "tip.textContent=[n.label,n.shape,n.memory].filter(Boolean).join('  ')};g.onmouseleave=()=>tip.style.display='none';vp.appendChild(g);});}"
        "function apply(){vp.setAttribute('transform',`translate(${tx},${ty}) scale(${scale})`)}"
        "svg.addEventListener('wheel',e=>{e.preventDefault();scale*=e.deltaY<0?1.1:.9;apply()},{passive:false});"
        "svg.addEventListener('mousedown',e=>{drag=true;last=[e.clientX,e.clientY]});"
        "window.addEventListener('mouseup',()=>drag=false);window.addEventListener('mousemove',e=>{if(!drag)return;"
        "tx+=e.clientX-last[0];ty+=e.clientY-last[1];last=[e.clientX,e.clientY];apply()});draw();apply();</script></body></html>"
    )


def _safe_id(value: str) -> str:
    """Return a CSS/SVG-safe identifier fragment.

    Parameters
    ----------
    value:
        Raw identifier.

    Returns
    -------
    str
        Sanitized identifier.
    """

    return "".join(char if char.isalnum() else "-" for char in value).strip("-")


# ---------------------------------------------------------------------------
# Builtin registrations: through the SAME public door out-of-tree exporters
# use (registry law 6.1 -- builtins never take a kernel side channel). The
# tier row is authoritative per member: present = native emitter, bridge =
# foreign-peer-shaped writer.
# ---------------------------------------------------------------------------
_BUILTIN_EXPORT_TARGETS: tuple[tuple[str, Any, str, dict[str, Any]], ...] = (
    ("svg", svg, "present", {"output": "file", "requires_extra": "none"}),
    ("html", html, "present", {"output": "file", "requires_extra": "none"}),
    ("chrome_trace", chrome_trace, "present", {"output": "file", "requires_extra": "none"}),
    (
        "chrome_trace_diff",
        chrome_trace_diff,
        "present",
        {"output": "file", "requires_extra": "none"},
    ),
    ("speedscope", speedscope, "present", {"output": "file", "requires_extra": "none"}),
    ("flamegraph", flamegraph, "present", {"output": "file", "requires_extra": "none"}),
    ("memory_timeline", memory_timeline, "present", {"output": "file", "requires_extra": "none"}),
    (
        "memory_timeline_v2",
        memory_timeline_v2,
        "present",
        {"output": "file", "requires_extra": "none"},
    ),
    ("csv", csv, "present", {"output": "file", "requires_extra": "pandas"}),
    ("parquet", parquet, "present", {"output": "file", "requires_extra": "pandas"}),
    ("json", json, "present", {"output": "file", "requires_extra": "pandas"}),
    ("xarray", xarray, "present", {"output": "object", "requires_extra": "xarray"}),
    ("netron", netron, "bridge", {"output": "file", "requires_extra": "none"}),
    ("model_explorer", model_explorer, "bridge", {"output": "file", "requires_extra": "none"}),
    (
        "model_explorer_diff",
        model_explorer_diff,
        "bridge",
        {"output": "file", "requires_extra": "none"},
    ),
    (
        "model_explorer_serve",
        model_explorer_serve,
        "bridge",
        {"output": "server", "requires_extra": "ai-edge-model-explorer"},
    ),
    ("tensorboard", tensorboard, "bridge", {"output": "tracker", "requires_extra": "none"}),
    ("wandb", wandb, "bridge", {"output": "tracker", "requires_extra": "wandb"}),
    ("mlflow", mlflow, "bridge", {"output": "tracker", "requires_extra": "none"}),
    ("aim", aim, "bridge", {"output": "tracker", "requires_extra": "none"}),
)

for _name, _fn, _tier, _caps in _BUILTIN_EXPORT_TARGETS:
    register_export_target(_name, _fn, tier=_tier, capabilities=_caps)
del _name, _fn, _tier, _caps

__all__ = [
    "NETRON_DISCLAIMER",
    "ReportOptions",
    "aim",
    "chrome_trace",
    "chrome_trace_diff",
    "csv",
    "export_target_info",
    "export_targets",
    "flamegraph",
    "html",
    "json",
    "memory_timeline",
    "memory_timeline_v2",
    "mlflow",
    "model_explorer",
    "model_explorer_diff",
    "model_explorer_serve",
    "netron",
    "parquet",
    "register_export_target",
    "resolve_export_target",
    "speedscope",
    "svg",
    "tensorboard",
    "to_model_explorer_dict",
    "unregister_export_target",
    "validate_model_explorer_payload",
    "wandb",
    "xarray",
]
