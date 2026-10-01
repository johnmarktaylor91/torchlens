"""Static export helpers for TorchLens logs.

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

from .._capture_honesty import (
    attach_dataframe_honesty,
    capture_honesty_facts,
    honesty_preamble_lines,
)
from .._io._json import loads_bounded
from ..utils.display import atomic_write_text


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


def html(log: Any, path: str | Path) -> Path:
    """Export a minimal self-contained HTML graph viewer.

    The output supports pan, zoom, and node hover without importing TorchLens'
    viewer or notebook extras and without loading network resources.

    Parameters
    ----------
    log:
        TorchLens ``Trace`` to export.
    path:
        Destination HTML path.

    Returns
    -------
    Path
        Written HTML path.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = _static_graph_data(log)
    payload = _json.dumps(data, separators=(",", ":"))
    atomic_write_text(destination, _honesty_xml_comment(log) + _render_html(payload))
    return destination


def chrome_trace(log: Any, path: str | Path) -> Path:
    """Export a Chrome tracing JSON timeline for one TorchLens log.

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

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "traceEvents": _chrome_trace_events(log),
        "displayTimeUnit": "ms",
        "metadata": {
            "schema": "torchlens.chrome_trace.v1",
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
        "torchlens_capture_honesty": capture_honesty_facts(log),
        "events": events,
    }
    atomic_write_text(destination, _json.dumps(payload, indent=2))
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


def tensorboard(log: Any, writer: Any, step: int = 0, prefix: str = "torchlens") -> Any:
    """Write TorchLens scalar/text summaries to an existing TensorBoard writer.

    Parameters
    ----------
    log:
        TorchLens ``Trace`` to summarize.
    writer:
        Existing writer object, for example ``SummaryWriter``.
    step:
        Global step for emitted summaries.
    prefix:
        Metric name prefix.

    Returns
    -------
    Any
        The writer object passed in.
    """

    _require_tracker_object(writer, method_name="tensorboard", required_method="add_scalar")
    writer.add_scalar(f"{prefix}/num_layers", len(getattr(log, "layer_list", [])), step)
    writer.add_scalar(
        f"{prefix}/total_activation_memory",
        int(getattr(log, "total_activation_memory", 0) or 0),
        step,
    )
    writer.add_text(f"{prefix}/model_class_name", str(getattr(log, "model_class_name", "")), step)
    add_text = getattr(writer, "add_text", None)
    if callable(add_text):
        add_text(f"{prefix}/capture_honesty", "; ".join(honesty_preamble_lines(log)), step)
    flush = getattr(writer, "flush", None)
    if callable(flush):
        flush()
    return writer


def wandb(log: Any, run: Any | None = None, name: str = "torchlens_trace") -> dict[str, Any]:
    """Create and optionally log a Weights & Biases table for a TorchLens log.

    Parameters
    ----------
    log:
        TorchLens ``Trace`` to export.
    run:
        Optional existing W&B run object. If omitted, ``wandb.run`` is used when
        present, but a new run is not created.
    name:
        Logged table key.

    Returns
    -------
    dict[str, Any]
        Mapping containing the created table and artifact placeholder.

    Raises
    ------
    ImportError
        If W&B is unavailable.
    """

    try:
        import wandb as wandb_module
    except ImportError as exc:
        raise ImportError(
            "wandb export requires the `wandb` extra: install torchlens[wandb]."
        ) from exc

    dataframe = _tracker_dataframe(log)
    table = wandb_module.Table(dataframe=dataframe)
    target_run = run if run is not None else getattr(wandb_module, "run", None)
    if target_run is not None:
        target_run.log({name: table})
    return {"table": table, "artifact": None, "capture_honesty": capture_honesty_facts(log)}


def mlflow(log: Any, client: Any | None = None, prefix: str = "torchlens") -> dict[str, Any]:
    """Log simple TorchLens metrics to an existing MLflow-like client.

    Parameters
    ----------
    log:
        TorchLens ``Trace`` to summarize.
    client:
        Optional object exposing ``log_metric``.
    prefix:
        Metric name prefix.

    Returns
    -------
    dict[str, Any]
        Metrics that were prepared for logging.
    """

    metrics = _summary_metrics(log)
    if client is not None:
        _require_tracker_object(client, method_name="mlflow", required_method="log_metric")
        for key, value in metrics.items():
            client.log_metric(f"{prefix}.{key}", value)
    # Honesty facts are returned (not logged): log_metric accepts numerics
    # only, and coercing verification facts to numbers would misstate them.
    return {**metrics, "capture_honesty": capture_honesty_facts(log)}


def aim(log: Any, run: Any | None = None, prefix: str = "torchlens") -> dict[str, Any]:
    """Track simple TorchLens metrics on an existing Aim-like run.

    Parameters
    ----------
    log:
        TorchLens ``Trace`` to summarize.
    run:
        Optional object exposing ``track``.
    prefix:
        Metric name prefix.

    Returns
    -------
    dict[str, Any]
        Metrics that were prepared for tracking.
    """

    metrics = _summary_metrics(log)
    if run is not None:
        _require_tracker_object(run, method_name="aim", required_method="track")
        for key, value in metrics.items():
            run.track(value, name=f"{prefix}.{key}")
    return {**metrics, "capture_honesty": capture_honesty_facts(log)}


def _require_tracker_object(target: Any, *, method_name: str, required_method: str) -> None:
    """Validate that a tracker export received a live tracker object."""

    if isinstance(target, str | Path):
        raise TypeError(
            f"torchlens.export.{method_name} expects an existing tracker object with "
            f"{required_method}(...), not a filesystem path."
        )
    if not callable(getattr(target, required_method, None)):
        raise TypeError(
            f"torchlens.export.{method_name} expects an object with "
            f"{required_method}(...); got {type(target).__name__}."
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


def model_explorer(log: Any, path: str | Path) -> Path:
    """Export a JSON graph using Google Model Explorer's graph schema.

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

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = _static_graph_data(log)
    incoming_edges: dict[str, list[dict[str, str]]] = {
        str(node["id"]): [] for node in data["nodes"]
    }
    for edge in data["edges"]:
        incoming_edges[str(edge["target"])].append({"sourceNodeId": str(edge["source"])})
    label = str(getattr(log, "trace_label", None) or getattr(log, "model_class_name", "model"))
    payload = {
        "schema": "torchlens.model_explorer.v2",
        "disclaimer": (
            "TorchLens graph-collection JSON for Google Model Explorer; a data export of the "
            "captured graph, not a runnable model. The top-level label/graphs shape matches "
            "Model Explorer's file-ingest contract (pinned against ai-edge-model-explorer "
            "0.1.32); acceptance by future external releases is not guaranteed."
        ),
        # Model Explorer's JSON ingest requires BOTH top-level keys label and
        # graphs to treat the file as a graph collection; without label the
        # app refuses with "Unsupported JSON format". Extra top-level keys
        # (schema, disclaimer, capture honesty) are tolerated by the pinned
        # ingest contract.
        "torchlens_capture_honesty": capture_honesty_facts(log),
        "label": label,
        "graphs": [
            {
                "id": str(
                    getattr(log, "trace_label", None) or getattr(log, "model_class_name", "model")
                ),
                "nodes": [
                    {
                        "id": node["id"],
                        "label": node["label"],
                        "namespace": node["type"],
                        "attrs": [
                            {"key": "shape", "value": node["shape"]},
                            {"key": "memory", "value": node["memory"]},
                        ],
                        "incomingEdges": incoming_edges[str(node["id"])],
                    }
                    for node in data["nodes"]
                ],
            }
        ],
    }
    atomic_write_text(destination, _json.dumps(payload, indent=2))
    return destination


#: Disclaimer embedded in the Netron export's model and graph doc strings.
NETRON_DISCLAIMER = (
    "TorchLens lossy graph export: not a runnable ONNX model; graph inspection "
    "only. Ops keep their captured TorchLens names under the ai.torchlens.lossy "
    "domain and carry no standard-ONNX execution semantics."
)


def netron(log: Any, path: str | Path) -> Path:
    """Export a lossy ONNX ``ModelProto`` JSON graph that Netron can open.

    The payload is valid ONNX protobuf JSON (camelCase field names, parseable
    into ``onnx.ModelProto``), which is the exact acceptance contract of
    Netron's ONNX JSON reader. It is intentionally NOT a runnable model: ops
    keep their captured TorchLens names under the custom
    ``ai.torchlens.lossy`` operator domain, only names, edges, and output
    shapes are preserved, and the disclaimer rides ``docString`` and
    ``metadataProps``.

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

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    entries = _iter_layers(log)
    repeated_labels = _repeated_layer_labels(entries)
    nodes = []
    for layer in entries:
        node_id = _export_node_id(layer, repeated_labels)
        shape = list(getattr(layer, "shape", ()) or ())
        node: dict[str, Any] = {
            "name": node_id,
            "opType": str(getattr(layer, "layer_type", None) or getattr(layer, "func_name", "")),
            "domain": "ai.torchlens.lossy",
            "input": [str(parent) for parent in (getattr(layer, "parents", []) or [])],
            "output": [node_id],
        }
        if shape and all(isinstance(dim, int) and not isinstance(dim, bool) for dim in shape):
            node["attribute"] = [{"name": "shape", "type": "INTS", "ints": shape}]
        nodes.append(node)
    payload = {
        "irVersion": 8,
        "producerName": "torchlens",
        "docString": NETRON_DISCLAIMER,
        "opsetImport": [{"domain": "ai.torchlens.lossy", "version": 1}],
        "metadataProps": [
            {"key": "torchlens.lossy_export", "value": "true"},
            {"key": "torchlens.runnable", "value": "false"},
            # Honesty facts as a JSON string value: metadataProps is the one
            # slot valid ONNX protobuf JSON offers for free-form metadata.
            {
                "key": "torchlens.capture_honesty",
                "value": _json.dumps(capture_honesty_facts(log)),
            },
        ],
        "graph": {
            "name": str(getattr(log, "model_class_name", "TorchLens graph")),
            "docString": NETRON_DISCLAIMER,
            "node": nodes,
        },
    }
    atomic_write_text(destination, _json.dumps(payload, indent=2))
    return destination


def _iter_layers(log: Any) -> list[Any]:
    """Return layer-pass entries in export order.

    Parameters
    ----------
    log:
        Model log-like object.

    Returns
    -------
    list[Any]
        Layer entries.
    """

    return list(
        getattr(log, "layer_list", None) or getattr(log, "layer_dict_main_keys", {}).values()
    )


def _repeated_layer_labels(entries: list[Any]) -> set[str]:
    """Return rolled layer labels that occur in multiple execution passes.

    Parameters
    ----------
    entries:
        Layer-pass entries in export order.

    Returns
    -------
    set[str]
        Labels requiring pass qualification for unique export node IDs.
    """

    counts: dict[str, int] = {}
    for entry in entries:
        label = str(getattr(entry, "layer_label", ""))
        counts[label] = counts.get(label, 0) + 1
    return {label for label, count in counts.items() if count > 1}


def _export_node_id(entry: Any, repeated_labels: set[str]) -> str:
    """Return a unique static-export node ID for one layer pass.

    Parameters
    ----------
    entry:
        Layer-pass entry.
    repeated_labels:
        Rolled labels requiring pass qualification.

    Returns
    -------
    str
        Pass-qualified ID for recurrent layers, otherwise the stable rolled label.
    """

    layer_label = str(getattr(entry, "layer_label", ""))
    if layer_label in repeated_labels:
        return str(getattr(entry, "label", layer_label))
    return layer_label


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


def _summary_metrics(log: Any) -> dict[str, int]:
    """Return common scalar metrics for tracker exports.

    Parameters
    ----------
    log:
        Model log to summarize.

    Returns
    -------
    dict[str, int]
        Scalar metrics.
    """

    return {
        "num_layers": len(getattr(log, "layer_list", [])),
        "num_saved_ops": int(getattr(log, "num_saved_ops", 0) or 0),
        "total_activation_memory": int(getattr(log, "total_activation_memory", 0) or 0),
    }


def _tracker_dataframe(log: Any) -> Any:
    """Return a tracker-safe dataframe with primitive cell values.

    Parameters
    ----------
    log:
        Model log to export.

    Returns
    -------
    Any
        Pandas dataframe suitable for strict tracker table types.
    """

    dataframe = log.to_pandas()
    # ``apply`` builds a new frame, which does not reliably propagate attrs.
    return attach_dataframe_honesty(dataframe.apply(lambda column: column.map(_tracker_cell)), log)


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


def _scalarize_cell(value: Any) -> Any:
    """Return a scalar-safe representation of a table cell.

    Shared body for :func:`_parquet_cell` and :func:`_tracker_cell`, which apply
    the identical primitive-or-repr coercion for two distinct call sites
    (pyarrow/Parquet column safety and strict tracker table types respectively).

    Parameters
    ----------
    value:
        Original dataframe cell.

    Returns
    -------
    Any
        Primitive value or string representation.
    """

    if value is None or isinstance(value, str | int | float | bool):
        return value
    try:
        import numpy as np
        import pandas as pd

        missing = pd.isna(value)
        if isinstance(missing, bool | np.bool_) and bool(missing):
            return None
    except Exception:
        pass
    return repr(value)


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


def _tracker_cell(value: Any) -> Any:
    """Return a scalar tracker-safe representation of a table cell.

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


def _static_graph_data(log: Any) -> dict[str, Any]:
    """Serialize a Trace into static graph data.

    Parameters
    ----------
    log:
        TorchLens ``Trace`` to serialize.

    Returns
    -------
    dict[str, Any]
        Node and edge metadata for SVG/HTML exporters.
    """

    entries = _iter_layers(log)
    repeated_labels = _repeated_layer_labels(entries)
    node_ids = {_export_node_id(entry, repeated_labels) for entry in entries}
    nodes: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        node_id = _export_node_id(entry, repeated_labels) or f"node_{index}"
        nodes.append(
            {
                "id": node_id,
                "label": str(getattr(entry, "layer_label", node_id)),
                "type": _node_type(entry),
                "shape": "x".join(str(dim) for dim in getattr(entry, "shape", ()) or ()),
                "memory": str(getattr(entry, "activation_memory", "")),
                "x": 80 + (index % 8) * 180,
                "y": 80 + (index // 8) * 110,
            }
        )
    edges: list[dict[str, str]] = []
    for entry in entries:
        target = _export_node_id(entry, repeated_labels)
        for parent in getattr(entry, "parents", None) or []:
            if parent in node_ids:
                edges.append({"source": str(parent), "target": target})
    width = max((int(node["x"]) for node in nodes), default=0) + 160
    height = max((int(node["y"]) for node in nodes), default=0) + 100
    return {
        "title": getattr(log, "model_class_name", "TorchLens graph"),
        "nodes": nodes,
        "edges": edges,
        "width": width,
        "height": height,
    }


def _node_type(entry: Any) -> str:
    """Return the semantic node type for an exported entry.

    Parameters
    ----------
    entry:
        Layer-pass log entry.

    Returns
    -------
    str
        Semantic node type.
    """

    if getattr(entry, "is_input", False):
        return "input"
    if getattr(entry, "is_output", False):
        return "output"
    if getattr(entry, "is_buffer", False):
        return "buffer"
    if getattr(entry, "is_terminal_bool", False):
        return "bool"
    if int(getattr(entry, "num_params", 0) or 0) > 0:
        return "parameterized"
    return "operation"


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


__all__ = [
    "aim",
    "chrome_trace",
    "chrome_trace_diff",
    "csv",
    "flamegraph",
    "html",
    "json",
    "memory_timeline",
    "mlflow",
    "model_explorer",
    "netron",
    "parquet",
    "speedscope",
    "svg",
    "tensorboard",
    "wandb",
    "xarray",
]
