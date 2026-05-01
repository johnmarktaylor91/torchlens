"""Static export helpers for TorchLens logs."""

from __future__ import annotations

import json as _json
from html import escape
from pathlib import Path
from typing import Any


def svg(log: Any, path: str | Path, *, editable: bool = True) -> Path:
    """Export a ModelLog graph as a lightweight SVG file.

    Parameters
    ----------
    log:
        TorchLens ``ModelLog`` to export.
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
    destination.write_text(_render_svg(data, editable=editable), encoding="utf-8")
    return destination


def html(log: Any, path: str | Path) -> Path:
    """Export a minimal self-contained HTML graph viewer.

    The output supports pan, zoom, and node hover without importing TorchLens'
    viewer or notebook extras and without loading network resources.

    Parameters
    ----------
    log:
        TorchLens ``ModelLog`` to export.
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
    destination.write_text(_render_html(payload), encoding="utf-8")
    return destination


def chrome_trace(log: Any, path: str | Path) -> Path:
    """Export a Chrome tracing JSON timeline for one TorchLens log.

    Parameters
    ----------
    log:
        TorchLens ``ModelLog`` to export.
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
        "metadata": {"schema": "torchlens.chrome_trace.v1"},
    }
    destination.write_text(_json.dumps(payload, indent=2), encoding="utf-8")
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
        },
    }
    destination.write_text(_json.dumps(payload, indent=2), encoding="utf-8")
    return destination


def speedscope(log: Any, path: str | Path) -> Path:
    """Export a speedscope evented profile for one TorchLens log.

    Parameters
    ----------
    log:
        TorchLens ``ModelLog`` to export.
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
        "shared": {"frames": frames},
        "profiles": [
            {
                "type": "evented",
                "name": getattr(log, "model_name", "TorchLens forward"),
                "unit": "microseconds",
                "startValue": 0,
                "endValue": cursor_us,
                "events": events,
            }
        ],
        "activeProfileIndex": 0,
    }
    destination.write_text(_json.dumps(payload, indent=2), encoding="utf-8")
    return destination


def flamegraph(log: Any, path: str | Path) -> Path:
    """Export a folded-stack flamegraph text file for one TorchLens log.

    Parameters
    ----------
    log:
        TorchLens ``ModelLog`` to export.
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
    model_name = str(getattr(log, "model_name", "TorchLens"))
    for layer in _iter_layers(log):
        stack = [model_name]
        stack.extend(str(module) for module in (getattr(layer, "containing_modules", None) or []))
        stack.append(_layer_display_name(layer))
        folded_stack = ";".join(_sanitize_flamegraph_frame(frame) for frame in stack)
        lines.append(f"{folded_stack} {_duration_us(layer)}")
    destination.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return destination


def memory_timeline(log: Any, path: str | Path) -> Path:
    """Export a tensor-scope memory timeline for one TorchLens log.

    This reports bytes for tensors observed and retained by TorchLens. It is
    not an allocator trace and should not be interpreted as CUDA caching
    allocator, CPU allocator, or peak process memory telemetry.

    Parameters
    ----------
    log:
        TorchLens ``ModelLog`` to export.
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
        bytes_value = int(getattr(layer, "tensor_memory", 0) or 0)
        live_bytes += bytes_value
        events.append(
            {
                "operation": getattr(layer, "operation_num", None),
                "layer": getattr(layer, "layer_label", None),
                "tensor_bytes": bytes_value,
                "cumulative_tensor_bytes": live_bytes,
            }
        )
    payload = {
        "schema": "torchlens.memory_timeline.v1",
        "scope": "tensor",
        "disclaimer": "Tensor scope only; not an allocator trace.",
        "events": events,
    }
    destination.write_text(_json.dumps(payload, indent=2), encoding="utf-8")
    return destination


def xarray(log: Any) -> Any:
    """Return a NeuroidAssembly-shaped xarray DataArray of saved activations.

    Parameters
    ----------
    log:
        TorchLens ``ModelLog`` whose saved tensor activations should be flattened
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
        If no saved tensor activations are available or presentation counts differ.
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
        activation = getattr(layer, "activation", None)
        if not isinstance(activation, torch.Tensor):
            continue
        values = activation.detach().cpu().numpy()
        if values.ndim == 0:
            flat = values.reshape(1, 1)
        elif values.ndim == 1:
            flat = values.reshape(1, -1)
        else:
            flat = values.reshape(values.shape[0], -1)
        if presentation_count is None:
            presentation_count = int(flat.shape[0])
        elif flat.shape[0] != presentation_count:
            raise ValueError("All exported activations must share the same presentation count.")
        arrays.append(flat)
        layer_name = str(getattr(layer, "layer_label_no_pass", getattr(layer, "layer_label", "")))
        label = str(getattr(layer, "layer_label", layer_name))
        layer_coord.extend([layer_name] * flat.shape[1])
        layer_label_coord.extend([label] * flat.shape[1])
        index_coord.extend(range(flat.shape[1]))

    if not arrays or presentation_count is None:
        raise ValueError("No saved tensor activations are available for xarray export.")

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
        name="activation",
        attrs={"assembly": "NeuroidAssembly", "source": "torchlens.export.xarray"},
    )


def tensorboard(log: Any, writer: Any, step: int = 0, prefix: str = "torchlens") -> Any:
    """Write TorchLens scalar/text summaries to an existing TensorBoard writer.

    Parameters
    ----------
    log:
        TorchLens ``ModelLog`` to summarize.
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

    writer.add_scalar(f"{prefix}/num_layers", len(getattr(log, "layer_list", [])), step)
    writer.add_scalar(
        f"{prefix}/total_activation_memory",
        int(getattr(log, "total_activation_memory", 0) or 0),
        step,
    )
    writer.add_text(f"{prefix}/model_name", str(getattr(log, "model_name", "")), step)
    flush = getattr(writer, "flush", None)
    if callable(flush):
        flush()
    return writer


def wandb(log: Any, run: Any | None = None, name: str = "torchlens_model_log") -> dict[str, Any]:
    """Create and optionally log a Weights & Biases table for a TorchLens log.

    Parameters
    ----------
    log:
        TorchLens ``ModelLog`` to export.
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
    return {"table": table, "artifact": None}


def mlflow(log: Any, client: Any | None = None, prefix: str = "torchlens") -> dict[str, Any]:
    """Log simple TorchLens metrics to an existing MLflow-like client.

    Parameters
    ----------
    log:
        TorchLens ``ModelLog`` to summarize.
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
        for key, value in metrics.items():
            client.log_metric(f"{prefix}.{key}", value)
    return metrics


def aim(log: Any, run: Any | None = None, prefix: str = "torchlens") -> dict[str, Any]:
    """Track simple TorchLens metrics on an existing Aim-like run.

    Parameters
    ----------
    log:
        TorchLens ``ModelLog`` to summarize.
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
        for key, value in metrics.items():
            run.track(value, name=f"{prefix}.{key}")
    return metrics


def csv(log: Any, path: str | Path, **kwargs: Any) -> Path:
    """Write ``ModelLog.to_pandas()`` to CSV.

    Parameters
    ----------
    log:
        TorchLens ``ModelLog`` to export.
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
    log.to_pandas().to_csv(destination, index=False, **kwargs)
    return destination


def parquet(log: Any, path: str | Path, **kwargs: Any) -> Path:
    """Write ``ModelLog.to_pandas()`` to Parquet.

    Parameters
    ----------
    log:
        TorchLens ``ModelLog`` to export.
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
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Parquet export requires pyarrow. Install with: pip install torchlens[tabular]"
        ) from exc
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _parquet_safe_dataframe(log.to_pandas()).to_parquet(destination, **kwargs)
    return destination


def json(
    log: Any,
    path: str | Path,
    *,
    orient: str = "records",
    **kwargs: Any,
) -> Path:
    """Write ``ModelLog.to_pandas()`` to JSON.

    Parameters
    ----------
    log:
        TorchLens ``ModelLog`` to export.
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
    log.to_pandas().to_json(destination, orient=orient, **kwargs)
    return destination


def model_explorer(log: Any, path: str | Path) -> Path:
    """Export a JSON graph compatible with static graph explorer tools.

    Parameters
    ----------
    log:
        TorchLens ``ModelLog`` to export.
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
    payload = {
        "schema": "torchlens.model_explorer.v1",
        "graphs": [
            {
                "id": str(getattr(log, "name", None) or getattr(log, "model_name", "model")),
                "nodes": [
                    {
                        "id": node["id"],
                        "label": node["label"],
                        "namespace": node["type"],
                        "attrs": {"shape": node["shape"], "memory": node["memory"]},
                    }
                    for node in data["nodes"]
                ],
                "edges": data["edges"],
            }
        ],
    }
    destination.write_text(_json.dumps(payload, indent=2), encoding="utf-8")
    return destination


def netron(log: Any, path: str | Path) -> Path:
    """Export a lossy ONNX-shaped graph description for Netron inspection.

    The output is intentionally not a runnable ONNX model. It preserves node
    names, operation labels, simple tensor shapes, and edges so Netron-style
    graph inspection tools have something static to inspect without implying
    execution equivalence.

    Parameters
    ----------
    log:
        TorchLens ``ModelLog`` to export.
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
    payload = {
        "ir_version": "torchlens-lossy-onnx-shaped-v1",
        "producer_name": "torchlens",
        "runnable": False,
        "disclaimer": "Lossy ONNX-shaped inspection graph; not a real ONNX runtime model.",
        "graph": {
            "name": str(getattr(log, "model_name", "TorchLens graph")),
            "node": [
                {
                    "name": str(getattr(layer, "layer_label", "")),
                    "op_type": str(getattr(layer, "func_name", getattr(layer, "layer_type", ""))),
                    "input": list(getattr(layer, "parent_layers", []) or []),
                    "output": [str(getattr(layer, "layer_label", ""))],
                    "attribute": [
                        {
                            "name": "tensor_shape",
                            "value": list(getattr(layer, "tensor_shape", ()) or ()),
                        }
                    ],
                }
                for layer in entries
            ],
        },
    }
    destination.write_text(_json.dumps(payload, indent=2), encoding="utf-8")
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

    duration = float(getattr(layer, "func_time", 0.0) or 0.0)
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
            "args": {"name": str(getattr(log, "model_name", "TorchLens forward"))},
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
                    "tensor_memory": getattr(layer, "tensor_memory", None),
                    "module_path": getattr(layer, "containing_module", None),
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
    for node_index, node_name in enumerate(bundle.supergraph.topological_order):
        node = bundle.supergraph.nodes[node_name]
        for member_name in getattr(node, "traces", []):
            layer = node.layer_refs.get(member_name)
            events.append(
                {
                    "name": node_name,
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
                        "delta": deltas.get(node_name, {}).get(member_name),
                        "tensor_memory": getattr(layer, "tensor_memory", None),
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
        "num_tensors_saved": int(getattr(log, "num_tensors_saved", 0) or 0),
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
    return dataframe.apply(lambda column: column.map(_tracker_cell))


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
    """Serialize a ModelLog into static graph data.

    Parameters
    ----------
    log:
        TorchLens ``ModelLog`` to serialize.

    Returns
    -------
    dict[str, Any]
        Node and edge metadata for SVG/HTML exporters.
    """

    entries = _iter_layers(log)
    node_ids = {getattr(entry, "layer_label", "") for entry in entries}
    nodes: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        node_id = str(getattr(entry, "layer_label", f"node_{index}"))
        nodes.append(
            {
                "id": node_id,
                "label": node_id,
                "type": _node_type(entry),
                "shape": "x".join(str(dim) for dim in getattr(entry, "tensor_shape", ()) or ()),
                "memory": str(getattr(entry, "tensor_memory_str", "")),
                "x": 80 + (index % 8) * 180,
                "y": 80 + (index // 8) * 110,
            }
        )
    edges: list[dict[str, str]] = []
    for entry in entries:
        target = str(getattr(entry, "layer_label", ""))
        for parent in getattr(entry, "parent_layers", None) or []:
            if parent in node_ids:
                edges.append({"source": str(parent), "target": target})
    width = max((int(node["x"]) for node in nodes), default=0) + 160
    height = max((int(node["y"]) for node in nodes), default=0) + 100
    return {
        "title": getattr(log, "model_name", "TorchLens graph"),
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

    if getattr(entry, "is_input_layer", False):
        return "input"
    if getattr(entry, "is_output_layer", False):
        return "output"
    if getattr(entry, "is_buffer_layer", False):
        return "buffer"
    if getattr(entry, "is_terminal_bool_layer", False):
        return "bool"
    if int(getattr(entry, "num_params_total", 0) or 0) > 0:
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
