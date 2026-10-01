"""Tracker-sink export targets (bridge tier; subpackage promotion, C01 item 18).

Each function writes TorchLens summaries into an EXISTING foreign tracker
object (TensorBoard writer, W&B run, MLflow client, Aim run) -- bridge-tier
members of the export-target registry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .._capture_honesty import (
    attach_dataframe_honesty,
    capture_honesty_facts,
    honesty_preamble_lines,
)
from ._common import _scalarize_cell

__tl_layer__ = "L8"


def tensorboard(log: Any, writer: Any, *, step: int, prefix: str = "torchlens") -> Any:
    """Write TorchLens scalar/text summaries to an existing TensorBoard writer.

    Parameters
    ----------
    log:
        TorchLens ``Trace`` to summarize.
    writer:
        Existing writer object, for example ``SummaryWriter``.
    step:
        Global step for emitted summaries. REQUIRED: the historical
        ``step=0`` default was exactly the axis footgun the tracker design
        bans (F26; every emission path carries a caller step).
    prefix:
        Metric name prefix.

    Returns
    -------
    Any
        The writer object passed in.
    """

    _require_tracker_object(writer, method_name="tensorboard", required_method="add_scalar")
    # One converter for every tracker exporter (F26 unification): the four
    # historical exporters had drifted onto three different summary sets.
    for key, value in _summary_metrics(log).items():
        writer.add_scalar(f"{prefix}/{key}", value, step)
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
    metrics = _summary_metrics(log)
    target_run = run if run is not None else getattr(wandb_module, "run", None)
    if target_run is not None:
        payload: dict[str, Any] = {name: table}
        payload.update({f"{name}/{key}": value for key, value in metrics.items()})
        target_run.log(payload)
    return {
        "table": table,
        "artifact": None,
        **metrics,
        "capture_honesty": capture_honesty_facts(log),
    }


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
