"""Tracker-sink export targets (bridge tier; subpackage promotion, C01 item 18).

Each function writes TorchLens summaries into an EXISTING foreign tracker
object (TensorBoard writer, W&B run, MLflow client, Aim run) -- bridge-tier
members of the export-target registry.
"""

from __future__ import annotations

import functools
import inspect
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .._capture_honesty import (
    attach_dataframe_honesty,
    capture_honesty_facts,
    honesty_preamble_lines,
)
from .._errors import ArgumentTypeError, KeywordConflictError
from ._common import _require_trace, _scalarize_cell

__tl_layer__ = "L8"


def tensorboard(trace: Any, writer: Any, *, step: int, prefix: str = "torchlens") -> Any:
    """Write TorchLens scalar/text summaries to an existing TensorBoard writer.

    Parameters
    ----------
    trace:
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

    # a non-Trace input used to write plausible zeros (getattr defaults).
    _require_trace(trace, exporter="tl.export.tensorboard")
    _require_tracker_object(writer, method_name="tensorboard", required_method="add_scalar")
    # One converter for every tracker exporter (F26 unification): the four
    # historical exporters had drifted onto three different summary sets.
    for key, value in _summary_metrics(trace).items():
        writer.add_scalar(f"{prefix}/{key}", value, step)
    writer.add_text(
        f"{prefix}/model_class_name", str(getattr(trace, "_source_class_name", "")), step
    )
    add_text = getattr(writer, "add_text", None)
    if callable(add_text):
        add_text(f"{prefix}/capture_honesty", "; ".join(honesty_preamble_lines(trace)), step)
    flush = getattr(writer, "flush", None)
    if callable(flush):
        flush()
    return writer


def wandb(trace: Any, to: Any | None = None, name: str = "torchlens_trace") -> dict[str, Any]:
    """Create and optionally log a Weights & Biases table for a TorchLens log.

    Parameters
    ----------
    trace:
        TorchLens ``Trace`` to export.
    to:
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

    dataframe = _tracker_dataframe(trace)
    table = wandb_module.Table(dataframe=dataframe)
    metrics = _summary_metrics(trace)
    target_run = to if to is not None else getattr(wandb_module, "run", None)
    if target_run is not None:
        payload: dict[str, Any] = {name: table}
        payload.update({f"{name}/{key}": value for key, value in metrics.items()})
        target_run.log(payload)
    return {
        "table": table,
        "artifact": None,
        **metrics,
        "capture_honesty": capture_honesty_facts(trace),
    }


def mlflow(
    trace: Any,
    client: Any | None = None,
    prefix: str = "torchlens",
    *,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Log TorchLens summary metrics, a per-layer table, the report and honesty tags to MLflow.

    Two client shapes are accepted, told apart by ``log_metric``'s signature
    (this package never imports mlflow): the fluent ``mlflow`` module
    (``log_metric(key, value, ...)``, logging to the active run, or to
    ``run_id`` when given) and an ``mlflow.MlflowClient``
    (``log_metric(run_id, key, value, ...)``), which needs ``run_id``.

    With a client, four things are logged to the run (parity with the W&B
    export):

    - the summary metrics ``<prefix>.num_layers``, ``<prefix>.num_saved_ops``
      and ``<prefix>.total_activation_memory``;
    - the per-layer statistics table as the artifact ``<prefix>/layers.json``
      (``log_table``);
    - the trace summary report plus the capture-honesty preamble and the
      full fact block as the text artifact ``<prefix>/report.txt``
      (``log_text``);
    - every capture-honesty fact as a STRING run tag
      ``<prefix>.capture_honesty.<fact>`` (``set_tag``): strings stay strings,
      other JSON values are written as compact sorted JSON
      (``true``/``false``/``null``/numbers/objects), never coerced to a metric.
      A fact whose JSON exceeds MLflow's tag-value limit is replaced by an
      ``<omitted: ...>`` marker naming the report artifact, which carries it
      in full.

    Every method the client must offer is checked before anything is logged,
    so a refusal never leaves a half-written run.

    Parameters
    ----------
    trace:
        TorchLens ``Trace`` to export.
    client:
        Optional ``mlflow`` module or ``mlflow.MlflowClient`` (or an object
        exposing ``log_metric``, ``log_table``, ``log_text`` and ``set_tag``).
    prefix:
        Metric, tag and artifact-folder name prefix.
    run_id:
        MLflow run to log into. Required with an ``MlflowClient``; with the
        fluent module it is forwarded as ``run_id=``.

    Returns
    -------
    dict[str, Any]
        The three summary metrics, ``table`` (the per-layer dataframe),
        ``report`` (the report text), ``tags`` (the flat string tags),
        ``artifacts`` (artifact paths logged; empty without a client) and
        ``capture_honesty`` (the full fact block).

    Raises
    ------
    TypeError
        If ``client`` lacks a required method, an ``MlflowClient`` comes
        without ``run_id``, or ``run_id`` is given to a client that cannot
        take it.
    """

    metrics = _summary_metrics(trace)
    if client is None and run_id is not None:
        raise KeywordConflictError(
            "torchlens.export.mlflow got run_id= without a client to log through",
            code="tracker_mlflow_run_id_without_client",
            remedy="pass client=mlflow.MlflowClient() with run_id=, or drop run_id=",
        )
    writers = _mlflow_writers(client, run_id) if client is not None else None
    parity = _tracker_parity(trace, prefix)
    artifacts: list[str] = []
    if writers is not None:
        for key, value in metrics.items():
            writers["log_metric"](f"{prefix}.{key}", value)
        artifacts = [f"{prefix}/layers.json", f"{prefix}/report.txt"]
        writers["log_table"](data=parity["table"], artifact_file=artifacts[0])
        writers["log_text"](parity["report"], artifacts[1])
        for key, value in parity["tags"].items():
            writers["set_tag"](key, value)
    return {**metrics, **parity, "artifacts": artifacts}


def _mlflow_writers(client: Any, run_id: str | None) -> dict[str, Callable[..., Any]]:
    """Validate the client and bind each logging method to ``run_id``."""

    _require_tracker_object(client, method_name="mlflow", required_method="log_metric")
    writers: dict[str, Callable[..., Any]] = {
        "log_metric": _mlflow_log_metric(client.log_metric, run_id)
    }
    for name in ("log_table", "log_text", "set_tag"):
        _require_tracker_object(client, method_name="mlflow", required_method=name)
    client_shape = _mlflow_is_client_shape(client.log_metric)
    for name in ("log_table", "log_text"):
        writers[name] = _mlflow_bind(client, name, run_id, client_shape=client_shape)
    writers["set_tag"] = _mlflow_bind_tag(client, run_id, client_shape=client_shape)
    return writers


def _mlflow_param_names(method: Any) -> list[str]:
    """Return a method's parameter names (empty when it has no signature)."""

    try:
        return list(inspect.signature(method).parameters)
    except (TypeError, ValueError):
        return []


def _mlflow_is_client_shape(log_metric: Any) -> bool:
    """True for ``MlflowClient``-style methods, whose first parameter is ``run_id``."""

    names = _mlflow_param_names(log_metric)
    return bool(names) and names[0] == "run_id"


def _mlflow_bind(client: Any, name: str, run_id: str | None, *, client_shape: bool) -> Any:
    """Bind ``run_id`` into ``client.<name>`` for the client shape in use."""

    method = getattr(client, name)
    if client_shape:
        return functools.partial(method, run_id)
    if run_id is None:
        return method
    if "run_id" in _mlflow_param_names(method):
        return functools.partial(method, run_id=run_id)
    raise KeywordConflictError(
        f"torchlens.export.mlflow got run_id= but this client's {name} takes no run_id",
        code="tracker_mlflow_run_id_unsupported",
        remedy="drop run_id=, or pass an mlflow.MlflowClient",
    )


def _mlflow_bind_tag(client: Any, run_id: str | None, *, client_shape: bool) -> Any:
    """Bind ``set_tag``; the fluent module's has no run id, so route through a client."""

    if client_shape or run_id is None:
        return _mlflow_bind(client, "set_tag", run_id, client_shape=client_shape)
    if "run_id" in _mlflow_param_names(client.set_tag):
        return functools.partial(client.set_tag, run_id=run_id)
    client_class = getattr(client, "MlflowClient", None)
    if not callable(client_class):
        raise KeywordConflictError(
            "torchlens.export.mlflow got run_id= but this client's set_tag takes no run_id",
            code="tracker_mlflow_run_id_unsupported",
            remedy="drop run_id=, or pass an mlflow.MlflowClient",
        )
    return functools.partial(client_class().set_tag, run_id)


def _mlflow_log_metric(log_metric: Any, run_id: str | None) -> Any:
    """Bind ``run_id`` for the client shape ``log_metric`` belongs to."""

    params = _mlflow_param_names(log_metric)
    if params and params[0] == "run_id":
        if run_id is None:
            raise ArgumentTypeError(
                "torchlens.export.mlflow got an MlflowClient-style client "
                "(log_metric(run_id, key, value)) without run_id=",
                code="tracker_mlflow_run_id_missing",
                remedy=(
                    "pass run_id=run.info.run_id, or pass the fluent `mlflow` module to "
                    "log into the active run"
                ),
            )
        return functools.partial(log_metric, run_id)
    if run_id is None:
        return log_metric
    if "run_id" in params:
        return functools.partial(log_metric, run_id=run_id)
    raise KeywordConflictError(
        "torchlens.export.mlflow got run_id= but this client's log_metric takes no run_id",
        code="tracker_mlflow_run_id_unsupported",
        remedy="drop run_id=, or pass an mlflow.MlflowClient",
    )


def aim(trace: Any, to: Any | None = None, prefix: str = "torchlens") -> dict[str, Any]:
    """Track TorchLens summary metrics, a per-layer table, the report and honesty tags on an Aim run.

    With a run, the export writes (Aim has no table or artifact type for
    this, so each piece uses the form Aim queries best):

    - the summary metrics as ``run.track(value, name="<prefix>.<metric>")``;
    - the per-layer statistics table as run params
      ``run["<prefix>.layers"] = [row, ...]`` (lossless, one dict per layer,
      keyed by column) PLUS one chartable series per numeric column,
      ``run.track(value, name="<prefix>.layers.<column>", step=<row index>,
      context={"table": "layers"})``;
    - the report (trace summary, honesty preamble, full fact block) as the
      run param ``run["<prefix>.report"]`` (a string);
    - the capture-honesty facts as the run param
      ``run["<prefix>.capture_honesty"]`` (the full dict) and as run tags
      ``run.add_tag("<prefix>.capture_honesty.<fact>=<value>")`` (strings;
      non-string facts as compact sorted JSON, never coerced to a metric).

    Every method the run must offer is checked before anything is written.

    Parameters
    ----------
    trace:
        TorchLens ``Trace`` to export.
    to:
        Optional object exposing ``track``, ``add_tag`` and item assignment.
    prefix:
        Metric, param and tag name prefix.

    Returns
    -------
    dict[str, Any]
        The three summary metrics, ``table`` (the per-layer dataframe),
        ``report`` (the report text), ``tags`` (the flat string tags) and
        ``capture_honesty`` (the full fact block).

    Raises
    ------
    TypeError
        If ``to`` lacks ``track``, ``add_tag`` or item assignment.
    """

    metrics = _summary_metrics(trace)
    if to is not None:
        for required in ("track", "add_tag", "__setitem__"):
            _require_tracker_object(to, method_name="aim", required_method=required)
    parity = _tracker_parity(trace, prefix)
    if to is not None:
        for key, value in metrics.items():
            to.track(value, name=f"{prefix}.{key}")
        _aim_write_table(to, parity["table"], prefix)
        to[f"{prefix}.report"] = parity["report"]
        to[f"{prefix}.capture_honesty"] = parity["capture_honesty"]
        for key, value in parity["tags"].items():
            to.add_tag(f"{key}={value}")
    return {**metrics, **parity}


def _aim_write_table(run: Any, table: Any, prefix: str) -> None:
    """Write the per-layer table as run params plus one series per numeric column."""

    rows = [
        {column: _aim_cell(value) for column, value in row.items()}
        for row in table.to_dict(orient="records")
    ]
    run[f"{prefix}.layers"] = rows
    for column in table.columns:
        name = f"{prefix}.layers.{column}"
        for index, row in enumerate(rows):
            value = row[column]
            if isinstance(value, int | float) and not isinstance(value, bool):
                run.track(value, name=name, step=index, context={"table": "layers"})


def _aim_cell(value: Any) -> Any:
    """Map a non-finite float to ``None`` (Aim params are JSON-like); keep the rest."""

    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


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
        "total_activation_memory": int(getattr(log, "_cost_total_activation_memory", 0) or 0),
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


#: MLflow's tag-value ceiling (``MAX_TAG_VAL_LENGTH``): longer facts are
#: replaced by a marker pointing at the report artifact that carries them.
_TAG_VALUE_MAX = 5000


def _tracker_parity(log: Any, prefix: str) -> dict[str, Any]:
    """Build the table, report and honesty tags every tracker export shares."""

    facts = capture_honesty_facts(log)
    report = _report_text(log, facts)
    return {
        "table": _tracker_dataframe(log),
        "report": report,
        "tags": _honesty_tags(facts, prefix, report_name=f"{prefix}/report.txt"),
        "capture_honesty": facts,
    }


def _fact_string(value: Any) -> str:
    """Render one honesty fact as a string: strings verbatim, the rest as sorted JSON."""

    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, default=str)


def _honesty_tags(facts: dict[str, Any], prefix: str, *, report_name: str) -> dict[str, str]:
    """Flatten the fact block to ``<prefix>.capture_honesty.<fact>`` string tags."""

    tags: dict[str, str] = {}
    for fact, value in facts.items():
        text = _fact_string(value)
        if len(text) > _TAG_VALUE_MAX:
            text = f"<omitted: {len(text)} characters; the full value is in {report_name}>"
        tags[f"{prefix}.capture_honesty.{fact}"] = text
    return tags


def _report_text(log: Any, facts: dict[str, Any]) -> str:
    """Compose the report artifact: honesty preamble, full facts, then the trace summary."""

    parts = [
        "TorchLens capture honesty",
        *honesty_preamble_lines(log),
        "",
        "Capture honesty facts (JSON)",
        json.dumps(facts, indent=2, sort_keys=True, default=str),
    ]
    summary = getattr(log, "summary", None)
    if callable(summary):
        parts.extend(["", "Trace summary", str(summary())])
    return "\n".join(parts) + "\n"
