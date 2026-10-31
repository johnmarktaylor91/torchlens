"""MLflowSink: the native forward-watch receiver over an existing MLflow run.

A serializer over the same records as ``WandbSink``, never an engine. It
wraps a run the user already started (it never calls ``mlflow.start_run``:
constructing the run is the user's visible enablement act) and logs through
the ``MlflowClient`` API, so every write names its run explicitly:

- scalars: ``client.log_metric(run_id, tag, value, step=point.step)``. The
  caller's step rides ``step=``; MLflow's own counter never stands in for it.
  Metric timestamps are MLflow's (wall clock at log time) and carry no
  training meaning.
- text (manifests, disclosures): one text artifact per point at
  ``<tag>/step_<NNNNNNNNNN>.txt``.
- raw histograms: MLflow has no histogram type, so each point is stored
  exactly as one JSON artifact ``<tag>/step_<NNNNNNNNNN>.json`` holding
  ``counts``, ``edges`` and the ``summary`` fields. Counts are integers and
  edges are the engine's own floats, unmodified. That is one artifact file
  per point: keep ``hist_every`` sparse on large models.

Tag shape: MLflow metric keys allow letters, digits, ``_ - . : /`` and
space. A TorchLens tag outside that set (the tag grammar admits more
printable ASCII, for example brackets) refuses typed with the code
``tracker_tag_unsafe`` and latches the sink, instead of being rewritten.
"""

from __future__ import annotations

import re
from typing import Any

from ._errors import SinkProtocolError, TagGrammarError
from ._records import TAG_SAFETY_CHECK_VERSION, HistogramPoint, ScalarPoint, TextPoint

__tl_layer__ = "L8"

#: Characters MLflow accepts in metric keys and artifact paths.
_MLFLOW_KEY = re.compile(r"[A-Za-z0-9_./: -]+")

_REQUIRED_METHODS = ("log_metric", "log_text", "log_dict")


def _mlflow_module() -> Any:
    """Import mlflow lazily; refuse typed naming the package."""

    try:
        import mlflow
    except ImportError as exc:
        raise SinkProtocolError(
            f"MLflowSink needs the mlflow package: {exc}.",
            code="tracker_sink_unavailable",
            sink="MLflowSink",
            remedy="pip install mlflow",
        ) from exc
    return mlflow


def _resolve_target(client: Any, run_id: str | None) -> tuple[Any, str | None, str | None]:
    """Turn the constructor arguments into ``(client, run_id, problem)``.

    The fluent module (anything with ``active_run`` and ``MlflowClient``)
    resolves to a fresh ``MlflowClient`` on the active run. ``problem`` is a
    human sentence when the arguments cannot name a live run, else ``None``.
    """

    if callable(getattr(client, "active_run", None)) and callable(
        getattr(client, "MlflowClient", None)
    ):
        if run_id is None:
            active = client.active_run()
            if active is None:
                return client, None, "the fluent mlflow module has no active run"
            run_id = active.info.run_id
        client = client.MlflowClient()
    missing = [name for name in _REQUIRED_METHODS if not callable(getattr(client, name, None))]
    if missing:
        return client, run_id, f"{type(client).__name__} has no {', '.join(missing)}"
    if not run_id:
        return client, None, "an MlflowClient needs run_id="
    return client, run_id, None


def _mlflow_key(tag: str) -> str:
    """Return ``tag`` when MLflow can carry it as a key or path; refuse typed otherwise."""

    if _MLFLOW_KEY.fullmatch(tag):
        return tag
    raise TagGrammarError(
        f"Tag {tag!r} cannot be expressed as an MLflow metric key: MLflow allows only "
        "letters, digits and _ - . : / and space. Refusing beats rewriting the tag, "
        "which would break the pasteable-leaf property.",
        code="tracker_tag_unsafe",
        tag=tag,
        check_version=TAG_SAFETY_CHECK_VERSION,
        remedy=(
            "Rename the offending module/parameter, or route this run to "
            "JSONLSink/TensorBoardSink, which carry every tag the grammar admits."
        ),
    )


class MLflowSink:
    """Native MLflow receiver over an existing run (it never starts or ends one)."""

    def __init__(self, client: Any = None, *, run_id: str | None = None) -> None:
        """Wrap a run the user created.

        Parameters
        ----------
        client:
            The fluent ``mlflow`` module (its active run is used, or
            ``run_id=`` when given), an ``mlflow.MlflowClient`` (needs
            ``run_id=``), or ``None`` to import ``mlflow`` and use its active
            run. This sink never calls ``mlflow.start_run``.
        run_id:
            The run to log into.
        """

        if client is None:
            client = _mlflow_module()
        client, run_id, problem = _resolve_target(client, run_id)
        if problem is not None or run_id is None:
            raise SinkProtocolError(
                f"MLflowSink needs a live MLflow run: {problem}. It never calls "
                "mlflow.start_run itself -- constructing the run is the user's "
                "visible enablement act.",
                code="tracker_sink_invalid",
                sink="MLflowSink",
                remedy=(
                    "Start a run first (with mlflow.start_run(): ...) and pass the mlflow "
                    "module, or pass mlflow.MlflowClient() with run_id=<run id>."
                ),
            )
        self._client = client
        self._run_id = run_id
        self._closed = False

    @property
    def run_id(self) -> str:
        """The MLflow run this sink writes into."""

        return self._run_id

    def capabilities(self) -> frozenset[str]:
        """Scalars, raw histograms (JSON artifacts), text, and run metadata."""

        return frozenset({"scalar", "raw_histogram", "text_manifest", "run_metadata"})

    def emit_scalar(self, point: ScalarPoint) -> None:
        """Log one metric at the caller's step (never MLflow's own counter)."""

        self._client.log_metric(self._run_id, _mlflow_key(point.tag), point.value, step=point.step)

    def emit_histogram(self, point: HistogramPoint) -> None:
        """Store exact counts and edges as one JSON artifact at the caller's step."""

        payload = {
            "tag": point.tag,
            "step": point.step,
            "counts": list(point.counts),
            "edges": list(point.bin_edges),
            "summary": dict(point.summary),
        }
        path = f"{_mlflow_key(point.tag)}/step_{point.step:010d}.json"
        self._client.log_dict(self._run_id, payload, path)

    def emit_text(self, point: TextPoint) -> None:
        """Store one text payload as a text artifact at the caller's step."""

        path = f"{_mlflow_key(point.tag)}/step_{point.step:010d}.txt"
        self._client.log_text(self._run_id, point.text, path)

    def flush(self) -> None:
        """Every write above is synchronous; nothing to force here."""

    def close(self) -> None:
        """Never ends the user's run (they own its lifecycle); idempotent."""

        self._closed = True


__all__ = ["MLflowSink"]
