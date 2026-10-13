"""CometSink: the native forward-watch receiver over an existing Comet experiment.

A serializer over the same records as ``WandbSink``, never an engine. It
wraps an experiment the user already created (``comet_ml.Experiment`` or
``comet_ml.OfflineExperiment``) and never constructs one, never ends one, and
never imports ``comet_ml`` itself: constructing the experiment is the user's
visible enablement act.

- scalars: ``experiment.log_metric(tag, value, step=point.step)``; the
  caller's step rides ``step=``.
- text (manifests, disclosures): ``experiment.log_text(text, step=point.step,
  metadata={"tag": tag})``.

Histograms are NOT advertised (no ``raw_histogram`` capability), so
``watch(..., to=CometSink(...), hist_every=...)`` refuses by name at attach.
The reason is measured: ``log_histogram_3d`` takes raw VALUES and bins them
itself on Comet's own fixed geometric grid (start 1e-12, stop 1e20, step 1.1),
so the exact counts on TorchLens's explicit signed-log2 edges cannot survive
it. Every histogram's summary statistics ride the scalar series instead
(``mean``, ``std``, ``norm``, ``min``, ``max``, ...), which Comet carries
exactly. Route raw histograms to ``TensorBoardSink``, ``WandbSink``,
``MLflowSink`` or ``JSONLSink``.

Comet adds its own bookkeeping (a ``curr_step`` parameter, a
``process duration`` metric) next to what this sink writes.
"""

from __future__ import annotations

from typing import Any

from ._errors import SinkProtocolError
from ._records import ScalarPoint, TextPoint

__tl_layer__ = "L8"

_REQUIRED_METHODS = ("log_metric", "log_text")


class CometSink:
    """Native Comet receiver over an existing experiment (it never starts or ends one)."""

    def __init__(self, experiment: Any) -> None:
        """Wrap an existing ``Experiment`` or ``OfflineExperiment`` object.

        Parameters
        ----------
        experiment:
            The object returned by ``comet_ml.Experiment(...)`` or
            ``comet_ml.OfflineExperiment(...)``. This sink never constructs
            one.
        """

        missing = [
            name for name in _REQUIRED_METHODS if not callable(getattr(experiment, name, None))
        ]
        if missing:
            raise SinkProtocolError(
                f"CometSink needs a live comet_ml experiment (got {type(experiment).__name__}, "
                f"which has no {', '.join(missing)}); it never constructs one -- creating the "
                "experiment is the user's visible enablement act.",
                code="tracker_sink_invalid",
                sink="CometSink",
                remedy=(
                    "Pass the object returned by comet_ml.Experiment(...) or "
                    "comet_ml.OfflineExperiment(...)."
                ),
            )
        self._experiment = experiment
        self._closed = False

    def capabilities(self) -> frozenset[str]:
        """Scalars, text, and run metadata; histograms are not advertised (see module doc)."""

        return frozenset({"scalar", "text_manifest", "run_metadata"})

    def emit_scalar(self, point: ScalarPoint) -> None:
        """Log one metric at the caller's step."""

        self._experiment.log_metric(point.tag, point.value, step=point.step)

    def emit_histogram(self, point: Any) -> None:
        """Unreachable through ``watch``: ``raw_histogram`` is not advertised."""

        raise SinkProtocolError(
            "CometSink does not carry raw histograms: log_histogram_3d re-bins raw values on "
            "Comet's own fixed grid, so exact counts on explicit edges cannot survive it.",
            code="tracker_sink_capability_missing",
            sink="CometSink",
            missing=("raw_histogram",),
            advertised=tuple(sorted(self.capabilities())),
            remedy="Route histograms to TensorBoardSink, WandbSink, MLflowSink or JSONLSink.",
        )

    def emit_text(self, point: TextPoint) -> None:
        """Log one text payload at the caller's step, tagged in its metadata."""

        self._experiment.log_text(point.text, step=point.step, metadata={"tag": point.tag})

    def flush(self) -> None:
        """Comet buffers internally (and writes at ``experiment.end()``); nothing to force."""

    def close(self) -> None:
        """Never ends the user's experiment (they own its lifecycle); idempotent."""

        self._closed = True


__all__ = ["CometSink"]
