"""Trackers: one engine, first-party sinks, measured relays (lane F26).

TorchLens never builds dashboards; it computes MORE and BETTER statistics
than the trackers' built-in watchers and FEEDS their dashboards. One
collection/reduction engine (the C06 ``torchlens.observability`` substrate)
produces immutable, tensor-free records; this package owns conversion,
namespace rendering, buffering, delivery, and sink failures:

- :func:`watch` -- attach-once watching with disclosed cost tiers (P:
  parameters/gradients/updates, the default, wraps no forward; M:
  module-hook activations; O: op capture, deferred this window with a typed
  teaching refusal), a mandatory caller ``global_step``, and the
  loud-failure taxonomy (nothing becomes an empty panel without a typed
  explanation).
- :class:`TensorBoardSink` -- the reference, CI-tested transport (raw
  histograms with exact counts on explicit edges; relay detection).
- :class:`WandbSink` -- the thin native serializer that preserves exactly
  what the wandb TB relay destroys (caller step axis, histogram summaries).
- :class:`JSONLSink` / :class:`MemorySink` -- dependency-free receivers
  (future-viewer plumbing / the reference test sink).
- :func:`build_graph_ir` / :func:`graph_ir_to_tensorboard` -- the executed
  DAG as a TB graph with evidence-qualified StepStats, truthful fields only.
- :class:`HFTrainerWatchCallback` / :class:`LightningWatchCallback` --
  explicit framework wrappers over the same engine.

Access spelling is ``import torchlens.trackers`` for now; root-facade
routing is an F35 registration fragment. Every spelling here is
DOCUMENTED-UNSTABLE pending naming-session ratification. The sink protocol
is documented (``docs/reference/trackers.md``) so third parties write their
own receivers -- TorchLens owns the engine, they own the sink.
"""

from __future__ import annotations

from ._amp import SCALE_EVIDENCE, correct_histogram, correct_spine, observed_grad_scale
from ._callbacks import HFTrainerWatchCallback, LightningWatchCallback, derived_cadence
from ._errors import (
    SinkDeliveryError,
    SinkProtocolError,
    TagGrammarError,
    TrackersError,
    WatchConfigError,
    WatchRuntimeError,
)
from ._graph import (
    TIMING_EVIDENCE,
    GraphIR,
    GraphNodeIR,
    build_graph_ir,
    graph_ir_to_tensorboard,
    node_name_for,
)
from ._protocol import (
    CAPABILITIES,
    DELIVERY_STATES,
    Sink,
    require_capabilities,
)
from ._records import (
    EMISSION_SCHEMA_VERSION,
    FAMILIES,
    TAG_SAFETY_CHECK_VERSION,
    HistogramPoint,
    ScalarPoint,
    StepEmission,
    TagGrammar,
    TextPoint,
    architecture_fingerprint,
    assert_tag_safe,
    build_manifest,
    emission_from_block,
    histogram_points,
    spine_scalars,
)
from ._sinks import JSONL_FORMAT, JSONLSink, MemorySink
from ._tensorboard import TensorBoardSink, detect_relays
from ._wandb import WANDB_BUCKET_CAP, WANDB_SAFE_DESCRIPTOR, WandbSink
from ._watch import (
    DEFAULT_EVERY,
    SIGNALS,
    WATCH_DISABLE_ENV,
    CloseReport,
    WatchSession,
    watch,
)

__tl_layer__ = "L8"

__all__ = [
    "CAPABILITIES",
    "DEFAULT_EVERY",
    "DELIVERY_STATES",
    "EMISSION_SCHEMA_VERSION",
    "FAMILIES",
    "JSONL_FORMAT",
    "SCALE_EVIDENCE",
    "SIGNALS",
    "TAG_SAFETY_CHECK_VERSION",
    "TIMING_EVIDENCE",
    "WANDB_BUCKET_CAP",
    "WANDB_SAFE_DESCRIPTOR",
    "WATCH_DISABLE_ENV",
    "CloseReport",
    "GraphIR",
    "GraphNodeIR",
    "HFTrainerWatchCallback",
    "HistogramPoint",
    "JSONLSink",
    "LightningWatchCallback",
    "MemorySink",
    "ScalarPoint",
    "Sink",
    "SinkDeliveryError",
    "SinkProtocolError",
    "StepEmission",
    "TagGrammar",
    "TagGrammarError",
    "TensorBoardSink",
    "TextPoint",
    "TrackersError",
    "WandbSink",
    "WatchConfigError",
    "WatchRuntimeError",
    "WatchSession",
    "architecture_fingerprint",
    "assert_tag_safe",
    "build_graph_ir",
    "build_manifest",
    "correct_histogram",
    "correct_spine",
    "derived_cadence",
    "detect_relays",
    "emission_from_block",
    "graph_ir_to_tensorboard",
    "histogram_points",
    "node_name_for",
    "observed_grad_scale",
    "require_capabilities",
    "spine_scalars",
    "watch",
]
