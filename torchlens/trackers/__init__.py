"""Experiment-tracker integration: one watch engine produces tensor-free statistics records, and sinks (TensorBoard, Weights & Biases, JSONL, memory, framework callbacks) deliver them.

The sink protocol is documented in ``docs/reference/trackers.md`` so third parties can write their own receivers."""

from __future__ import annotations

from .._observability._artifact import HistoryReader
from .._observability._render import HistoryView, SeriesPoint
from ._amp import (
    SCALE_EVIDENCE,
    correct_histogram,
    correct_running_stats,
    corrected_grad_block,
    observed_grad_scale,
    unscale_stage,
)
from ._callbacks import HFTrainerWatchCallback, LightningWatchCallback, derived_cadence
from ._comet import CometSink
from ._graph import (
    TIMING_EVIDENCE,
    GraphIR,
    GraphNodeIR,
    build_graph_ir,
    graph_ir_to_tensorboard,
    node_name_for,
)
from ._mlflow import MLflowSink
from ._protocol import (
    CAPABILITIES,
    DELIVERY_STATES,
    Sink,
    require_capabilities,
)
from ._records import (
    EMISSION_SCHEMA_VERSION,
    FAMILIES,
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
    running_stats_scalars,
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
    "HistoryView",
    "SeriesPoint",
    "CAPABILITIES",
    "DEFAULT_EVERY",
    "DELIVERY_STATES",
    "EMISSION_SCHEMA_VERSION",
    "FAMILIES",
    "JSONL_FORMAT",
    "SCALE_EVIDENCE",
    "SIGNALS",
    "TIMING_EVIDENCE",
    "WANDB_BUCKET_CAP",
    "WANDB_SAFE_DESCRIPTOR",
    "WATCH_DISABLE_ENV",
    "CloseReport",
    "CometSink",
    "GraphIR",
    "GraphNodeIR",
    "HFTrainerWatchCallback",
    "HistogramPoint",
    "HistoryReader",
    "JSONLSink",
    "LightningWatchCallback",
    "MLflowSink",
    "MemorySink",
    "ScalarPoint",
    "Sink",
    "StepEmission",
    "TagGrammar",
    "TensorBoardSink",
    "TextPoint",
    "WandbSink",
    "WatchSession",
    "architecture_fingerprint",
    "assert_tag_safe",
    "build_graph_ir",
    "build_manifest",
    "correct_histogram",
    "correct_running_stats",
    "corrected_grad_block",
    "derived_cadence",
    "detect_relays",
    "emission_from_block",
    "graph_ir_to_tensorboard",
    "histogram_points",
    "node_name_for",
    "observed_grad_scale",
    "require_capabilities",
    "running_stats_scalars",
    "unscale_stage",
    "watch",
]
