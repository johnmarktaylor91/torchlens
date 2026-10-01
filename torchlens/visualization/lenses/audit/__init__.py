"""The lens validation harness (themes memo build item 17).

Four pieces, all deterministic and model-free unless a member says
otherwise:

- ``corpus``: the evidence corpus -- toy builders every check can run on,
  guarded real-model builders, and the per-artifact run manifest.
- ``stage0``: the Stage-0 deterministic audit -- geometry, size caps,
  colour spread, coverage, disclosure. A Stage-0 failure is a build bug and
  never reaches an evaluator.
- ``answer_key``: RenderIR-derived answer keys for the naive battery
  (hand-maintained keys are prohibited).
- ``battery``: the naive-evaluator packet builder, scorer, and the
  anchor-midpoint threshold-freezing arithmetic. The harness produces
  packets and scores responses; RUNNING evaluators (fresh Fable instances)
  is D03's delegated job.

Every spelling here is DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

from .answer_key import AnswerKey, generate_answer_key
from .battery import (
    HONESTY_CLASSES,
    BatteryPacket,
    BatteryScore,
    build_packet,
    freeze_threshold,
    score_responses,
)
from .corpus import CORPUS, CorpusMember, RunMeasurements, build_run_manifest
from .stage0 import (
    AuditFinding,
    AuditReport,
    Stage0Checks,
    palette_distinguishability,
    run_stage0,
    simulate_cvd,
)

__all__ = [
    "CORPUS",
    "HONESTY_CLASSES",
    "AnswerKey",
    "AuditFinding",
    "AuditReport",
    "BatteryPacket",
    "BatteryScore",
    "CorpusMember",
    "RunMeasurements",
    "Stage0Checks",
    "build_packet",
    "build_run_manifest",
    "freeze_threshold",
    "generate_answer_key",
    "palette_distinguishability",
    "run_stage0",
    "score_responses",
    "simulate_cvd",
]
