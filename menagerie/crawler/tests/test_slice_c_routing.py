"""Slice C framework routing and platform-evidence tests."""

from __future__ import annotations

import pytest

from menagerie.crawler.constants import EnvironmentPhase, PlatformRequirement
from menagerie.crawler.effort import EffortCapExceeded, EffortTracker, StageCap
from menagerie.crawler.routing import (
    Arm64HeavyBuildAttempts,
    MissingPlatformEvidenceError,
    ModelRequirements,
    PlatformEvidence,
    defer_for_platform,
    phase_routes,
    route_model,
)


def test_torch_model_routes_by_declared_package_need() -> None:
    """A PyTorch graph dependency selects the graph intent in phase one."""

    route = route_model(
        ModelRequirements("m_graph", "pytorch", frozenset({"torch", "torch-geometric"}))
    )
    assert route.intent == "graph"
    assert route.phase is EnvironmentPhase.PYTORCH


@pytest.mark.parametrize(
    ("framework", "intent"),
    (("tensorflow", "tf-keras-arm64"), ("jax", "jax-flax-arm64"), ("paddle", "paddle-arm64")),
)
def test_native_frameworks_route_to_phase_two(framework: str, intent: str) -> None:
    """Native TensorFlow, JAX, and Paddle models remain in the ordered tail."""

    route = route_model(ModelRequirements(f"m_{framework}", framework))
    assert route.intent == intent
    assert route.phase is EnvironmentPhase.NATIVE_TAIL
    ordered = phase_routes((route, route_model(ModelRequirements("m_torch", "torch"))))
    assert [item.phase for item in ordered] == [
        EnvironmentPhase.PYTORCH,
        EnvironmentPhase.NATIVE_TAIL,
    ]


def test_evidenced_cuda_failure_defers_instead_of_failing() -> None:
    """Literal focused-probe CUDA evidence supports the closed deferred status."""

    evidence = PlatformEvidence(
        PlatformRequirement.CUDA,
        "focused-probe",
        "extension load reports that CUDA_HOME is required",
    )
    decision = defer_for_platform("m_cuda", PlatformRequirement.CUDA, (evidence,))
    assert decision.status == "deferred:needs-cuda"
    assert decision.evidence == (evidence,)


def test_detectron2_mmcv_pool_gets_exactly_two_arm64_attempts() -> None:
    """The first x86 build failure retries and the second creates a deferral."""

    tracker = EffortTracker({"environment": StageCap(attempts=2)})
    attempts = Arm64HeavyBuildAttempts(tracker)
    evidence = PlatformEvidence(
        PlatformRequirement.X86,
        "focused-probe",
        "compiler rejects the arm64 target in the same detectron2 source build",
    )
    assert attempts.record_failure("m_detect", evidence) is None
    decision = attempts.record_failure("m_detect", evidence)
    assert decision is not None
    assert decision.status == "deferred:needs-x86"
    assert decision.attempts == 2
    assert attempts.attempts("m_detect") == 2
    with pytest.raises(EffortCapExceeded):
        attempts.record_failure("m_detect", evidence)
    assert tracker.failures[-1].metric == "attempts"


def test_deferral_without_matching_platform_evidence_is_refused() -> None:
    """A static package name cannot speculate a platform deferral."""

    with pytest.raises(MissingPlatformEvidenceError):
        defer_for_platform("m_guess", PlatformRequirement.CUDA, ())
    x86_evidence = PlatformEvidence(
        PlatformRequirement.X86, "source", "upstream documents x86-only assembly"
    )
    with pytest.raises(MissingPlatformEvidenceError):
        defer_for_platform("m_guess", PlatformRequirement.CUDA, (x86_evidence,))
