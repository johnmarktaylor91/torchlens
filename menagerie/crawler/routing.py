"""Dependency-aware intent routing and evidenced arm64 platform deferral."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional

from menagerie.crawler.constants import EnvironmentPhase, PlatformRequirement
from menagerie.crawler.effort import EffortTracker

_NATIVE_INTENTS = {
    "tensorflow": "tf-keras-arm64",
    "tf": "tf-keras-arm64",
    "keras": "tf-keras-arm64",
    "jax": "jax-flax-arm64",
    "flax": "jax-flax-arm64",
    "paddle": "paddle-arm64",
    "paddlepaddle": "paddle-arm64",
}

_PACKAGE_INTENTS: tuple[tuple[str, frozenset[str]], ...] = (
    ("detect-misc", frozenset({"detectron2", "mmdet"})),
    ("mmlab", frozenset({"mmengine", "mmcv", "mmpretrain", "mmsegmentation"})),
    ("graph", frozenset({"torch-geometric", "torch_geometric", "dgl", "ogb"})),
    ("audio", frozenset({"torchaudio", "librosa", "speechbrain"})),
    (
        "tabular-ts",
        frozenset({"pytorch-forecasting", "tsai", "pytorch-tabnet", "sktime"}),
    ),
)


class RoutingError(ValueError):
    """Base class for invalid or unsupported routing decisions."""


class MissingPlatformEvidenceError(RoutingError):
    """Raised when code attempts a speculative CUDA/x86 deferral."""


@dataclass(frozen=True)
class ModelRequirements:
    """Framework and direct package requirements used for deterministic routing."""

    stable_id: str
    framework: str
    packages: frozenset[str] = frozenset()
    exact_repository: bool = False
    legacy_torch: bool = False


@dataclass(frozen=True)
class IntentRoute:
    """Selected environment intent and ordered campaign phase."""

    stable_id: str
    intent: str
    phase: EnvironmentPhase


@dataclass(frozen=True)
class PlatformEvidence:
    """Captured source/probe/runtime evidence for one platform requirement."""

    requirement: PlatformRequirement
    source: str
    detail: str
    package_identity: Optional[str] = None
    source_identity: Optional[str] = None

    def __post_init__(self) -> None:
        """Require literal, attributable evidence rather than a static hint."""

        if self.source not in {"source", "focused-probe", "runtime"}:
            raise ValueError("platform evidence source must be source, focused-probe, or runtime")
        if not self.detail.strip():
            raise ValueError("platform evidence detail must be non-empty")


@dataclass(frozen=True)
class DeferralDecision:
    """Closed deferred status backed by captured platform evidence."""

    stable_id: str
    status: str
    evidence: tuple[PlatformEvidence, ...]
    attempts: int


@dataclass
class PlatformEvidenceStore:
    """Learn evidence only for matching exact package/source identities."""

    _by_model: dict[str, list[PlatformEvidence]] = field(default_factory=dict)
    _by_identity: dict[str, list[PlatformEvidence]] = field(default_factory=dict)

    def collect(self, stable_id: str, evidence: PlatformEvidence) -> None:
        """Store one observed evidence item for a model and exact identities."""

        self._by_model.setdefault(stable_id, []).append(evidence)
        for identity in (evidence.package_identity, evidence.source_identity):
            if identity is not None:
                self._by_identity.setdefault(identity, []).append(evidence)

    def for_model(
        self, stable_id: str, *, identities: Iterable[str] = ()
    ) -> tuple[PlatformEvidence, ...]:
        """Return model-local plus matching exact-identity evidence."""

        collected = list(self._by_model.get(stable_id, ()))
        for identity in identities:
            collected.extend(self._by_identity.get(identity, ()))
        return tuple(dict.fromkeys(collected))


def route_model(model: ModelRequirements) -> IntentRoute:
    """Assign a model to an intent using framework and declared packages.

    Parameters
    ----------
    model:
        Model routing requirements.

    Returns
    -------
    IntentRoute
        Deterministic intent and phase.
    """

    framework = model.framework.strip().lower()
    native = _NATIVE_INTENTS.get(framework)
    if native is not None:
        return IntentRoute(model.stable_id, native, EnvironmentPhase.NATIVE_TAIL)
    if framework not in {"torch", "pytorch"}:
        raise RoutingError(f"unsupported declared framework: {model.framework!r}")
    normalized = frozenset(package.lower() for package in model.packages)
    for intent, markers in _PACKAGE_INTENTS:
        if normalized & markers:
            return IntentRoute(model.stable_id, intent, EnvironmentPhase.PYTORCH)
    if model.legacy_torch:
        intent = "legacy-torch"
    elif model.exact_repository:
        intent = "oddballs"
    else:
        intent = "core"
    return IntentRoute(model.stable_id, intent, EnvironmentPhase.PYTORCH)


def phase_routes(routes: Iterable[IntentRoute]) -> tuple[IntentRoute, ...]:
    """Order all PyTorch routes before the native-framework tail."""

    order = {EnvironmentPhase.PYTORCH: 0, EnvironmentPhase.NATIVE_TAIL: 1}
    return tuple(
        sorted(routes, key=lambda route: (order[route.phase], route.intent, route.stable_id))
    )


def defer_for_platform(
    stable_id: str,
    requirement: PlatformRequirement | str,
    evidence: Iterable[PlatformEvidence],
    *,
    attempts: int = 0,
) -> DeferralDecision:
    """Create a CUDA/x86 deferral only from matching captured evidence."""

    required = PlatformRequirement(requirement)
    matching = tuple(item for item in evidence if item.requirement is required)
    if not matching:
        raise MissingPlatformEvidenceError(
            f"cannot defer {stable_id} for {required.value} without captured evidence"
        )
    return DeferralDecision(
        stable_id=stable_id,
        status=f"deferred:needs-{required.value}",
        evidence=matching,
        attempts=attempts,
    )


class Arm64HeavyBuildAttempts:
    """Bound detectron2/mmcv CPU source builds to two recorded attempts."""

    def __init__(self, effort: EffortTracker) -> None:
        """Use the shared effort tracker for environment attempt accounting."""

        self._effort = effort
        self._attempts: dict[str, int] = {}

    def record_failure(
        self, stable_id: str, evidence: PlatformEvidence
    ) -> Optional[DeferralDecision]:
        """Record one heavy-build failure and defer after the second x86 failure.

        Parameters
        ----------
        stable_id, evidence:
            Affected model and evidence captured by its build attempt.

        Returns
        -------
        DeferralDecision or None
            Deferral on the second attempt, otherwise ``None``.
        """

        self._effort.consume("environment", attempts=1)
        count = self._attempts.get(stable_id, 0) + 1
        self._attempts[stable_id] = count
        if evidence.requirement is PlatformRequirement.CUDA:
            return defer_for_platform(stable_id, evidence.requirement, (evidence,), attempts=count)
        if count == 2:
            return defer_for_platform(
                stable_id, PlatformRequirement.X86, (evidence,), attempts=count
            )
        return None

    def attempts(self, stable_id: str) -> int:
        """Return the number of recorded heavy-build attempts for a model."""

        return self._attempts.get(stable_id, 0)


def intent_capabilities() -> Mapping[str, frozenset[str]]:
    """Return package markers used by the deterministic router."""

    return {intent: packages for intent, packages in _PACKAGE_INTENTS}
