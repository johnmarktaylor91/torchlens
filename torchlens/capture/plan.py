"""Compiled, backend-neutral capture intent for the unification spine.

The Stage 2 compatibility adapter compiles these plans beside the legacy
``Trace`` configuration.  Producers still follow their existing paths; later
stages will consume the plan directly from the capture kernel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable, Mapping


class EnrichmentLevel(str, Enum):
    """Execution-cost tier requested for an operation observation.

    ``SHELL`` retains only the topology/correctness minimum, ``METADATA``
    additionally requests descriptive operation facts, and ``PAYLOAD``
    requests a retained value lease.  Levels are demands rather than schemas.
    """

    SHELL = "shell"
    METADATA = "metadata"
    PAYLOAD = "payload"


def _freeze_intent(value: Any) -> Any:
    """Recursively freeze standard intent containers.

    Parameters
    ----------
    value
        Compiled-intent value to make read-only where possible.

    Returns
    -------
    Any
        Immutable container equivalent, or the original opaque value.
    """

    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_intent(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_intent(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_intent(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class CapturePlan:
    """Immutable compiled intent for a single capture run.

    Parameters
    ----------
    projection_target
        Internal projection selected for this run, such as ``"trace"`` or
        ``"recording"``.
    required_completeness
        Facts the consumer requires the backend to make observable.
    default_enrichment
        Demand used by operations without a more-specific entry.
    enrichment_by_operation
        Immutable operation-key to demanded enrichment mapping.
    selectors
        Normalized immediate selector intent.
    deferred_candidates
        Selector candidates that must be resolved after the forward pass.
    interventions
        Compiled intervention intent.
    storage
        Payload storage intent.
    history
        History/lookback intent.
    backward
        Backward capture intent.
    execution_context
        Execution and random-state intent.
    stop_policy
        Compiled halt/non-finite/forward-error policy.
    required_capabilities
        Backend capabilities that must be available before capture starts.
    backend_name
        Backend against which the plan was compiled.
    """

    projection_target: str
    required_completeness: frozenset[str] = field(default_factory=frozenset)
    default_enrichment: EnrichmentLevel = EnrichmentLevel.SHELL
    enrichment_by_operation: Mapping[str, EnrichmentLevel] = field(
        default_factory=lambda: MappingProxyType({})
    )
    selectors: Any = None
    deferred_candidates: tuple[Any, ...] = ()
    interventions: Any = None
    storage: Any = None
    history: Any = None
    backward: Any = None
    execution_context: Any = None
    stop_policy: Any = None
    required_capabilities: frozenset[str] = field(default_factory=frozenset)
    backend_name: str = "torch"

    def __post_init__(self) -> None:
        """Freeze collection fields so compiled intent cannot change mid-run."""

        object.__setattr__(self, "required_completeness", frozenset(self.required_completeness))
        object.__setattr__(self, "required_capabilities", frozenset(self.required_capabilities))
        object.__setattr__(self, "deferred_candidates", tuple(self.deferred_candidates))
        object.__setattr__(
            self,
            "enrichment_by_operation",
            MappingProxyType(dict(self.enrichment_by_operation)),
        )
        for field_name in (
            "selectors",
            "interventions",
            "storage",
            "history",
            "backward",
            "execution_context",
            "stop_policy",
        ):
            object.__setattr__(self, field_name, _freeze_intent(getattr(self, field_name)))

    @classmethod
    def compile(
        cls,
        *,
        projection_target: str,
        available_capabilities: Iterable[str],
        required_capabilities: Iterable[str] = (),
        required_completeness: Iterable[str] = (),
        default_enrichment: EnrichmentLevel = EnrichmentLevel.SHELL,
        enrichment_by_operation: Mapping[str, EnrichmentLevel] | None = None,
        selectors: Any = None,
        deferred_candidates: Iterable[Any] = (),
        interventions: Any = None,
        storage: Any = None,
        history: Any = None,
        backward: Any = None,
        execution_context: Any = None,
        stop_policy: Any = None,
        backend_name: str = "torch",
    ) -> "CapturePlan":
        """Compile intent and reject unsupported requirements before capture.

        Parameters
        ----------
        projection_target
            Requested internal projection.
        available_capabilities
            Capability names supplied by the selected backend.
        required_capabilities
            Capability names demanded by the request.
        required_completeness
            Facts the product requires the backend to observe.
        default_enrichment
            Default demanded enrichment level.
        enrichment_by_operation
            Per-operation enrichment overrides.
        selectors, deferred_candidates, interventions, storage, history, backward
            Normalized capture intent retained for later kernel stages.
        execution_context
            Execution-context intent.
        stop_policy
            Compiled stop/error policy.
        backend_name
            Name of the backend being compiled.

        Returns
        -------
        CapturePlan
            Frozen compiled request.

        Raises
        ------
        ValueError
            If a requested backend capability is unavailable.
        """

        available = frozenset(available_capabilities)
        required = frozenset(required_capabilities)
        unavailable = sorted(required - available)
        if unavailable:
            joined = ", ".join(unavailable)
            raise ValueError(
                f"CapturePlan for backend {backend_name!r} requires unavailable capabilities: "
                f"{joined}."
            )
        return cls(
            projection_target=projection_target,
            required_completeness=frozenset(required_completeness),
            default_enrichment=default_enrichment,
            enrichment_by_operation=enrichment_by_operation or {},
            selectors=selectors,
            deferred_candidates=tuple(deferred_candidates),
            interventions=interventions,
            storage=storage,
            history=history,
            backward=backward,
            execution_context=execution_context,
            stop_policy=stop_policy,
            required_capabilities=required,
            backend_name=backend_name,
        )

    def enrichment_for(self, operation_key: str) -> EnrichmentLevel:
        """Return the precompiled enrichment demand for one operation key.

        Parameters
        ----------
        operation_key
            Backend-normalized operation identifier.

        Returns
        -------
        EnrichmentLevel
            Specific override when present, otherwise the default demand.
        """

        return self.enrichment_by_operation.get(operation_key, self.default_enrichment)
