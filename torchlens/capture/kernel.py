"""Fixed-order controller for live operation capture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from .plan import EnrichmentLevel

if TYPE_CHECKING:
    from .session import CaptureSession

InterventionTarget = Callable[[Any], Any]
ProducerTarget = Callable[..., None]


@dataclass(slots=True)
class OpObservation:
    """Transient live values observed for one backend operation.

    Parameters
    ----------
    operation_key
        Backend-normalized operation name used for enrichment lookup.
    value
        Live output value.  This object must not escape into the durable journal.
    """

    operation_key: str
    value: Any


class CaptureKernel:
    """Run one statically ordered operation pipeline.

    The producer targets remain compatibility adapters in Stage 4a: they build
    the exact legacy immutable events, while this controller fixes the order in
    which intervention and production are entered.  Tier targets are compiled
    once per operation key and disabled tiers are represented by ``None``.
    """

    __slots__ = ("_session", "_default_metadata", "_default_payload")

    def __init__(self, session: CaptureSession) -> None:
        """Compile direct enrichment gates for a capture session.

        Parameters
        ----------
        session
            Mutable owner of the active capture run.
        """

        self._session = session
        default = session.plan.default_enrichment
        self._default_metadata = default in {EnrichmentLevel.METADATA, EnrichmentLevel.PAYLOAD}
        self._default_payload = default is EnrichmentLevel.PAYLOAD

    def apply_intervention(
        self,
        observation: OpObservation,
        target: InterventionTarget,
    ) -> Any:
        """Apply intervention to a live value before any durable fact freezes.

        Parameters
        ----------
        observation
            Transient observation holding the live backend value.
        target
            Pre-existing live intervention implementation.

        Returns
        -------
        Any
            Original or replaced value to pass to downstream user code.
        """

        self._reserve_identity_context(observation.operation_key)
        observation.value = target(observation.value)
        self._session.counters["kernel_interventions"] = (
            self._session.counters.get("kernel_interventions", 0) + 1
        )
        return observation.value

    def emit(
        self,
        operation_key: str,
        producer: ProducerTarget,
        *producer_args: Any,
    ) -> None:
        """Run the fixed post-intervention producer pipeline.

        Parameters
        ----------
        operation_key
            Backend-normalized operation name.
        producer
            Precompiled exhaustive, predicate, or refresh producer target.
        *producer_args
            Existing positional producer inputs, passed through unchanged.
        """

        self._reserve_identity_context(operation_key)
        metadata_enabled, payload_enabled = self._enrichment_gates(operation_key)
        if metadata_enabled:
            self._normalize_metadata()
        self._select_or_defer()
        if payload_enabled:
            self._retain_payload()
        producer(*producer_args)
        self._append_facts_and_sidecars()
        self._update_indexes_history()
        self._evaluate_nonfinite_halt()

    def _enrichment_gates(self, operation_key: str) -> tuple[bool, bool]:
        """Return direct metadata and payload gates for an operation key."""

        override = self._session.plan.enrichment_by_operation.get(operation_key)
        if override is None:
            return self._default_metadata, self._default_payload
        return (
            override in {EnrichmentLevel.METADATA, EnrichmentLevel.PAYLOAD},
            override is EnrichmentLevel.PAYLOAD,
        )

    def _reserve_identity_context(self, operation_key: str) -> None:
        """Mark entry into identity/context reservation without allocating sidecars."""

        del operation_key
        self._session.counters["kernel_observations"] = (
            self._session.counters.get("kernel_observations", 0) + 1
        )

    def _normalize_metadata(self) -> None:
        """Enter the demanded metadata normalization tier."""

        self._session.counters["kernel_metadata"] = (
            self._session.counters.get("kernel_metadata", 0) + 1
        )

    def _select_or_defer(self) -> None:
        """Enter the selection/defer stage."""

    def _retain_payload(self) -> None:
        """Enter the demanded payload-retention tier."""

        self._session.counters["kernel_payload"] = (
            self._session.counters.get("kernel_payload", 0) + 1
        )

    def _append_facts_and_sidecars(self) -> None:
        """Mark completion of producer append into journal and sidecars."""

    def _update_indexes_history(self) -> None:
        """Mark completion of producer index and history updates."""

    def _evaluate_nonfinite_halt(self) -> None:
        """Mark completion of producer non-finite and halt evaluation."""
