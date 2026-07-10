"""Fact, decision, payload, and completeness ledgers for capture sessions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from ..ir.events import OpEvent


@dataclass(frozen=True, slots=True, order=True)
class EventId:
    """Stable identity for one operation event within a capture session.

    Parameters
    ----------
    raw_index
        Monotonic raw operation index reserved by the producer.
    label_raw
        Producer-stable raw label for the operation output.
    """

    raw_index: int
    label_raw: str

    @classmethod
    def from_event(cls, event: OpEvent) -> "EventId":
        """Build the stable session identity for an emitted operation event.

        Parameters
        ----------
        event
            Immutable operation event emitted by an existing producer.

        Returns
        -------
        EventId
            Identity keyed by the producer's existing raw reservation.
        """

        return cls(raw_index=event.raw_index, label_raw=event.label_raw)


@dataclass(frozen=True, slots=True)
class EventFact:
    """One immutable event fact stored in the journal.

    Parameters
    ----------
    event_id
        Stable event identity.
    event
        Existing frozen backend event; the adapter does not transform it.
    """

    event_id: EventId
    event: OpEvent


class EventJournal:
    """Append-only journal of immutable operation facts.

    The mutable journal owns only indexing and ordering.  Its public snapshots
    expose frozen :class:`EventFact` objects, so a later projector cannot alter
    producer facts through this adapter.
    """

    def __init__(self) -> None:
        """Initialize an empty journal."""

        self._facts: list[EventFact] = []
        self._by_id: dict[EventId, EventFact] = {}

    def append(self, event: OpEvent) -> EventId:
        """Append an immutable producer fact and return its stable identity.

        Parameters
        ----------
        event
            Existing event to journal without copying or enriching it.

        Returns
        -------
        EventId
            Stable key for all event sidecars.

        Raises
        ------
        ValueError
            If the producer attempts to reuse a stable event identity.
        """

        event_id = EventId.from_event(event)
        if event_id in self._by_id:
            raise ValueError(f"Duplicate stable capture event id: {event_id!r}")
        fact = EventFact(event_id=event_id, event=event)
        self._facts.append(fact)
        self._by_id[event_id] = fact
        return event_id

    def replace(self, event: OpEvent) -> EventId:
        """Replace a fact with its immutable producer-updated event.

        Parameters
        ----------
        event
            Updated frozen event with the same stable identity.

        Returns
        -------
        EventId
            Stable identity retained by the replacement.

        Raises
        ------
        KeyError
            If no previously journaled fact owns the event identity.
        """

        event_id = EventId.from_event(event)
        if event_id not in self._by_id:
            raise KeyError(event_id)
        fact = EventFact(event_id=event_id, event=event)
        self._by_id[event_id] = fact
        for index, existing in enumerate(self._facts):
            if existing.event_id == event_id:
                self._facts[index] = fact
                break
        return event_id

    def clear(self) -> None:
        """Release all journaled event references.

        Returns
        -------
        None
            Removes the session-local mirror once legacy forward cleanup has
            reached its historical release point.
        """

        self._facts.clear()
        self._by_id.clear()

    @property
    def facts(self) -> tuple[EventFact, ...]:
        """Return journaled facts in producer order."""

        return tuple(self._facts)

    @property
    def by_id(self) -> Mapping[EventId, EventFact]:
        """Return a read-only stable-id lookup of journaled facts."""

        return MappingProxyType(self._by_id)


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    """Selection and intervention decisions for one operation event.

    Parameters
    ----------
    predicate_matched
        Existing producer predicate decision.
    intervention_fired
        Whether an intervention fired.
    intervention_replaced
        Whether an intervention returned a replacement output.
    fire_results
        Existing intervention sidecar facts.
    """

    predicate_matched: bool
    intervention_fired: bool
    intervention_replaced: bool
    fire_results: tuple[Any, ...]


class DecisionLedger:
    """Mutable sidecar ledger for event-local selection decisions."""

    def __init__(self) -> None:
        """Initialize an empty decision ledger."""

        self._records: dict[EventId, DecisionRecord] = {}

    def append_from_event(self, event_id: EventId, event: OpEvent) -> None:
        """Record the existing event decision fields without re-evaluating them.

        Parameters
        ----------
        event_id
            Stable identity for ``event``.
        event
            Existing producer event.
        """

        self._records[event_id] = DecisionRecord(
            predicate_matched=event.predicate_matched,
            intervention_fired=event.intervention_fired,
            intervention_replaced=event.intervention_replaced,
            fire_results=event.fire_results,
        )

    def clear(self) -> None:
        """Release all decision sidecars.

        Returns
        -------
        None
            Removes session-local references after the active run ends.
        """

        self._records.clear()

    @property
    def records(self) -> Mapping[EventId, DecisionRecord]:
        """Return a read-only stable-id lookup of decisions."""

        return MappingProxyType(self._records)


@dataclass(frozen=True, slots=True)
class PayloadRecord:
    """Payload lease sidecar for one event.

    Parameters
    ----------
    output
        Existing event output reference.  The adapter creates no tensor copy,
        storage write, or additional payload retention.
    """

    output: object


class PayloadLedger:
    """Mutable sidecar ledger for producer-owned payload leases."""

    def __init__(self) -> None:
        """Initialize an empty payload ledger."""

        self._records: dict[EventId, PayloadRecord] = {}

    def append_from_event(self, event_id: EventId, event: OpEvent) -> None:
        """Reference an existing event payload without retaining a new copy.

        Parameters
        ----------
        event_id
            Stable identity for ``event``.
        event
            Existing producer event.
        """

        self._records[event_id] = PayloadRecord(output=event.output)

    def clear(self) -> None:
        """Release all payload leases.

        Returns
        -------
        None
            Drops activation references at the legacy forward-memory release
            point rather than extending their lifetime through the session.
        """

        self._records.clear()

    @property
    def records(self) -> Mapping[EventId, PayloadRecord]:
        """Return a read-only stable-id lookup of payload sidecars."""

        return MappingProxyType(self._records)


class CompletenessState(str, Enum):
    """Truthful observability state for one requested capture fact."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class CompletenessManifest:
    """Observability facts where an absent entry is always unavailable.

    This deliberately does not use booleans: a missing observation must never
    be read as evidence that an event did *not* occur.
    """

    def __init__(self) -> None:
        """Initialize an empty manifest with no claimed observations."""

        self._states: dict[str, CompletenessState] = {}

    def mark_available(self, demand: str) -> None:
        """Mark one requested fact as observed.

        Parameters
        ----------
        demand
            Backend-neutral fact demand that was observable.
        """

        self._states[demand] = CompletenessState.AVAILABLE

    def mark_unavailable(self, demand: str) -> None:
        """Explicitly mark one requested fact unavailable.

        Parameters
        ----------
        demand
            Backend-neutral fact demand that was unavailable.
        """

        self._states[demand] = CompletenessState.UNAVAILABLE

    def clear(self) -> None:
        """Release all recorded completeness state.

        Returns
        -------
        None
            Clears run-local observability bookkeeping after teardown.
        """

        self._states.clear()

    def state_for(self, demand: str) -> CompletenessState:
        """Return observability for one demand, defaulting to unavailable.

        Parameters
        ----------
        demand
            Backend-neutral fact demand to inspect.

        Returns
        -------
        CompletenessState
            ``UNAVAILABLE`` for every missing entry, never a false event fact.
        """

        return self._states.get(demand, CompletenessState.UNAVAILABLE)

    @property
    def states(self) -> Mapping[str, CompletenessState]:
        """Return a read-only snapshot of explicitly recorded observability."""

        return MappingProxyType(self._states)
