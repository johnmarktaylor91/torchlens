"""Sibling projections over a sealed capture run core."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Iterable

from ..ir.events import OpEvent
from .session import CapturedRunCore

if TYPE_CHECKING:
    from ..fastlog.types import ActivationRecord


def _event_from_core(core: CapturedRunCore, fact_index: int) -> OpEvent:
    """Resolve one event fact with its authoritative stable-id sidecars.

    Parameters
    ----------
    core
        Sealed source for the projection.
    fact_index
        Producer-order index of the event fact.

    Returns
    -------
    OpEvent
        Immutable event view with decision and payload fields sourced from the
        ledgers keyed by the fact's stable event identity.
    """

    fact = core.event_facts[fact_index]
    event = fact.event
    decision = core.decisions.get(fact.event_id)
    payload = core.payloads.get(fact.event_id)
    if decision is not None:
        event = replace(
            event,
            predicate_matched=decision.predicate_matched,
            intervention_fired=decision.intervention_fired,
            intervention_replaced=decision.intervention_replaced,
            fire_results=decision.fire_results,
        )
    if payload is not None:
        event = replace(event, output=payload.output)
    return event


@dataclass(frozen=True, slots=True)
class TraceProjector:
    """Read Trace Step-0 operation events from a sealed run core."""

    core: CapturedRunCore

    def events(self) -> tuple[OpEvent, ...]:
        """Return operation events in producer order.

        Returns
        -------
        tuple[OpEvent, ...]
            Repeatedly readable operation facts resolved through stable-id
            decision and payload ledgers.
        """

        return tuple(
            _event_from_core(self.core, index) for index in range(len(self.core.event_facts))
        )


@dataclass(frozen=True, slots=True)
class RecordingProjection:
    """Sparse activation records and lookup indexes projected from sealed facts."""

    records: tuple["ActivationRecord", ...]
    by_pass: dict[int, list[int]]
    by_label: dict[str, list[tuple[int, int]]]
    by_address: dict[str, list[int]]


class RecordingProjector:
    """Build Recording activation indexes directly from sealed run cores."""

    def project(self, cores: Iterable[CapturedRunCore]) -> RecordingProjection:
        """Build the sparse activation projection without first building a Trace.

        Parameters
        ----------
        cores
            Sealed cores in Recorder pass order.

        Returns
        -------
        RecordingProjection
            Retained records and the current public lookup indexes.
        """

        from .projections import activation_record_from_event

        records: list[ActivationRecord] = []
        by_pass: dict[int, list[int]] = {}
        by_label: dict[str, list[tuple[int, int]]] = {}
        by_address: dict[str, list[int]] = {}
        for core in cores:
            for event in TraceProjector(core).events():
                record = activation_record_from_event(event)
                if record is None:
                    continue
                index = len(records)
                records.append(record)
                by_pass.setdefault(record.ctx.pass_index, []).append(index)
                by_label.setdefault(record.ctx.label, []).append((record.ctx.pass_index, index))
                if record.ctx.raw_label is not None:
                    by_label.setdefault(record.ctx.raw_label, []).append(
                        (record.ctx.pass_index, index)
                    )
                if record.ctx.address is not None:
                    by_address.setdefault(record.ctx.address, []).append(index)
        return RecordingProjection(tuple(records), by_pass, by_label, by_address)
