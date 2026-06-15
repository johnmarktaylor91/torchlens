"""Portable container registry records for capture-time boundary containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .container import ContainerSpec, OutputPathComponent


class Role(Enum):
    """Boundary role for an observed container object."""

    MODEL_INPUT = "model_input"
    CALL_INPUT = "call_input"
    CALL_OUTPUT = "call_output"
    MODEL_OUTPUT = "model_output"


class Phase(Enum):
    """Capture phase at which a container snapshot was observed."""

    PRE_CALL = "pre_call"
    POST_CALL = "post_call"


@dataclass(frozen=True, slots=True)
class FuncSite:
    """Function-call boundary where a container was observed."""

    func_call_id: int
    position: object


@dataclass(frozen=True, slots=True)
class ModuleSite:
    """Module-call boundary where a container was observed."""

    module_call_label: str
    position: object


@dataclass(frozen=True, slots=True)
class ModelSite:
    """Model boundary where a container was observed."""

    model_ref: str
    position: object


Site = FuncSite | ModuleSite | ModelSite


@dataclass(frozen=True, slots=True)
class ContainerLeafOccurrence:
    """One tensor occurrence at one container path."""

    path: tuple[OutputPathComponent, ...]
    producer_op_label: str | None
    tensor_identity: str | None
    occ_index: int


@dataclass(frozen=True, slots=True)
class ContainerSnapshot:
    """Portable snapshot of one container observation."""

    site: Site
    role: Role
    phase: Phase
    observed_at_event_index: int
    spec: ContainerSpec
    leaf_occurrences: tuple[ContainerLeafOccurrence, ...]
    reconstructable: bool


@dataclass(slots=True)
class ContainerRecord:
    """Portable record for one container identity ordinal."""

    ordinal: int
    object_kind: str
    label: str
    first_seen_event_index: int
    roles: set[Role] = field(default_factory=set)
    snapshots: list[ContainerSnapshot] = field(default_factory=list)


@dataclass(slots=True)
class IdentityEntry:
    """Capture-only identity entry retaining the live container object."""

    obj: object
    ordinal: int


@dataclass(slots=True)
class ContainerRegistry:
    """Capture-time two-tier registry for boundary container identities."""

    id_to_entry: dict[int, IdentityEntry] = field(default_factory=dict)
    records: dict[int, ContainerRecord] = field(default_factory=dict)
    next_ordinal: int = 0

    def register_snapshot(
        self,
        container: object,
        *,
        site: Site,
        role: Role,
        phase: Phase,
        observed_at_event_index: int,
        spec: ContainerSpec,
        leaf_occurrences: tuple[ContainerLeafOccurrence, ...],
        reconstructable: bool,
    ) -> ContainerRecord:
        """Register one observed container snapshot.

        Parameters
        ----------
        container:
            Live container object whose identity is being tracked.
        site:
            Boundary site for this observation.
        role:
            Input/output role at the boundary.
        phase:
            Pre-call or post-call observation phase.
        observed_at_event_index:
            Monotonic capture event index for the observation.
        spec:
            Portable container structure.
        leaf_occurrences:
            Ordered tensor leaf occurrences. Repeated tensors at different paths
            must appear more than once.
        reconstructable:
            Whether ``spec`` is sufficient for runtime reconstruction.

        Returns
        -------
        ContainerRecord
            Portable record updated with the snapshot.
        """

        entry = self._entry_for(container, observed_at_event_index=observed_at_event_index)
        record = self.records[entry.ordinal]
        record.roles.add(role)
        record.snapshots.append(
            ContainerSnapshot(
                site=site,
                role=role,
                phase=phase,
                observed_at_event_index=observed_at_event_index,
                spec=spec,
                leaf_occurrences=leaf_occurrences,
                reconstructable=reconstructable,
            )
        )
        return record

    def clear_live_state(self) -> None:
        """Release capture-only strong references and identity indexes."""

        self.id_to_entry.clear()

    def _entry_for(self, container: object, *, observed_at_event_index: int) -> IdentityEntry:
        """Return or create the identity entry for ``container``.

        Parameters
        ----------
        container:
            Live container object.
        observed_at_event_index:
            First-seen event index used for newly-created records.

        Returns
        -------
        IdentityEntry
            Capture-only entry for the current live object.
        """

        object_id = id(container)
        entry = self.id_to_entry.get(object_id)
        if entry is not None and entry.obj is container:
            return entry

        ordinal = self.next_ordinal
        self.next_ordinal += 1
        entry = IdentityEntry(obj=container, ordinal=ordinal)
        self.id_to_entry[object_id] = entry
        object_kind = _object_kind(container)
        self.records[ordinal] = ContainerRecord(
            ordinal=ordinal,
            object_kind=object_kind,
            label=f"{object_kind}#{ordinal}",
            first_seen_event_index=observed_at_event_index,
        )
        return entry


def _object_kind(container: object) -> str:
    """Return a portable display kind for a container object.

    Parameters
    ----------
    container:
        Container instance.

    Returns
    -------
    str
        ``module.qualname`` for the concrete container type.
    """

    container_type = type(container)
    return f"{container_type.__module__}.{container_type.__qualname__}"
