"""Portable container registry records for capture-time boundary containers."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
import inspect
import re
from typing import Any

import torch

from .container import (
    ContainerSpec,
    DataclassField,
    DictKey,
    HFKey,
    NamedField,
    OutputPathComponent,
    TupleIndex,
    get_registered_container,
)


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
    site_aliases: tuple[Site, ...] = ()


@dataclass(slots=True)
class ContainerRecord:
    """Portable record for one container identity ordinal."""

    ordinal: int
    object_kind: str
    label: str
    first_seen_event_index: int
    roles: set[Role] = field(default_factory=set)
    snapshots: list[ContainerSnapshot] = field(default_factory=list)

    @property
    def spec(self) -> ContainerSpec:
        """Return the sole snapshot spec for this record.

        Returns
        -------
        ContainerSpec
            The only captured snapshot specification.

        Raises
        ------
        ValueError
            If the record has no snapshots or more than one snapshot.
        """

        if len(self.snapshots) != 1:
            raise ValueError(
                "ContainerRecord.spec is only available for records with exactly one "
                "snapshot; use spec_at(site=..., role=...) for multi-snapshot records."
            )
        return self.snapshots[0].spec

    def spec_at(self, *, site: Site | None = None, role: Role | None = None) -> ContainerSpec:
        """Return a snapshot spec selected by site and role.

        Parameters
        ----------
        site:
            Optional boundary site. Site aliases are considered matches.
        role:
            Optional boundary role.

        Returns
        -------
        ContainerSpec
            Matching snapshot specification.

        Raises
        ------
        ValueError
            If the selector is ambiguous or matches no snapshot.
        """

        return self.snapshot_at(site=site, role=role).spec

    def snapshot_at(
        self,
        *,
        site: Site | None = None,
        role: Role | None = None,
    ) -> ContainerSnapshot:
        """Return a snapshot selected by site and role.

        Parameters
        ----------
        site:
            Optional boundary site. Site aliases are considered matches.
        role:
            Optional boundary role.

        Returns
        -------
        ContainerSnapshot
            Matching snapshot.

        Raises
        ------
        ValueError
            If the selector is ambiguous or matches no snapshot.
        """

        snapshots = [
            snapshot
            for snapshot in self.snapshots
            if (role is None or snapshot.role == role)
            and (site is None or _snapshot_matches_site(snapshot, site))
        ]
        if len(snapshots) != 1:
            detail = "matched no snapshots" if not snapshots else "matched multiple snapshots"
            raise ValueError(
                f"ContainerRecord snapshot selector {detail}; pass a more specific "
                "site=... and role=... selector."
            )
        return snapshots[0]


@dataclass(slots=True)
class IdentityEntry:
    """Capture-only identity entry retaining the live container object."""

    obj: object
    ordinal: int


@dataclass(frozen=True, slots=True)
class WalkResult:
    """Portable structure walk result for one boundary container."""

    spec: ContainerSpec
    leaf_occurrences: tuple[ContainerLeafOccurrence, ...]
    reconstructable: bool


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
        snapshot = ContainerSnapshot(
            site=site,
            role=role,
            phase=phase,
            observed_at_event_index=observed_at_event_index,
            spec=spec,
            leaf_occurrences=leaf_occurrences,
            reconstructable=reconstructable,
        )
        if record.snapshots and _snapshots_dedup_equivalent(record.snapshots[-1], snapshot):
            previous = record.snapshots[-1]
            record.snapshots[-1] = ContainerSnapshot(
                site=previous.site,
                role=previous.role,
                phase=previous.phase,
                observed_at_event_index=previous.observed_at_event_index,
                spec=previous.spec,
                leaf_occurrences=previous.leaf_occurrences,
                reconstructable=previous.reconstructable,
                site_aliases=(*previous.site_aliases, site),
            )
        else:
            record.snapshots.append(snapshot)
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


def walk_container(value: Any, *, role: Role, capability: str) -> WalkResult | None:
    """Walk a boundary value into a portable container spec and tensor leaves.

    Parameters
    ----------
    value:
        Boundary value to inspect.
    role:
        Container role being captured.
    capability:
        Backend structure capability. ``"none"`` disables walking.

    Returns
    -------
    WalkResult | None
        Structure and tensor occurrences when ``value`` is an eligible
        tensor-bearing container, otherwise ``None``.
    """

    del role
    if capability == "none" or isinstance(value, torch.Tensor):
        return None
    if inspect.isgenerator(value) or isinstance(value, Iterator):
        opaque_spec = ContainerSpec(
            kind="opaque",
            type_module=type(value).__module__,
            type_qualname=type(value).__qualname__,
        )
        return WalkResult(spec=opaque_spec, leaf_occurrences=(), reconstructable=False)
    if not _container_has_tensor_leaf(value, memo=set()):
        return None
    spec = _build_container_spec(value)
    if spec is None:
        return None
    occurrences = tuple(
        ContainerLeafOccurrence(
            path=occurrence.path,
            producer_op_label=occurrence.producer_op_label,
            tensor_identity=occurrence.tensor_identity,
            occ_index=occ_index,
        )
        for occ_index, occurrence in enumerate(_walk_tensor_occurrences(value, path=()))
    )
    return WalkResult(spec=spec, leaf_occurrences=occurrences, reconstructable=True)


def _snapshots_dedup_equivalent(left: ContainerSnapshot, right: ContainerSnapshot) -> bool:
    """Return whether two consecutive snapshots can share one snapshot body."""

    return (
        left.role == right.role
        and left.phase == right.phase
        and left.spec == right.spec
        and left.leaf_occurrences == right.leaf_occurrences
        and left.reconstructable == right.reconstructable
    )


def _snapshot_matches_site(snapshot: ContainerSnapshot, site: Site) -> bool:
    """Return whether a snapshot was observed at ``site`` or an alias.

    Parameters
    ----------
    snapshot:
        Snapshot to inspect.
    site:
        Boundary site selector.

    Returns
    -------
    bool
        ``True`` when the primary site or one of its aliases matches.
    """

    return snapshot.site == site or site in snapshot.site_aliases


def _container_has_tensor_leaf(value: Any, *, memo: set[int]) -> bool:
    """Return whether ``value`` contains a tensor leaf at any nesting depth."""

    if isinstance(value, torch.Tensor):
        return not isinstance(value, torch.nn.Parameter)
    if _is_literal(value) or isinstance(value, torch.Size):
        return False
    if inspect.isgenerator(value) or isinstance(value, Iterator):
        return False
    object_id = id(value)
    if object_id in memo:
        return False
    memo.add(object_id)
    for _component, child in _iter_container_children(value):
        if _container_has_tensor_leaf(child, memo=memo):
            return True
    return False


def _build_container_spec(value: Any) -> ContainerSpec | None:
    """Build a portable spec for a supported container value."""

    if _is_literal(value) or isinstance(value, torch.Size):
        return ContainerSpec(kind="literal", literal_value=value)
    if inspect.isgenerator(value) or isinstance(value, Iterator):
        return ContainerSpec(
            kind="opaque",
            type_module=type(value).__module__,
            type_qualname=type(value).__qualname__,
        )
    children = tuple(_iter_container_children(value))
    child_specs: list[tuple[OutputPathComponent, ContainerSpec]] = []
    for component, child in children:
        child_spec = _build_container_spec(child)
        if child_spec is not None:
            child_specs.append((component, child_spec))
    registered = get_registered_container(type(value))
    if registered is not None:
        flattened, aux_data = registered.flatten(value)
        module, qualname = _container_type_ref(value)
        return ContainerSpec(
            kind="registered",
            length=len(flattened),
            type_module=module,
            type_qualname=qualname,
            child_specs=tuple(child_specs),
            aux_data=aux_data,
        )
    if _is_hf_model_output(value):
        keys = tuple(value.keys())
        module, qualname = _container_type_ref(value)
        return ContainerSpec(
            kind="hf_model_output",
            length=len(keys),
            keys=keys,
            type_module=module,
            type_qualname=qualname,
            child_specs=tuple(child_specs),
        )
    torch_fields = _torch_return_type_fields(value)
    if _is_namedtuple_instance(value) or torch_fields:
        fields = torch_fields or tuple(value._fields)
        module, qualname = _container_type_ref(value)
        return ContainerSpec(
            kind="namedtuple",
            length=len(value),
            fields=fields,
            type_module=module,
            type_qualname=qualname,
            child_specs=tuple(child_specs),
        )
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = tuple(field.name for field in dataclasses.fields(value))
        module, qualname = _container_type_ref(value)
        return ContainerSpec(
            kind="dataclass",
            length=len(fields),
            fields=fields,
            type_module=module,
            type_qualname=qualname,
            child_specs=tuple(child_specs),
        )
    if isinstance(value, dict):
        return ContainerSpec(
            kind="dict",
            length=len(value),
            keys=tuple(value.keys()),
            child_specs=tuple(child_specs),
        )
    if isinstance(value, tuple):
        return ContainerSpec(kind="tuple", length=len(value), child_specs=tuple(child_specs))
    if isinstance(value, list):
        return ContainerSpec(kind="list", length=len(value), child_specs=tuple(child_specs))
    return None


def _walk_tensor_occurrences(
    value: Any,
    *,
    path: tuple[OutputPathComponent, ...],
) -> Iterator[ContainerLeafOccurrence]:
    """Yield tensor leaf occurrences with stable container paths."""

    if isinstance(value, torch.Tensor):
        if not isinstance(value, torch.nn.Parameter):
            producer_label = _tensor_label(value)
            yield ContainerLeafOccurrence(
                path=path,
                producer_op_label=producer_label,
                tensor_identity=producer_label,
                occ_index=0,
            )
        return
    for component, child in _iter_container_children(value):
        yield from _walk_tensor_occurrences(child, path=(*path, component))


def _iter_container_children(value: Any) -> Iterator[tuple[OutputPathComponent, Any]]:
    """Yield supported container children without consuming generators."""

    registered = get_registered_container(type(value))
    if registered is not None:
        children, _aux_data = registered.flatten(value)
        for index, child in enumerate(children):
            yield TupleIndex(index), child
        return
    if _is_hf_model_output(value):
        for key in value.keys():
            yield HFKey(key), value[key]
        return
    torch_fields = _torch_return_type_fields(value)
    if _is_namedtuple_instance(value) or torch_fields:
        for field_name in torch_fields or tuple(value._fields):
            yield NamedField(field_name), getattr(value, field_name)
        return
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            yield DataclassField(field.name), getattr(value, field.name)
        return
    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(key, torch.Tensor):
                continue
            yield DictKey(key), child
        return
    if isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            yield TupleIndex(index), child


def _tensor_label(value: torch.Tensor) -> str | None:
    """Return TorchLens tensor label metadata without importing torch backends eagerly."""

    meta = getattr(value, "_tl", None)
    label = getattr(meta, "label_raw", None)
    return label if isinstance(label, str) else None


def _is_literal(value: Any) -> bool:
    """Return whether ``value`` is a replay-safe literal leaf."""

    return isinstance(
        value,
        (int, float, bool, str, bytes, type(None), torch.dtype, torch.device, slice),
    )


def _container_type_ref(value: Any) -> tuple[str | None, str | None]:
    """Return the importable type reference for ``value``."""

    cls = type(value)
    return cls.__module__, cls.__qualname__


def _is_namedtuple_instance(value: Any) -> bool:
    """Return whether ``value`` is a namedtuple instance."""

    return isinstance(value, tuple) and hasattr(value, "_fields")


def _is_hf_model_output(value: Any) -> bool:
    """Return whether ``value`` looks like a HuggingFace ``ModelOutput``."""

    cls = type(value)
    if any(
        base.__module__.startswith("transformers") and base.__name__ == "ModelOutput"
        for base in cls.__mro__
    ):
        return True
    return (
        (cls.__module__.startswith("transformers") or cls.__name__.endswith("ModelOutput"))
        and hasattr(value, "keys")
        and hasattr(value, "__getitem__")
    )


def _torch_return_type_fields(value: Any) -> tuple[str, ...]:
    """Best-effort field names for tuple-like torch return types."""

    if not isinstance(value, tuple):
        return ()
    n_fields = getattr(value, "n_fields", None)
    n_unnamed = getattr(value, "n_unnamed_fields", 0)
    if not isinstance(n_fields, int) or n_fields <= 0 or n_unnamed:
        return ()
    field_names = tuple(
        match.group(1)
        for match in re.finditer(r"^\s*([A-Za-z_]\w*)=", repr(value), flags=re.MULTILINE)
    )
    if len(field_names) != n_fields:
        return ()
    return field_names
