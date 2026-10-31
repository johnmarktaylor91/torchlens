"""Structural field-policy tables for TorchLens record classes."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .._io import FieldPolicy


class StorageKind(Enum):
    """Physical storage class of one declared field in the trace core.

    The declared schema (docs/reference/trace_core_design.md section 3.4)
    carries one storage kind per field; the column layout, facade
    descriptors, and state adapters are generated from it. Persistence
    policy (``FieldPolicy``) NEVER implies a storage kind — the axes stay
    separate.
    """

    #: Plain per-row scalar (numbers, bools, short strings, enums).
    SCALAR = "scalar"
    #: Value interned in the per-trace pool; the row stores a pool id.
    INTERNED = "interned"
    #: Ancestor-closure pool id (bitset or sparse encoding, pool-internal).
    BITSET = "bitset"
    #: Variable-cardinality relation stored in the edge-occurrence table.
    EDGE = "edge"
    #: Group-shared fact stored once per group block, joined by group id.
    GROUP = "group"
    #: Tensor/blob payload handle into the identity-preserving arena.
    PAYLOAD = "payload"
    #: Non-portable runtime/session state (never serialized, never columnar).
    RUNTIME = "runtime"
    #: Computed on access; owns no storage at all.
    COMPUTED = "computed"


@dataclass(frozen=True)
class StorageBinding:
    """Physical-storage axes for one declared field.

    Parameters
    ----------
    kind:
        Storage class of the field's backing.
    annotation:
        Public type annotation string presented on the generated facade
        descriptor, or ``None`` when the class declares none.
    codec:
        Physical column codec name (resolved when the field's family
        columnarizes), or ``None`` for object-backed storage.
    mutability:
        ``"immutable"``, ``"mutable_container"`` (public in-place mutation is
        observable and must stay so), or ``"copy_on_read"`` (every read
        returns a fresh container; caller mutation is discarded).
    """

    kind: StorageKind
    annotation: str | None = None
    codec: str | None = None
    mutability: str = "immutable"


@dataclass(frozen=True)
class RecordFieldPolicy:
    """Policy metadata for one field on one TorchLens record class.

    Parameters
    ----------
    name:
        Field name on the record class.
    order:
        Zero-based FIELD_ORDER position, or ``None`` for runtime/internal fields
        that must still have portable policy coverage.
    portable_policy:
        Portable scrub policy used by save/load.
    fork_policy:
        Optional Trace fork policy for Trace/Op fields.
    default_fill:
        Default used when older serialized state is missing this field.
    user_facing:
        Whether the field belongs to the record's public FIELD_ORDER surface.
    """

    name: str
    order: int | None
    portable_policy: FieldPolicy
    fork_policy: Any | None = None
    default_fill: Any = None
    user_facing: bool = True
    storage: StorageBinding | None = None


RecordFieldPolicyTable = OrderedDict[str, RecordFieldPolicy]


def build_record_field_policy_table(
    field_order: list[str],
    portable_state_spec: Mapping[str, FieldPolicy],
    *,
    fork_policy: Mapping[str, Any] | None = None,
    default_fill_state: Mapping[str, Any] | None = None,
    schema_key: str | None = None,
) -> RecordFieldPolicyTable:
    """Build one structural policy table for a record class.

    Parameters
    ----------
    field_order:
        Ordered user-facing field names for the record.
    portable_state_spec:
        Portable scrub policy by field name. Fields not in ``field_order`` are
        retained in the policy table as non-user-facing runtime fields.
    fork_policy:
        Optional fork-copy policy by field name.
    default_fill_state:
        Optional default-fill values by field name.
    schema_key:
        Declared-schema key ("op", "trace", ...) used to attach the
        checked-in ``StorageBinding`` for each field. ``None`` leaves the
        ``storage`` axis unset (legacy callers).

    Returns
    -------
    RecordFieldPolicyTable
        Ordered policy table. FIELD_ORDER entries appear first, then portable
        runtime-only entries in their portable-spec order.
    """

    fork_policy = fork_policy or {}
    default_fill_state = default_fill_state or {}
    storage_bindings: Mapping[str, StorageBinding] = {}
    if schema_key is not None:
        # Deferred import: the bindings module is generated data that imports
        # StorageBinding/StorageKind from this module. Missing entries are
        # tolerated only so the generator can bootstrap; the regenerate-and-
        # diff CI test enforces completeness.
        from ._schema_bindings import STORAGE_BINDINGS

        storage_bindings = STORAGE_BINDINGS.get(schema_key, {})
    table: RecordFieldPolicyTable = OrderedDict()
    for index, field_name in enumerate(field_order):
        table[field_name] = RecordFieldPolicy(
            name=field_name,
            order=index,
            portable_policy=portable_state_spec.get(field_name, FieldPolicy.KEEP),
            fork_policy=fork_policy.get(field_name),
            default_fill=default_fill_state.get(field_name),
            user_facing=True,
            storage=storage_bindings.get(field_name),
        )
    for field_name, portable_policy in portable_state_spec.items():
        if field_name in table:
            continue
        table[field_name] = RecordFieldPolicy(
            name=field_name,
            order=None,
            portable_policy=portable_policy,
            fork_policy=fork_policy.get(field_name),
            default_fill=default_fill_state.get(field_name),
            user_facing=False,
            storage=storage_bindings.get(field_name),
        )
    return table


def field_order_from_policy(table: Mapping[str, RecordFieldPolicy]) -> list[str]:
    """Return the FIELD_ORDER view generated from a record policy table.

    Parameters
    ----------
    table:
        Record field policy table.

    Returns
    -------
    list[str]
        User-facing fields sorted by declared order.
    """

    ordered = [policy for policy in table.values() if policy.user_facing]
    return [policy.name for policy in sorted(ordered, key=lambda policy: policy.order or 0)]


def portable_state_spec_from_policy(
    table: Mapping[str, RecordFieldPolicy],
) -> dict[str, FieldPolicy]:
    """Return the portable-state view generated from a policy table.

    Parameters
    ----------
    table:
        Record field policy table.

    Returns
    -------
    dict[str, FieldPolicy]
        Portable scrub policy by field name.
    """

    return {name: policy.portable_policy for name, policy in table.items()}


def fork_policy_from_policy(table: Mapping[str, RecordFieldPolicy]) -> dict[str, Any]:
    """Return the fork-policy view generated from a policy table.

    Parameters
    ----------
    table:
        Record field policy table.

    Returns
    -------
    dict[str, Any]
        Fork policy by user-facing field name.
    """

    return {
        name: policy.fork_policy
        for name, policy in table.items()
        if policy.user_facing and policy.fork_policy is not None
    }


def default_fill_state_from_policy(table: Mapping[str, RecordFieldPolicy]) -> dict[str, Any]:
    """Return the default-fill view generated from a policy table.

    Parameters
    ----------
    table:
        Record field policy table.

    Returns
    -------
    dict[str, Any]
        Default-fill values by user-facing field name.
    """

    return {name: policy.default_fill for name, policy in table.items() if policy.user_facing}
