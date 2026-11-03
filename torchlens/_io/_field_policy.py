"""``FieldPolicy``: the portable scrub policy enum (leaf module, no ``_io`` import).

The IR, record and intervention vocabularies import the enum from here so
they never execute the ``torchlens._io`` facade (which imports torch and the
whole format contract). ``torchlens._io`` and ``format_contract`` re-export
it unchanged.
"""

from __future__ import annotations

from enum import Enum


class FieldPolicy(str, Enum):
    """Portable scrub policy for one serialized field."""

    KEEP = "keep"
    BLOB = "blob"
    BLOB_RECURSIVE = "blob_recursive"
    DROP = "drop"
    STRINGIFY = "stringify"
    WEAKREF_STRIP = "weakref_strip"


# Pickle-visible identity stays the historical facade path: bundle metadata
# pickles reference this type by (module, qualname), and the guarded
# unpickler allowlists exactly that path -- the physical split must not
# change a single persisted byte (move-compat law, memo org rule 9).
FieldPolicy.__module__ = "torchlens._io"


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


# Preserve the existing enum identity while its definition lives below record classes.
StorageKind.__module__ = "torchlens._data_classes.field_policy"
