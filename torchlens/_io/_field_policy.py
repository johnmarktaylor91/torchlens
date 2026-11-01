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
