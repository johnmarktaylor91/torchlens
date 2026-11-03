"""Evidence-item origin classes, seams and the carry-mode vocabulary (FK-5, FK-6).

Every graded evidence item on a relation row names WHERE its claim was
measured: its ``class`` is one of the seven ground-truth origin kinds in
:data:`EVIDENCE_ORIGIN_KINDS`. The vocabulary is closed and SCHEMA-SCOPED:
the two TorchLens-owned evidence schemas (``version_boundary_v1`` and
``turn_boundary_v1``) each admit a subset and REQUIRE the key; a foreign
schema id admits any of the seven when the key is present. Both doors (relate
and load) validate it, so a persisted item never carries an uncontracted
class.

Rollups are DERIVED, never persisted (FK-3, FK-6): ``exogenous`` is the count
of items whose class lies outside the member-measured kinds
(:data:`EVIDENCE_EXOGENOUS_ORIGIN_KINDS`), computed at read time by
:mod:`torchlens.bundle._relation_grades`; no item ever stores
``class="exogenous"``.

``carry_mode`` on a ``successor_of`` row is a closed vocabulary
(:data:`RELATION_CARRY_MODES`): ``none`` is a declared fresh-start claim that
GRADES ``disclosed`` and is checkable as a class (a ``declared_carry`` item
that diverges is detected secretly-carried history), never a blank
unavailable (FK-5).

Seams: a ``successor_of`` row may DECLARE the seams its carry crosses
(``seams=("chat_template",)``); an item may name one of them under ``seam``.
A divergence at a declared seam is seam-local: it stays ``divergent`` on
record but never escalates the row to ``broken`` (FK-2).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

__all__ = [
    "EVIDENCE_CLASS_KEY",
    "EVIDENCE_EXOGENOUS_ORIGIN_KINDS",
    "EVIDENCE_MEMBER_ORIGIN_KINDS",
    "EVIDENCE_ORIGIN_KINDS",
    "EVIDENCE_SCHEMA_ORIGIN_KINDS",
    "EVIDENCE_SEAM_KEY",
    "RELATION_CARRY_MODES",
]

#: Persisted item key carrying the origin kind (the ruling's ``item.class``).
EVIDENCE_CLASS_KEY = "class"

#: Persisted item key naming the declared seam a claim was measured across.
EVIDENCE_SEAM_KEY = "seam"

#: Origin kinds measured FROM a member's own captured record.
EVIDENCE_MEMBER_ORIGIN_KINDS = frozenset(
    {
        # The member's input stream (prompt tokens, carried prefix).
        "member_input",
        # The member's output stream (generated tokens, logits).
        "member_output",
        # The member's entry or exit state digest (cache, hidden state).
        "member_state",
        # The member's parameter snapshot.
        "member_params",
    }
)

#: Origin kinds declared or recorded OUTSIDE the member's captured record.
#: Their count is the derived ``exogenous`` rollup (FK-3).
EVIDENCE_EXOGENOUS_ORIGIN_KINDS = frozenset(
    {
        # A declared carry claim between the two members (what ``carry_mode`` asserts).
        "declared_carry",
        # A declared registered transform between them (a chat template, a quantization step).
        "declared_transform",
        # A linked harness record (a correlation id: a pointer, never proof).
        "harness_record",
    }
)

#: The seven ground-truth origin kinds (FK-6), closed.
EVIDENCE_ORIGIN_KINDS = EVIDENCE_MEMBER_ORIGIN_KINDS | EVIDENCE_EXOGENOUS_ORIGIN_KINDS

#: Schema-scoped origin vocabularies for the TorchLens-owned evidence schemas.
#: Items under these schema ids REQUIRE ``class`` and must draw it from the
#: schema's subset; a foreign schema id admits any of the seven when present.
EVIDENCE_SCHEMA_ORIGIN_KINDS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "version_boundary_v1": frozenset(
            {"member_input", "member_output", "member_state", "member_params", "declared_transform"}
        ),
        "turn_boundary_v1": frozenset(
            {
                "member_input",
                "member_output",
                "member_state",
                "declared_carry",
                "declared_transform",
                "harness_record",
            }
        ),
    }
)

#: Closed ``carry_mode`` vocabulary for ``successor_of`` rows (FK-5).
#:
#: - ``none``: a declared fresh start; nothing of the earlier member is carried.
#: - ``verbatim``: the earlier member's record is carried byte-for-byte.
#: - ``summarized``: a declared lossy carry (a summary, a truncation).
RELATION_CARRY_MODES = frozenset({"none", "verbatim", "summarized"})


def validate_carry_mode(value: Any) -> None:
    """Refuse a ``carry_mode`` outside :data:`RELATION_CARRY_MODES`."""

    if value not in RELATION_CARRY_MODES:
        raise ValueError(
            f"bundle-relation param 'carry_mode' is {value!r}, outside the closed "
            f"vocabulary {sorted(RELATION_CARRY_MODES)}"
        )


def validate_seams(value: Any) -> tuple[str, ...]:
    """Return the declared seams as a tuple of distinct non-empty names, refusing anything else."""

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(
            f"bundle-relation param 'seams' must be a sequence of seam names, got {value!r}"
        )
    seams = tuple(value)
    if not seams or not all(isinstance(seam, str) and seam for seam in seams):
        raise ValueError(
            f"bundle-relation param 'seams' must hold non-empty seam-name strings, got {value!r}"
        )
    if len(set(seams)) != len(seams):
        raise ValueError(f"bundle-relation param 'seams' repeats a seam name: {value!r}")
    return seams


def validate_evidence_class(index: int, item: Mapping[str, Any], *, schema: str) -> None:
    """Validate one item's ``class`` against the schema-scoped closed vocabulary (FK-6)."""

    scoped = EVIDENCE_SCHEMA_ORIGIN_KINDS.get(schema)
    if EVIDENCE_CLASS_KEY not in item:
        if scoped is None:
            return
        raise ValueError(
            f"bundle-relation evidence item {index} under schema {schema!r} is missing "
            f"the 'class' key; this schema requires one of {sorted(scoped)}"
        )
    kind = item[EVIDENCE_CLASS_KEY]
    admitted = scoped if scoped is not None else EVIDENCE_ORIGIN_KINDS
    if kind not in admitted:
        raise ValueError(
            f"bundle-relation evidence item {index} carries class {kind!r}, outside the "
            f"closed origin vocabulary for schema {schema!r}: {sorted(admitted)} "
            "(rollups such as 'exogenous' are derived at read, never persisted)"
        )


def validate_evidence_seam(index: int, item: Mapping[str, Any], *, seams: Sequence[str]) -> None:
    """Refuse an item ``seam`` the row did not declare under ``seams``."""

    if EVIDENCE_SEAM_KEY not in item:
        return
    seam = item[EVIDENCE_SEAM_KEY]
    if not isinstance(seam, str) or not seam:
        raise ValueError(
            f"bundle-relation evidence item {index} 'seam' must be a non-empty seam name, got {seam!r}"
        )
    if seam not in seams:
        raise ValueError(
            f"bundle-relation evidence item {index} names seam {seam!r}, which the row does "
            f"not declare under 'seams' ({sorted(seams)}); a seam-local reading needs a "
            "declared seam"
        )
