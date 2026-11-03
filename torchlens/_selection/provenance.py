"""Per-entry provenance disclosure for :mod:`torchlens._selection.selection`.

One resolved site entry carries a :class:`SelectionProvenance`: the closed
relation lattice (how the exact-as-a-set mask relates to the true semantic
support), a free-text ``source``, and two optional typed disclosures a
producer may set (``basis``: how the entry was computed; ``field_status``:
how exact the field behind it is). The lattice transfer functions used by the
selection operators live here too. The names callers use are re-exported from
``torchlens._selection.selection``, the home its callers import from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RelationName = Literal["exact", "upper_bound", "lower_bound", "unknown"]
_RELATIONS: tuple[str, ...] = ("exact", "upper_bound", "lower_bound", "unknown")

#: How a producer computed an entry: propagated field geometry, or
#: per-unit autograd probes.
_BASES: tuple[str, ...] = ("geometric", "gradient")

#: How exact the field behind an entry is: the geometric field statuses a
#: producer folds (``exact``, ``whole_input``, ``upper_bound``), measured by
#: probes (``empirical``), or no field consulted (``none``).
_FIELD_STATUSES: tuple[str, ...] = ("exact", "whole_input", "upper_bound", "empirical", "none")


# ---------------------------------------------------------------------------
# Provenance relation lattice (closed, S2-routed) + composition tables.
# ---------------------------------------------------------------------------

_FLIP: dict[str, str] = {
    "exact": "exact",
    "upper_bound": "lower_bound",
    "lower_bound": "upper_bound",
    "unknown": "unknown",
}


def _join_relation(a: str, b: str) -> str:
    """JOIN transfer for both ``|`` and ``&`` (symmetric; exact is identity)."""

    if a == "exact":
        return b
    if b == "exact":
        return a
    if a == b:
        return a
    return "unknown"


def _difference_relation(a: str, b: str) -> str:
    """DIFFERENCE transfer (ordered): ``rel(A - B) = join(rel(A), FLIP(rel(B)))``."""

    return _join_relation(a, _FLIP[b])


def _meet_relation(a: str, b: str) -> str:
    """Multi-site MEET (exact on top, unknown on bottom, upper/lower incomparable)."""

    return _join_relation(a, b)


@dataclass(frozen=True)
class SelectionProvenance:
    """Disclosure attached to one resolved site entry.

    ``relation`` states how the mask relates to true semantic support:
    ``exact`` (the mask IS the support as a set), ``upper_bound`` (superset),
    ``lower_bound`` (subset), or ``unknown``. Producer inexactness (interval
    hulls) rides here — the mask itself is always exact as a set.
    ``basis`` (``geometric`` | ``gradient``) and ``field_status`` (``exact``
    | ``whole_input`` | ``upper_bound`` | ``empirical`` | ``none``) are
    ``None`` unless the producer states them.
    """

    relation: str
    source: str = ""
    basis: str | None = None
    field_status: str | None = None

    def __post_init__(self) -> None:
        """Validate the closed relation, basis and field-status vocabularies."""

        if self.relation not in _RELATIONS:
            raise ValueError(
                f"provenance relation must be one of {_RELATIONS}; got {self.relation!r}."
            )
        if self.basis is not None and self.basis not in _BASES:
            raise ValueError(f"provenance basis must be one of {_BASES}; got {self.basis!r}.")
        if self.field_status is not None and self.field_status not in _FIELD_STATUSES:
            raise ValueError(
                f"provenance field_status must be one of {_FIELD_STATUSES}; "
                f"got {self.field_status!r}."
            )


def _flip_provenance(provenance: SelectionProvenance) -> SelectionProvenance:
    """Provenance of a complemented entry: the relation flips, the rest stays."""

    return SelectionProvenance(
        relation=_FLIP[provenance.relation],
        source=provenance.source,
        basis=provenance.basis,
        field_status=provenance.field_status,
    )


def _compose_provenance(
    left: SelectionProvenance, right: SelectionProvenance, relation: str
) -> SelectionProvenance:
    """Provenance of two entries composed at one site.

    ``relation`` is the operator's lattice transfer of the two relations. A
    typed disclosure survives only when both operands state the same value;
    otherwise the composed entry no longer has one basis or field status.
    """

    return SelectionProvenance(
        relation=relation,
        source=left.source or right.source,
        basis=left.basis if left.basis == right.basis else None,
        field_status=left.field_status if left.field_status == right.field_status else None,
    )
