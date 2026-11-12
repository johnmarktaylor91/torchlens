"""Provenance id of a facet registry snapshot, reused while the recipes are unchanged."""

from __future__ import annotations

import hashlib
import operator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .facets import FacetFunction

#: ``(publics, provenance id)`` of the last snapshot. ``FacetFunction`` is frozen and holds
#: only strings, string tuples and a bool, so the same public objects in the same order
#: (by identity) hash to the same id.
_LAST_PROVENANCE: list[tuple[tuple[FacetFunction, ...], str]] = []


def provenance_id(publics: tuple[FacetFunction, ...]) -> str:
    """Return the snapshot provenance id of ``publics``, reusing the last one when identical."""

    last = _LAST_PROVENANCE[0] if _LAST_PROVENANCE else None
    if (
        last is not None
        and len(last[0]) == len(publics)
        and all(map(operator.is_, last[0], publics))
    ):
        return last[1]
    digest = hashlib.sha256()
    for public in publics:
        digest.update(public.name.encode("utf-8"))
        digest.update(repr(public.class_names).encode("utf-8"))
        digest.update(repr(public.qualnames).encode("utf-8"))
        digest.update(str(public.has_predicate).encode("utf-8"))
        digest.update(public.target_scope.encode("utf-8"))
        digest.update(public.source.encode("utf-8"))
    provenance = digest.hexdigest()[:16]
    _LAST_PROVENANCE[:] = [(publics, provenance)]
    return provenance
