"""Bundle site lookup: ``at``, the cached supergraph and structural consistency."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

from ..intervention._topology.topology import build_supergraph

if TYPE_CHECKING:
    from ..intervention._topology.topology import Supergraph
    from . import Bundle


def bundle_at(self: Bundle, site: Any) -> Any:
    """Return the cross-member Super view for ``site`` (gate attached)."""

    view = self._label_view(site) if isinstance(site, str) else self._site_view(site)
    names = list(getattr(view, "members", {}))
    pairs = [(left, right) for index, left in enumerate(names) for right in names[index + 1 :]]
    view._compare_gate = functools.partial(self._require_comparable, "at.compare", pairs)
    return view


def ensure_supergraph(self: Bundle) -> Supergraph:
    """Build and cache the supergraph lazily."""

    if self._supergraph is None:
        self._supergraph = build_supergraph(list(self._members.values()), list(self._members))
    return self._supergraph


def is_structurally_consistent(self: Bundle) -> bool:
    """Whether every member has an equal ``graph_shape_hash`` value."""

    hashes = {member._provenance_fingerprint for member in self._members.values()}
    return len(hashes) == 1
