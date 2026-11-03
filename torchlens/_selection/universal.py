"""The universal selector ``tl.all_sites()``, the named form of ``tl.label_matches(".*")``.

Split out of ``selectors`` (which sits at its file cap) when the public name was
added; nothing in ``selectors`` depends on it.
"""

from __future__ import annotations

from dataclasses import dataclass

from .selectors import RegexSelector

#: Label pattern of the universal selector: ``re.search`` of ``.*`` matches every label.
_ALL_SITES_PATTERN = ".*"


@dataclass(frozen=True, repr=False)
class _AllSitesSelector(RegexSelector):
    """Universal selector: matches every site.

    The named form of ``tl.label_matches(".*")``. It stores the same kind
    and payload (``regex``, ``".*"``), so it resolves, matches at capture
    time, binds and persists exactly like that spelling; only its name and
    repr differ. A saved spec reloads as ``tl.label_matches(".*")``, which
    selects the same sites.
    """

    def __init__(self) -> None:
        """Create the universal selector."""

        super().__init__(_ALL_SITES_PATTERN)

    def __repr__(self) -> str:
        """Return the constructor spelling ``tl.all_sites()``."""

        return "tl.all_sites()"


def all_sites() -> _AllSitesSelector:
    """Create the universal selector, which matches every site.

    ``trace[tl.all_sites()]`` selects every op of the trace as a whole
    site, the same result as ``trace[tl.label_matches(".*")]``. It composes
    with ``&``, ``|``, ``-`` and ``~`` like any selector; ``tl.all_sites() -
    region`` is the complement of a region in the whole trace, and
    ``tl.neighborhood`` / ``tl.between`` / ``trace.subgraph`` read that
    difference as the sites outside ``region``. Resolution refuses typed
    (``non_tensor_site`` / ``no_index_space``) when some op has no output
    index space, rather than dropping it. DOCUMENTED-UNSTABLE spelling.

    Returns
    -------
    _AllSitesSelector
        Immutable selector.
    """

    return _AllSitesSelector()
