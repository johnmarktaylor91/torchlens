"""Facet unions at the write doors: ``head_a | head_b`` splits into its facet branches.

``hooks.expand_facet_hook_entries`` expands ONE facet selector (``tl.head``,
``tl.facet``, ``module & tl.head``) into slice-aware hook entries on its home
ops. A ``|`` of facet selectors builds a ``CompositeSelector``, which general
site resolution refuses (facets resolve only through the write doors and the
bare ``trace[selector]`` read). The attach path therefore splits a union first:
every facet branch becomes its own entry, so it resolves and writes exactly as
the single selector would, and any non-facet branches stay one ordinary entry.
``(module_a | module_b) & facet`` distributes to
``(module_a & facet) | (module_b & facet)`` before the split.

A composite that mixes a facet into anything else (``facet & tl.func(...)``, a
facet under ``~``) is not split and keeps the facet capability refusal.
"""

from __future__ import annotations

import functools
import operator
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from .._ir.selector_eval import selector_contains_kind
from .._selection._module_facet_fold import _fold_module_facet
from .._selection.selector_base import CompositeSelector
from .._selection.selectors import FacetSelector
from .hooks import NormalizedHookEntry

__all__: list[str] = []


def _union_branches(selector: Any) -> list[Any] | None:
    """Return the ``|`` branches of ``selector`` with module scopes distributed.

    ``None`` means ``selector`` is neither a facet, a union, nor a
    distributable ``(module | module) & facet``.
    """

    if isinstance(selector, FacetSelector):
        return [selector]
    if not isinstance(selector, CompositeSelector):
        return None
    if selector.operator == "and":
        return _distribute_module_scopes(selector.selectors)
    branches: list[Any] = []
    for child in selector.selectors:
        branches.extend(_union_branches(child) or [child])
    return branches


def _distribute_module_scopes(children: Sequence[Any]) -> list[Any] | None:
    """Distribute ``(module_a | module_b) & facet`` into scoped facet selectors."""

    if len(children) != 2:
        return None
    facets = [child for child in children if isinstance(child, FacetSelector)]
    scopes = [child for child in children if not isinstance(child, FacetSelector)]
    if len(facets) != 1 or len(scopes) != 1:
        return None
    modules = _module_alternatives(scopes[0])
    if modules is None:
        return None
    return [_fold_module_facet(module, facets[0]) for module in modules]


def _module_alternatives(selector: Any) -> list[Any] | None:
    """Return the module selectors of a ``|`` made only of ``tl.module(...)``."""

    if not isinstance(selector, CompositeSelector) or selector.operator != "or":
        return None
    modules = list(selector.selectors)
    if not all(getattr(module, "selector_kind", None) == "module" for module in modules):
        return None
    return modules


def split_facet_union(site_target: Any) -> tuple[tuple[FacetSelector, ...], Any] | None:
    """Split a union holding facet selectors into facet branches and a remainder.

    Parameters
    ----------
    site_target:
        A hook entry's site target.

    Returns
    -------
    tuple[tuple[FacetSelector, ...], Any] | None
        The distinct facet branches in first-appearance order and the ``|`` of
        the remaining non-facet branches (``None`` when there are none), or
        ``None`` when ``site_target`` is not such a union.
    """

    if not isinstance(site_target, CompositeSelector):
        return None
    branches = _union_branches(site_target)
    if branches is None:
        return None
    facets = tuple(dict.fromkeys(b for b in branches if isinstance(b, FacetSelector)))
    rest = [b for b in branches if not isinstance(b, FacetSelector)]
    if not facets or any(selector_contains_kind(b, "facet", unwrap=True) for b in rest):
        return None
    remainder = functools.reduce(operator.or_, rest) if rest else None
    return facets, remainder


def is_facet_union(site_target: Any) -> bool:
    """Return whether ``site_target`` is a union the write doors split into facets.

    Parameters
    ----------
    site_target:
        Selector-like site target.

    Returns
    -------
    bool
        Whether :func:`split_facet_union` splits it.
    """

    return split_facet_union(site_target) is not None


def split_facet_union_entries(
    entries: Sequence[NormalizedHookEntry],
) -> list[NormalizedHookEntry]:
    """Replace each facet-union entry by one entry per branch.

    Every facet branch keeps the entry's edit, helper and metadata, so
    ``expand_facet_hook_entries`` resolves and writes it exactly as it would
    the single facet selector; the non-facet branches stay one entry.

    Parameters
    ----------
    entries:
        Normalized hook entries.

    Returns
    -------
    list[NormalizedHookEntry]
        Entries with every facet union split.
    """

    split_entries: list[NormalizedHookEntry] = []
    for entry in entries:
        split = split_facet_union(entry.site_target)
        if split is None:
            split_entries.append(entry)
            continue
        facets, remainder = split
        targets = (*facets, remainder) if remainder is not None else facets
        split_entries.extend(replace(entry, site_target=target) for target in targets)
    return split_entries
