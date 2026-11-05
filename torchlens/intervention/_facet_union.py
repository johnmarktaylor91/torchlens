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
facet under ``~``) is not split and keeps the facet capability refusal: it stays
in the non-facet remainder, whose site validation refuses it before any hook
attaches.

Selectors are recognised by their ``selector_kind`` (``"facet"``, ``"and"``,
``"or"``, ``"module"``) rather than by class, so this module imports nothing from
the selector or hook modules: ``_trace_hooks`` and ``_trace_edits`` import it,
and an import back would join their import cycle (``scripts/checks/layering.py``).
"""

from __future__ import annotations

import functools
import operator
from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from .._selection._module_facet_fold import _fold_module_facet

if TYPE_CHECKING:
    from .hooks import NormalizedHookEntry


def _kind(selector: Any) -> Any:
    """Return a selector's ``selector_kind`` (``None`` for a non-selector)."""

    return getattr(selector, "selector_kind", None)


def _composite_children(selector: Any, operator_name: str) -> tuple[Any, ...] | None:
    """Return the children of an ``operator_name`` composite, else ``None``."""

    children = getattr(selector, "selectors", None)
    if _kind(selector) != operator_name or not isinstance(children, tuple):
        return None
    return children


def _union_branches(selector: Any) -> list[Any] | None:
    """Return the ``|`` branches of ``selector`` with module scopes distributed.

    ``None`` means ``selector`` is neither a facet, a union, nor a
    distributable ``(module | module) & facet``.
    """

    if _kind(selector) == "facet":
        return [selector]
    conjuncts = _composite_children(selector, "and")
    if conjuncts is not None:
        return _distribute_module_scopes(conjuncts)
    alternatives = _composite_children(selector, "or")
    if alternatives is None:
        return None
    branches: list[Any] = []
    for child in alternatives:
        branches.extend(_union_branches(child) or [child])
    return branches


def _distribute_module_scopes(children: Sequence[Any]) -> list[Any] | None:
    """Distribute ``(module_a | module_b) & facet`` into scoped facet selectors."""

    if len(children) != 2:
        return None
    facets = [child for child in children if _kind(child) == "facet"]
    scopes = [child for child in children if _kind(child) != "facet"]
    if len(facets) != 1 or len(scopes) != 1:
        return None
    modules = _composite_children(scopes[0], "or")
    if modules is None or not all(_kind(module) == "module" for module in modules):
        return None
    return [_fold_module_facet(module, facets[0]) for module in modules]


def split_facet_union(site_target: Any) -> tuple[tuple[Any, ...], Any] | None:
    """Split a union holding facet selectors into facet branches and a remainder.

    Parameters
    ----------
    site_target:
        A hook entry's site target.

    Returns
    -------
    tuple[tuple[Any, ...], Any] | None
        The distinct facet branches in first-appearance order and the ``|`` of
        the remaining non-facet branches (``None`` when there are none), or
        ``None`` when ``site_target`` is not a composite holding a facet branch.
    """

    if _composite_children(site_target, "and") is None and (
        _composite_children(site_target, "or") is None
    ):
        return None
    branches = _union_branches(site_target)
    if branches is None:
        return None
    facets = tuple(dict.fromkeys(b for b in branches if _kind(b) == "facet"))
    if not facets:
        return None
    rest = [b for b in branches if _kind(b) != "facet"]
    remainder = functools.reduce(operator.or_, rest) if rest else None
    return facets, remainder


def is_facet_union(site_target: Any) -> bool:
    """Return whether ``site_target`` is a union made only of facet branches.

    The set door writes one value into every branch, so a union that also
    holds non-facet sites keeps the single-site refusal there.

    Parameters
    ----------
    site_target:
        Selector-like site target.

    Returns
    -------
    bool
        Whether :func:`split_facet_union` splits it into facets alone.
    """

    split = split_facet_union(site_target)
    return split is not None and split[1] is None


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
