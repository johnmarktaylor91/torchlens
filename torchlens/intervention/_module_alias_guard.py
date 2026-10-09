"""Refuse a shared module's alias spelling in a selector, identically at every door.

A module object registered under two names (``self.enc = blk; self.dec =
self.enc``) is reported once by ``named_modules()``, under the first name.
TorchLens labels every call of it with that canonical name (``enc:1``,
``enc:2``) and records the other name in ``Module.all_addresses``. No hook can
tell which attribute a forward called the object through, so
``tl.module("dec")`` cannot select "the dec call": honoring it would also edit
the ``enc`` call, a number the user did not ask for (AUD-CODE 3.7d).

Every door that reads a module selector (``tl.trace`` and ``tl.record``
``intervene=`` / ``save=`` / ``halt=`` before the forward, post-hoc site
resolution, ``spec.bind``) therefore calls :func:`refuse_module_alias_spellings`
with its model's alias map, and the refusal is one typed error with one code
and one message wherever it is raised.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .errors import BindingPreflightError

#: Selector kinds whose value is a module address (bare or ``"addr:N"``).
_MODULE_ADDRESS_KINDS = frozenset({"module", "in_module"})


def bare_module_address(address: str) -> str:
    """Strip a trailing ``:pass`` qualifier from a module address.

    Parameters
    ----------
    address:
        Module address, bare (``"enc"``) or pass-qualified (``"enc:2"``).

    Returns
    -------
    str
        The address without its pass qualifier.
    """

    base, sep, tail = address.rpartition(":")
    if sep and tail.isdigit():
        return base
    return address


def module_address_terms(*sources: Any) -> list[str]:
    """Collect every module address a selector source names.

    Parameters
    ----------
    *sources:
        Selectors, intervention specs, hook entries, mappings keyed by
        selectors, or sequences of these. ``None``, strings and callables name
        no module selector.

    Returns
    -------
    list[str]
        ``tl.module`` / ``tl.in_module`` values and ``tl.site`` module paths,
        including those under ``&`` / ``|`` / ``~`` and retroactive wrappers.
    """

    terms: list[str] = []
    stack: list[Any] = list(sources)
    while stack:
        source = stack.pop()
        terms.extend(_direct_terms(source))
        stack.extend(_children(source))
    return terms


def _direct_terms(source: Any) -> list[str]:
    """Return the module addresses one selector-like node names itself."""

    kind = getattr(source, "selector_kind", None)
    if kind in _MODULE_ADDRESS_KINDS and isinstance(source.selector_value, str):
        return [source.selector_value]
    if kind == "site" and getattr(source, "module_path", None):
        return [str(source.module_path)]
    return []


def _children(source: Any) -> list[Any]:
    """Return the nested selector sources of one node (composites, specs, plans)."""

    from .selectors import (
        CompositeSelector,
        FollowedBySelector,
        NotSelector,
        PrecededBySelector,
    )
    from .spec import InterventionSpec

    if source is None or isinstance(source, str):
        return []
    if isinstance(source, CompositeSelector):
        nested: tuple[Any, ...] = tuple(source.selectors)
    elif isinstance(source, NotSelector):
        nested = (source.selector,)
    elif isinstance(source, (FollowedBySelector, PrecededBySelector)):
        nested = (source.inner,)
    elif isinstance(source, InterventionSpec):
        nested = tuple(rule.where for rule in source.rules)
    elif isinstance(source, (Mapping, list, tuple)):
        nested = tuple(source)
    else:
        site_target = getattr(source, "site_target", None)
        nested = () if site_target is None else (site_target,)
    return list(nested)


def model_module_aliases(model: Any) -> dict[str, str]:
    """Map every alias address of a live model's shared modules to its canonical address.

    Parameters
    ----------
    model:
        Root ``nn.Module``.

    Returns
    -------
    dict[str, str]
        Alias address -> canonical ``named_modules()`` address (empty when no
        module is registered twice). The walk is the one model preparation
        uses to fill ``Module.all_addresses``, so every door sees one map.
    """

    from ..backends.torch.model_prep import shared_module_addresses

    return {
        alias: addresses[0]
        for addresses in shared_module_addresses(model).values()
        for alias in addresses[1:]
    }


def trace_module_aliases(trace: Any) -> dict[str, str]:
    """Map every alias address recorded on a finalized Trace to its canonical address.

    Parameters
    ----------
    trace:
        Finalized Trace whose ``modules`` carry ``all_addresses``.

    Returns
    -------
    dict[str, str]
        Alias address -> canonical address (empty when no module is shared or
        the Trace carries no module accessor).
    """

    try:
        modules = list(trace.modules)
    except (AttributeError, TypeError):
        # Module-less products (cleaned-up or non-torch traces) record no aliases.
        return {}
    return {
        alias: str(module.address)
        for module in modules
        for alias in getattr(module, "all_addresses", ()) or ()
        if alias != module.address
    }


def refuse_module_alias_spellings(aliases: Mapping[str, str], *sources: Any) -> None:
    """Refuse when a selector source spells a shared module by an alias address.

    Parameters
    ----------
    aliases:
        Alias address -> canonical address for the door's model.
    *sources:
        Selector sources, as :func:`module_address_terms` accepts.

    Raises
    ------
    BindingPreflightError
        ``bind_static_anchor_unresolved`` naming each alias spelling and its
        canonical ``named_modules()`` name, which fires at every call site.
    """

    if not aliases:
        return
    spelled = [
        (term, aliases[bare_module_address(term)])
        for term in dict.fromkeys(module_address_terms(*sources))
        if bare_module_address(term) in aliases
    ]
    if not spelled:
        return
    canonical = spelled[0][1]
    raise BindingPreflightError(
        "; ".join(
            f"{term!r} is an alias of {target!r} (the same module object registered under "
            f"both names; named_modules() reports only {target!r}, and a rule anchored "
            "there fires at EVERY call site of the shared module)"
            for term, target in spelled
        ),
        code="bind_static_anchor_unresolved",
        remedy=f"select the shared module by its named_modules() name ({canonical!r}), "
        "which fires at every call site of it, or add a pass label to select one call "
        f"({canonical + ':2'!r} is its second call)",
    )


def refuse_model_alias_spellings(model: Any, *sources: Any) -> None:
    """Refuse alias spellings against a live model, walking it only when needed.

    Parameters
    ----------
    model:
        Root ``nn.Module`` the door runs.
    *sources:
        Selector sources, as :func:`module_address_terms` accepts.

    Raises
    ------
    BindingPreflightError
        As :func:`refuse_module_alias_spellings`.
    """

    if module_address_terms(*sources):
        refuse_module_alias_spellings(model_module_aliases(model), *sources)


def refuse_trace_alias_spellings(trace: Any, *sources: Any) -> None:
    """Refuse alias spellings against a finalized Trace's recorded alias map.

    Parameters
    ----------
    trace:
        Finalized Trace the post-hoc door resolves against.
    *sources:
        Selector sources, as :func:`module_address_terms` accepts.

    Raises
    ------
    BindingPreflightError
        As :func:`refuse_module_alias_spellings`.
    """

    if module_address_terms(*sources):
        refuse_module_alias_spellings(trace_module_aliases(trace), *sources)
