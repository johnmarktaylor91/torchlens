"""Site-query refusals: the argument checks, the fan-out bound, and the no-match and multi-pass refusals.

Shared by ``resolver._resolve_sites``, ``resolver.find_sites`` and
``resolver.resolve_site_plan``; ``selection`` reads the multi-pass teaching
message. Imports only the selection error classes, so it adds no module to an
import cycle.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from .._pass_tokens import PASS_TOKEN_ALL, split_pass_token
from .._utils._display_errors import lookup_message
from .errors import AmbiguousSiteError, SiteResolutionError

if TYPE_CHECKING:
    from .resolver import Site, SiteBranch
    from .selectors import BaseSelector


def check_site_query_arguments(query: Any, *, strict: bool, max_fanout: int) -> None:
    """Refuse a fan-out bound that is ``None`` or below one, and a bare string in strict mode.

    Parameters
    ----------
    query:
        Selector input as the caller spelled it.
    strict:
        Whether non-portable query forms are refused.
    max_fanout:
        Maximum number of sites the caller allows.

    Raises
    ------
    SiteResolutionError
        For ``max_fanout=None`` or a bare string under ``strict``.
    AmbiguousSiteError
        For ``max_fanout`` below one.
    """

    if max_fanout is None:
        raise SiteResolutionError(
            "max_fanout=None is not supported; pass an explicit integer.",
            code="max_fanout_invalid",
        )
    if max_fanout < 1:
        raise AmbiguousSiteError("max_fanout must be at least 1.", code="max_fanout_invalid")
    if strict and isinstance(query, str):
        raise SiteResolutionError(
            "Bare strings are non-portable in strict mode; use tl.label(...), "
            "tl.func(...), or another typed selector.",
            code="bare_string_selector_strict",
        )


def refuse_excess_fanout(query: Any, matched: Sequence[Any], max_fanout: int) -> None:
    """Refuse a resolved site set larger than ``max_fanout``.

    Raises
    ------
    AmbiguousSiteError
        With ``fields["code"] == "site_selector_excess_fanout"``.
    """

    if len(matched) > max_fanout:
        raise AmbiguousSiteError(
            f"site {query!r} matched {len(matched)} sites, exceeding max_fanout={max_fanout}. "
            "Pass a larger max_fanout explicitly or use a narrower selector.",
            code="site_selector_excess_fanout",
        )


def multipass_bare_label_message(layer_label: str, pass_indices: Sequence[int]) -> str:
    """Return the teaching refusal message for a bare multi-pass layer label.

    Parameters
    ----------
    layer_label:
        Bare layer label the caller supplied.
    pass_indices:
        Pass indices recorded for the layer, in execution order.

    Returns
    -------
    str
        Message naming the layer, its pass count, and at most five
        pass-qualified display keys the caller can address instead.
    """

    return lookup_message(
        f"bare label {layer_label!r} is ambiguous: the layer ran {len(pass_indices)} passes; "
        "each pass is a distinct op with its own activation.",
        [f"{layer_label}:{index}" for index in pass_indices],
        f"Address one pass explicitly, '{layer_label}:last', or '{layer_label}:all'; "
        f"read the aggregate with trace.layers[{layer_label!r}].",
    )


def _refuse_bare_multipass_label(selector: BaseSelector, matched: Sequence[Site]) -> None:
    """Refuse an exact-label query that spans several passes of one layer.

    A label selector addresses ONE op; when its spelling is layer-wide on a
    multi-pass (recurrence-grouped) layer it matches every pass, and any
    single-op consumer would have to guess a pass — the exact silent
    wrong-pass corruption the replay engine refuses. Predicate selectors
    (``tl.func``, module selectors, ...) keep their fan-out semantics.

    Raises
    ------
    AmbiguousSiteError
        With ``fields["code"] == "multipass_bare_label_ambiguous"``.
    """

    selector_kind = getattr(selector, "selector_kind", None)
    if len(matched) <= 1 or selector_kind not in ("label", "contains"):
        return
    token = split_pass_token(str(getattr(selector, "selector_value", "")))
    if selector_kind == "label" and token is not None and token[1] == PASS_TOKEN_ALL:
        # ``label:all`` is the set spelling: naming every pass is its meaning.
        return
    layer_labels = {getattr(site, "layer_label", None) for site in matched}
    if len(layer_labels) != 1:
        return
    layer_label = next(iter(layer_labels))
    if not isinstance(layer_label, str):
        return
    if selector_kind == "contains" and getattr(selector, "selector_value", None) != layer_label:
        # A genuine substring pattern keeps its documented fan-out semantics;
        # only the exact bare layer label is the pass-ambiguous address.
        return
    pass_indices = sorted(int(getattr(site, "pass_number", 1) or 1) for site in matched)
    raise AmbiguousSiteError(
        multipass_bare_label_message(matched[0].short_layer_label, pass_indices),
        code="multipass_bare_label_ambiguous",
        layer_label=layer_label,
        pass_indices=tuple(pass_indices),
    )


def refuse_unmatched_plan(normalized: Any, resolved: Sequence[SiteBranch]) -> None:
    """Refuse a door plan that matched no site, and a bare multi-pass label on a forward branch.

    Raises
    ------
    SiteResolutionError
        With ``fields["code"] == "site_selector_no_match"`` when no branch matched.
    AmbiguousSiteError
        From :func:`_refuse_bare_multipass_label`.
    """

    if not any(branch.sites for branch in resolved):
        raise SiteResolutionError(
            f"selector {normalized!r} matched 0 sites. Use trace[selector] to inspect what a selector matches.",
            code="site_selector_no_match",
        )
    for branch in resolved:
        if branch.universe != "node":
            _refuse_bare_multipass_label(branch.selector, branch.sites)
