"""Site-signal classification: which pass a selector site lives in."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .spec import SiteSignal

#: Selector kinds whose sites live in the backward pass: the gradient wrapper,
#: the autograd-node selectors and the backward-pass qualifier.
_BACKWARD_SELECTOR_KINDS: frozenset[str] = frozenset(
    {
        "grad",
        "grad_fn",
        "grad_fn_label",
        "grad_kind",
        "without_op",
        "intervening",
        "backward_pass",
    }
)


def _selector_children(site: Any) -> tuple[Any, ...]:
    """Return the operand selectors of a composite or negated selector."""

    kind = getattr(site, "selector_kind", None)
    if kind in ("and", "or"):
        children = getattr(site, "selectors", None)
        if children is None:
            children = getattr(site, "selector_value", ())
        return tuple(children or ())
    if kind == "not":
        child = getattr(site, "selector", None)
        if child is None:
            child = getattr(site, "selector_value", None)
        return () if child is None else (child,)
    return ()


def _site_passes(site: Any) -> frozenset[str]:
    """Return the passes (``"forward"``/``"backward"``) a site addresses.

    A union collects its operands' passes; an intersection or negation is in
    the backward pass when any operand is (mixed intersections already refuse
    at construction). Strings, records and every other selector address the
    forward pass.
    """

    kind = getattr(site, "selector_kind", None)
    if kind in _BACKWARD_SELECTOR_KINDS:
        return frozenset({"backward"})
    children = _selector_children(site)
    if not children:
        return frozenset({"forward"})
    passes = frozenset().union(*(_site_passes(child) for child in children))
    if kind == "or":
        return passes
    return frozenset({"backward"}) if "backward" in passes else frozenset({"forward"})


def site_signal(site: Any) -> SiteSignal:
    """Return the pass a site addresses, derived from the site alone.

    Parameters
    ----------
    site:
        Label string, record, selector or frozen target spec.

    Returns
    -------
    SiteSignal
        ``"backward"`` for ``tl.grad(...)`` and autograd-node sites,
        ``"forward"`` for activation sites, ``"both"`` for a union of the two.
    """

    passes = _site_passes(site)
    if passes == {"forward", "backward"}:
        return "both"
    return "backward" if "backward" in passes else "forward"


def _contains_kind(site: Any, kind: str) -> bool:
    """Whether a selector tree contains a node of ``kind``."""

    if getattr(site, "selector_kind", None) == kind:
        return True
    return any(_contains_kind(child, kind) for child in _selector_children(site))


def is_gradient_site(site: Any) -> bool:
    """Whether a site (selector or frozen target) contains a ``tl.grad`` address."""

    return _contains_kind(site, "grad")


def addresses_autograd_nodes(site: Any) -> bool:
    """Whether a site addresses autograd nodes (grad_fn records) only.

    Node selectors (``tl.grad_fn``, ``tl.grad_output()``, a lone
    ``tl.in_backward_pass(n)``, ...) resolve against grad_fn records; a
    ``tl.grad(...)`` site is a gradient of a forward op and is not one.
    """

    return site_signal(site) == "backward" and not _contains_kind(site, "grad")
