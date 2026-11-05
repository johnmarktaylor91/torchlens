"""Facet writes: which records serve facet views, and the facet slice an edit is writing.

``hooks`` resolves ``tl.facet`` and ``tl.head`` selectors through ``_iter_facet_records``
and ``_scoped_facet_view`` and wraps each write with ``_facet_fire_scope``;
``helpers.patch_from`` reads the active slice (``_active_facet_fire``) so a donor
value is sliced the same way. Nothing here imports the selector or facet modules at
run time, so this module stays out of the package's import cycles.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from .._errors import InvalidArgumentError, PayloadUnavailableError
from ._hooks_support import _multi_call_addresses
from .errors import SiteResolutionError

if TYPE_CHECKING:
    from .._selection.selectors import FacetSelector


def _scoped_facet_view(record: Any, selector: FacetSelector) -> Any:
    """Return ``record``'s facet view when ``selector`` may use it, else ``None``.

    A module record that ran more than once is skipped (its view raises
    ``module_call_ambiguous``; the ops of its calls still serve), and refused
    with ``facet_selector_module_multi_call`` when the selector is scoped to it.
    """

    record_address = getattr(record, "address", None)
    if selector.module_address is not None and record_address != selector.module_address:
        return None
    call_addresses = _multi_call_addresses(record)
    if call_addresses is not None:
        if selector.module_address is not None:
            _refuse_multi_call_module(selector, str(record_address), call_addresses)
        return None
    try:
        return getattr(record, "facets", None)
    except PayloadUnavailableError:
        # An op whose payload was not saved serves no facet value; skipping it
        # lets a selectively saved trace resolve the facets it did keep.
        return None


def _refuse_multi_call_module(
    selector: FacetSelector, address: str, call_addresses: tuple[str, ...]
) -> None:
    """Refuse a facet selector scoped to a module that ran more than once.

    A facet write installs its hook on the home op's layer, which fires on
    every call of the module, so no call-qualified selector can edit one call;
    each call's facets are read at its call-qualified address.
    """

    raise InvalidArgumentError(
        f"{selector!r} is scoped to module {address!r}, which ran {len(call_addresses)} "
        "times; a facet write there would fire on every call",
        code="facet_selector_module_multi_call",
        remedy=f"read one call's facets at trace[{call_addresses[0]!r}].facets (calls: "
        f"{', '.join(call_addresses)}); write through a selector scoped to a module that "
        "runs once",
        address=address,
        call_addresses=call_addresses,
    )


def _iter_facet_records(log: Any) -> Iterator[Any]:
    """Yield records that can own semantic facet views.

    Parameters
    ----------
    log:
        Trace to inspect.

    Yields
    ------
    Any
        Module records followed by op records.
    """

    modules = getattr(log, "modules", ())
    try:
        yield from modules
    except TypeError:
        pass
    yield from log._layer_list


def _facet_home_label(spec: Any) -> str:
    """Return the live hook label for a facet home op.

    Parameters
    ----------
    spec:
        FacetSpec-like object.

    Returns
    -------
    str
        Home op layer label.
    """

    home = getattr(spec, "home", None)
    label_value = (
        getattr(home, "_layer_label_raw", None)
        or getattr(home, "_label_raw", None)
        or getattr(home, "layer_label", None)
        or getattr(home, "label", None)
    )
    if label_value is None:
        label_value = getattr(spec, "home_label", None)
    if label_value is None:
        raise SiteResolutionError("Facet selector resolved to a home without a layer label.")
    return str(label_value)


def _check_facet_write_conflicts(
    claims: Sequence[tuple[str, torch.Tensor, str]],
    *,
    home_label: str,
    mask: torch.Tensor,
) -> None:
    """Raise when a facet write overlaps an existing same-home claim.

    Parameters
    ----------
    claims:
        Existing write claims.
    home_label:
        Home label for the candidate write.
    mask:
        Candidate boolean write mask.

    Returns
    -------
    None
        Raises on overlapping writes.
    """

    for claimed_home, claimed_mask, claimed_name in claims:
        if claimed_home != home_label:
            continue
        if tuple(claimed_mask.shape) != tuple(mask.shape):
            continue
        if bool(torch.logical_and(claimed_mask.to(mask.device), mask).any()):
            raise SiteResolutionError(
                f"Facet intervention conflict on home {home_label!r}: write overlaps "
                f"existing facet {claimed_name!r}."
            )


@dataclass(frozen=True, slots=True)
class _FacetFire:
    """The facet slice a facet scatter wrapper is editing right now.

    Set for the duration of the wrapped user hook so a donor-reading edit
    (``patch_from``) can read the SAME facet view of its donor value instead
    of handing back the whole home-op output.
    """

    spec: Any
    selector: FacetSelector


_ACTIVE_FACET_FIRE: ContextVar[_FacetFire | None] = ContextVar(
    "torchlens_active_facet_fire", default=None
)


def _active_facet_fire() -> _FacetFire | None:
    """Return the facet slice being edited by the enclosing scatter wrapper, if any."""

    return _ACTIVE_FACET_FIRE.get()


@contextmanager
def _facet_fire_scope(spec: Any, selector: FacetSelector) -> Iterator[None]:
    """Make ``(spec, selector)`` the active facet slice while a wrapped user hook runs."""

    token = _ACTIVE_FACET_FIRE.set(_FacetFire(spec=spec, selector=selector))
    try:
        yield
    finally:
        _ACTIVE_FACET_FIRE.reset(token)
