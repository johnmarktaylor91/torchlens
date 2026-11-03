"""Facet-snapshot slot: the engine asks for a trace's recipe snapshot without importing facets.

``_capture`` constructs every ``Trace`` with the active facet-recipe snapshot, which the
appliance layer (``torchlens._semantic.facets``) computes. The engine must not import
upward, so the snapshot function is REGISTERED here: ``_semantic.facets`` registers
``snapshot`` when it loads, and the public capture door registers a lazy provider so a
capture that passes plain recipe callables before anything imported the facets module
still gets a real snapshot. With nothing registered a recipe-free capture records
``None`` (readers fall back to the current registry); a capture WITH recipes refuses,
because dropping them silently would lose user input.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from .kernel import RegistryError

__tl_layer__ = "L1"
__all__ = ["facet_snapshot", "register_facet_snapshot_provider"]

FacetSnapshotProvider = Callable[[Sequence[Any] | None], Any]

_PROVIDER: FacetSnapshotProvider | None = None


def register_facet_snapshot_provider(provider: FacetSnapshotProvider) -> None:
    """Install ``provider`` as the function that builds a trace's facet snapshot.

    Parameters
    ----------
    provider:
        ``(extra_recipes) -> FacetRegistrySnapshot``; the last registration wins
        (the facets module's direct ``snapshot`` replaces the door's lazy provider).
    """

    global _PROVIDER
    _PROVIDER = provider


def facet_snapshot(extra_recipes: Sequence[Any] | None) -> Any:
    """Return the facet snapshot for a new trace, or ``None`` when none can be built.

    Parameters
    ----------
    extra_recipes:
        Per-trace additive recipes (``tl.trace(..., facets=[...])``), or ``None``.

    Returns
    -------
    Any
        The registered provider's snapshot; ``None`` when nothing is registered and
        there are no per-trace recipes to lose.

    Raises
    ------
    RegistryError
        ``facet_snapshot_provider_missing``: per-trace recipes were passed but no
        snapshot provider is registered, so they could not be recorded.
    """

    provider = _PROVIDER
    if provider is not None:
        return provider(extra_recipes)
    if not extra_recipes:
        return None
    raise RegistryError(
        "Per-trace facet recipes were passed but no facet snapshot provider is registered.",
        code="facet_snapshot_provider_missing",
        recipe_count=len(extra_recipes),
        remedy=(
            "Capture through torchlens.trace (its door registers the provider) or import "
            "torchlens.facets before passing recipes to an internal capture entry."
        ),
    )
