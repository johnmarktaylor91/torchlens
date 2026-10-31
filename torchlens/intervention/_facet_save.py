"""The capture-time save selector for ``trace(save=<facet selector>)``.

Save-only: ``intervene=``, ``bind`` and capture-time ``hooks=`` never reach
this module (they evaluate the facet selector itself, which refuses at
capture), so no facet edit can land on a whole home tensor.
"""

from __future__ import annotations

from typing import Any

from .selectors import BaseSelector, FacetSelector, WhereSelector, in_module, module


def _is_op_event(ctx: Any) -> bool:
    """Return whether a capture-time record context is an operation event."""

    return getattr(ctx, "kind", "op") == "op"


class _FacetSaveSelector(WhereSelector):
    """The predicate selector ``trace(save=<facet selector>)`` runs, spelled as the facet."""

    def __repr__(self) -> str:
        """Return the facet selector's own spelling, for the zero-match save warning."""

        return str(self.name_hint)


def _facet_save_predicate(selector: FacetSelector) -> BaseSelector:
    """Return the capture-time save selector for ``trace(save=<facet selector>)``.

    The selector keeps the ops a later ``trace[selector]`` read needs. For an
    SAE latent family that is its registered site (a string site is read as a
    module address); for a head or named facet it is every op inside a module
    whose class has a recipe declaring the facet, a superset of the facet's
    homes, because the recipes pick their home ops from the finished module
    record.

    Parameters
    ----------
    selector:
        Facet selector the caller passed as ``save=``.

    Returns
    -------
    BaseSelector
        Ordinary save selector (module selectors keep their post-capture
        retention path; the recipe-interior predicate counts its matches for
        the zero-match save warning).
    """

    from .._semantic._families import family_key

    family = family_key(selector.name)
    if family is not None:
        site = family[0].site
        home = site if isinstance(site, BaseSelector) else module(str(site))
        if selector.module_address is not None:
            home = home & in_module(selector.module_address)
        return home

    names = frozenset(("q", "k", "v") if selector.name is None else (selector.name,))
    from .._semantic.facets import _REGISTRY

    if not any(names & set(recipe.declared_facets) for recipe in _REGISTRY):
        # No recipe declares the facet: it is structural (``out``, an output
        # path), served by every op record, so every op is a possible home.
        return _FacetSaveSelector(_is_op_event, name_hint=repr(selector))
    declares: dict[tuple[int, str], bool] = {}

    def _declares(address: str) -> bool:
        """Return whether the live module at ``address`` has a recipe declaring the facet."""

        # Capture frames carry module addresses but no class names, so the
        # class is read from the live model of the active capture.
        from .._data_classes._runtime_handles import source_model_from_trace
        from .._semantic.facets import info
        from .._state import active_capture

        model = source_model_from_trace(active_capture()[0])
        key = (id(model), address)
        if key not in declares:
            try:
                class_name = "" if model is None else type(model.get_submodule(address)).__name__
            except (AttributeError, TypeError):
                class_name = ""
            declares[key] = bool(class_name) and bool(names & set(info(class_name)["facets"]))
        return declares[key]

    def _recipe_interior(ctx: Any) -> bool:
        """Return whether an op event runs inside a module whose recipe declares the facet."""

        if not _is_op_event(ctx):
            return False
        return any(
            (selector.module_address is None or frame.address == selector.module_address)
            and _declares(str(frame.address))
            for frame in getattr(ctx, "module_stack", ()) or ()
        )

    return _FacetSaveSelector(_recipe_interior, name_hint=repr(selector))
