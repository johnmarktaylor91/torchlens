"""Per-trace facet recipes (``trace(facets=...)``) on the preview backends.

Facet recipes are backend-neutral: a recipe reads an operation record and
returns facet values, and the snapshot that carries recipes on a trace is the
same registry snapshot the torch door builds. Every preview backend therefore
attaches recipes through this one helper instead of refusing them, and the
``facet_recipes`` capability record owns the public ``facets`` option.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

from .._registry._facet_snapshot import facet_snapshot
from ..errors import BackendUnsupportedError
from ._options import is_missing

__tl_layer__ = "L2"

__all__ = ["attach_facet_recipes", "pop_facet_recipes", "snapshot_facet_recipes"]

_REFUSAL_CODE = "trace_option_unsupported"


def snapshot_facet_recipes(facet_recipes: Any, *, backend_name: str) -> Any:
    """Validate ``trace(facets=...)`` recipes and return the per-trace facet snapshot.

    Parameters
    ----------
    facet_recipes:
        One recipe callable or an iterable of them.
    backend_name:
        The backend's public name, for the refusal text.

    Returns
    -------
    Any
        The facet registry snapshot carrying the registered recipes plus these.

    Raises
    ------
    BackendUnsupportedError
        ``trace_option_unsupported`` when the value is not an iterable of callables.
    """

    if callable(facet_recipes):
        facet_recipes = (facet_recipes,)
    try:
        recipe_tuple = tuple(facet_recipes)
    except TypeError as exc:
        raise BackendUnsupportedError(
            f"{backend_name} backend trace(facets=...) requires an iterable of recipe callables.",
            code=_REFUSAL_CODE,
            option="facets",
        ) from exc
    if not all(callable(recipe) for recipe in recipe_tuple):
        raise BackendUnsupportedError(
            f"{backend_name} backend trace(facets=...) requires recipe callables.",
            code=_REFUSAL_CODE,
            option="facets",
        )
    return facet_snapshot(recipe_tuple)


def pop_facet_recipes(kwargs: MutableMapping[str, Any]) -> Any:
    """Pop ``facets`` from a backend's extra kwargs; ``None`` when absent or MISSING."""

    value = kwargs.pop("facets", None)
    return None if value is None or is_missing(value) else value


def attach_facet_recipes(trace: Any, facet_recipes: Any, *, backend_name: str) -> None:
    """Snapshot ``trace(facets=...)`` recipes onto the finished trace, when given."""

    if facet_recipes is not None:
        trace.facet_functions = snapshot_facet_recipes(facet_recipes, backend_name=backend_name)
