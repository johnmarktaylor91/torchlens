"""Files a trace save writes beside its record tables: the intervention recipe and replay recipes.

``bundle.save`` calls :func:`write_trace_sidecars` where it wrote the intervention
recipe alone, so the size-capped save body did not grow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ._replay_recipe_codec import write_replay_recipes
from .format3.payloads import PayloadStore
from .format3.recipe import write_trace_recipe

__all__ = ("write_trace_sidecars",)


def write_trace_sidecars(
    root: Path,
    trace: Any,
    *,
    store: PayloadStore,
    save_level: str,
    backend: str,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Write the intervention recipe and, when it applies, the replay recipe section.

    Parameters
    ----------
    root:
        Staged trace artifact directory.
    trace:
        The trace being saved.
    store:
        The artifact's payload store (call before it closes).
    save_level:
        Resolved save level.
    backend:
        The trace's backend name.

    Returns
    -------
    tuple[dict[str, Any] | None, list[dict[str, Any]]]
        The manifest ``intervention`` object (``None`` without a staged rule)
        and the intervention recipe's disclosures, as ``write_trace_recipe``
        returns them.
    """

    entry = write_trace_recipe(root, trace, store=store, save_level=save_level)
    write_replay_recipes(root, trace, backend=backend, save_level=save_level)
    return entry
