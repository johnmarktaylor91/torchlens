"""Multi-match warning wording for the selector resolver.

Imports nothing at runtime, so it adds no module to an import cycle.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .resolver import SelectorInput


def _multi_match_message(query: SelectorInput, records: Sequence[Any]) -> str:
    """Return the multi-match warning text for a resolved site set.

    A set holding a synthetic model-output alias (``output_N``) together with
    the op it aliases does not fan out: a value edit applied at both sites
    compounds on the one returned value, so the warning says so.

    Parameters
    ----------
    query:
        Selector input as the caller spelled it.
    records:
        The record each resolved site describes (``resolver._site_record``), in
        execution order.

    Returns
    -------
    str
        Warning message.
    """

    matched_labels = {getattr(site, "layer_label", None) for site in records}
    pairs = [
        (str(getattr(site, "layer_label", None)), str(parent))
        for site in records
        if getattr(site, "is_output", False)
        for parent in (getattr(site, "parents", ()) or ())
        if parent in matched_labels
    ]
    if not pairs:
        return f"selector {query!r} matched {len(records)} sites and will fan out."
    named = ", ".join(f"{alias!r} aliases {producer!r}" for alias, producer in pairs)
    return (
        f"selector {query!r} matched {len(records)} sites, including a model-output alias "
        f"and the op it aliases ({named}); a value edit applied at both compounds on the "
        "same returned value instead of reaching independent sites. Narrow the selector to "
        "the producing op."
    )
