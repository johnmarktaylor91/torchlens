"""Pure candidate-string and grad-fn accessor helpers for selector evaluation."""

from __future__ import annotations

from typing import Any


def _module_candidate_string(candidate: Any) -> str:
    """Return a selector-compatible module candidate string.

    Parameters
    ----------
    candidate:
        Module candidate from an op, module call, or capture context.

    Returns
    -------
    str
        Address or pass-qualified address for :func:`module_address_matches`.
    """

    if isinstance(candidate, tuple) and candidate and isinstance(candidate[0], str):
        if len(candidate) > 1:
            return f"{candidate[0]}:{candidate[1]}"
        return candidate[0]
    return str(candidate)


def _grad_fn_type_matches(site: Any, requested: str) -> bool:
    """Return whether a grad_fn_handle type matches user spelling flexibly.

    Parameters
    ----------
    site:
        Candidate grad_fn_handle log.
    requested:
        Requested type string.

    Returns
    -------
    bool
        Whether class name or normalized type matches.
    """

    lowered = requested.lower()
    normalized = lowered.removesuffix("backward0").removesuffix("backward")
    candidates = {
        site.class_name.lower(),
        site.type.lower(),
        site.label.lower(),
    }
    return lowered in candidates or normalized in candidates


def _grad_fn_call_values(site: Any) -> tuple[Any, ...]:
    """Return GradFnCall values from dict or accessor-backed call storage.

    Parameters
    ----------
    site:
        Candidate grad_fn_handle log.

    Returns
    -------
    tuple[Any, ...]
        Recorded GradFnCall values.
    """

    calls = site.calls
    if isinstance(calls, dict):
        return tuple(calls.values())
    return tuple(calls._list)
