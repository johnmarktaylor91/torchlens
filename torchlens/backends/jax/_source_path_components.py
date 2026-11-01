"""Source-path component normalization for JAX equation captures.

Extracted from ``jaxpr`` so the interpreter module stays under the file cap.
"""

from __future__ import annotations

import re

_ITERATION_COMPONENT_RE = re.compile(
    r"^(?P<prefix>(?:scan|while)[_\-:]?)?(?:iter|iteration)(?P<sep>[:=_-])\d+$"
)
_BRACKETED_ITERATION_RE = re.compile(r"(?P<prefix>\b(?:scan|while))\[(?:iter=)?\d+\]")


def _normalize_jax_source_path_component(component: str, *, is_leaf: bool) -> str | None:
    """Normalize one source-path component.

    Parameters
    ----------
    component
        Source-path component.
    is_leaf
        Whether this component names the captured equation itself.

    Returns
    -------
    str | None
        Normalized component, or ``None`` when the component is only an
        iteration marker.
    """

    iteration_match = _ITERATION_COMPONENT_RE.match(component)
    if iteration_match is not None:
        return None
    without_bracketed_iteration = _BRACKETED_ITERATION_RE.sub(
        lambda match: match.group("prefix"),
        component,
    )
    if is_leaf:
        _ordinal, separator, primitive_name = without_bracketed_iteration.partition(":")
        if separator and _ordinal.isdigit() and primitive_name:
            return primitive_name
    return without_bracketed_iteration
