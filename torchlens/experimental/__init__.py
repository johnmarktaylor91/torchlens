"""Experimental TorchLens APIs with unstable naming and behavior."""

from __future__ import annotations

import re
from typing import Any

from torch import nn

_ATTRIBUTE_PART_RE = re.compile(r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?P<indexes>(?:\[[0-9]+\])*)")


def attribute_walk(model: nn.Module, address: str) -> Any:
    """Resolve a dotted/indexed attribute-walk address on a module.

    Parameters
    ----------
    model:
        Root PyTorch module.
    address:
        Address such as ``"transformer.h[5].mlp.output"``.

    Returns
    -------
    Any
        Object reached by walking attributes and integer indexes.

    Raises
    ------
    AttributeError
        If any attribute component is missing.
    IndexError
        If any index component is invalid.
    ValueError
        If the address syntax is malformed.
    """

    current: Any = model
    if not address:
        return current
    for part in address.split("."):
        match = _ATTRIBUTE_PART_RE.fullmatch(part)
        if match is None:
            raise ValueError(f"Malformed attribute-walk segment: {part!r}")
        name = match.group("name")
        if not hasattr(current, name):
            raise AttributeError(f"{type(current).__name__!s} has no attribute {name!r}")
        current = getattr(current, name)
        indexes = re.findall(r"\[([0-9]+)\]", match.group("indexes"))
        for index_text in indexes:
            current = current[int(index_text)]
    return current


__all__ = ["attribute_walk", "dagua", "node_styles"]
