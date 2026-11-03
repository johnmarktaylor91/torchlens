"""Pure call and output probes for the bound-model rule lane."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any


def _is_in_place_call(func_name: str, kwargs: dict[str, Any]) -> bool:
    """Whether one intercepted torch call mutates its input storage in place."""

    if kwargs.get("inplace") is True:
        return True
    return func_name.endswith("_") and not func_name.endswith("__")


def _has_tensor(value: Any, tensor_type: type) -> bool:
    """Cheap check that an output holds at least one top-level tensor."""

    if isinstance(value, tensor_type):
        return True
    if isinstance(value, (list, tuple)):
        return any(isinstance(item, tensor_type) for item in value)
    if isinstance(value, dict):
        return any(isinstance(item, tensor_type) for item in value.values())
    return False


def _alias_disclosure(aliases: dict[str, str]) -> dict[str, tuple[str, ...]]:
    """Invert the alias map into canonical address -> alias spellings."""

    disclosure: dict[str, list[str]] = {}
    for alias, canonical in aliases.items():
        disclosure.setdefault(canonical, []).append(alias)
    return {canonical: tuple(names) for canonical, names in disclosure.items()}


def _bound_leaves(
    output: Any, leaves: Iterable[tuple[Any, tuple[Any, ...], Any]]
) -> Iterator[tuple[Any, tuple[Any, ...]]]:
    """Keep the walked output leaves the bound boundary door can address by container path.

    ``leaves`` is the exhaustive-mode walk of ``output`` (tensor, container
    path, container spec). Its fallback for a container it cannot address (a
    distribution object, an arbitrary class) still finds tensors inside it,
    with the empty path. No rebuild is possible there, so such leaves are
    skipped: the rule records zero fire and the output stays whole, matching
    the capture door, which sees no tensor site in such an output. A tensor
    output is its own single leaf at the empty path and is kept.
    """

    for out, container_path, _spec in leaves:
        if container_path != () or out is output:
            yield out, container_path
