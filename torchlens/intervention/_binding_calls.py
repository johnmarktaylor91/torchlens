"""Pure call and output probes for the bound-model rule lane."""

from __future__ import annotations

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
