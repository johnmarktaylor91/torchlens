"""Container walking for parent-tensor argument positions (args, kwargs and nested containers)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, cast


def _is_supported_parent_arg_container(value: object) -> bool:
    """Return whether ``value`` participates in parent arg-position mapping.

    Parameters
    ----------
    value:
        Candidate nested argument value.

    Returns
    -------
    bool
        True when ``value`` is a supported list/tuple or mapping container.
    """

    return isinstance(value, (Mapping, tuple, list))


def _iter_arg_container_items(
    container_kind: str | object,
    container: object,
) -> Iterable[tuple[Any, Any]]:
    """Yield items from an args/kwargs structure or nested supported container.

    Parameters
    ----------
    container_kind:
        Either ``"args"``, ``"kwargs"``, or the nested container object itself.
    container:
        Args tuple, kwargs mapping, or a nested supported container.

    Returns
    -------
    Iterable[tuple[Any, Any]]
        Position keys paired with container values.

    Raises
    ------
    TypeError
        If ``container`` is not a supported args/kwargs or nested container shape.
    """

    if container_kind == "args":
        return enumerate(cast(tuple[Any, ...], container))
    if container_kind == "kwargs":
        return cast(dict[Any, Any], container).items()
    if isinstance(container, Mapping):
        return container.items()
    if isinstance(container, (list, tuple)):
        return enumerate(container)
    raise TypeError(f"Unsupported parent-arg container: {type(container)!r}")
