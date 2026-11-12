"""Private per-backend replay adapters (see :mod:`._contract`).

``get_adapter(name)`` imports ``_replay_adapters.<name>`` on first use. The
set of names is closed; an adapter module keeps its framework imports inside
its functions, so looking one up never imports the framework.
"""

from __future__ import annotations

import importlib
from types import ModuleType

from ._contract import (
    RECIPE_FORMAT_VERSION,
    InputOccurrence,
    OpRecipe,
    ReplayAdapter,
    ReplayRecipeError,
    decode_recipe,
    encode_recipe,
    validate_recipe,
)

__all__ = [
    "ADAPTER_BACKENDS",
    "RECIPE_FORMAT_VERSION",
    "InputOccurrence",
    "OpRecipe",
    "ReplayAdapter",
    "ReplayAdapterUnavailable",
    "ReplayRecipeError",
    "decode_recipe",
    "encode_recipe",
    "get_adapter",
    "has_adapter",
    "validate_recipe",
]

ADAPTER_BACKENDS: frozenset[str] = frozenset({"jax", "tf", "mlx", "paddle", "tinygrad"})
"""Backend names that may have a replay adapter module. Torch replays
through its own engine and has none."""

_ADAPTER_FUNCTIONS = (
    "can_replay",
    "forward",
    "vjp",
    "recipe",
    "bind_recipes",
    "zeros_like",
    "add",
    "to_host",
    "close",
)


class ReplayAdapterUnavailable(LookupError):
    """No replay adapter exists for this backend name, or its module does not
    define every contract function."""


def has_adapter(name: str) -> bool:
    """Return whether ``name`` has an importable, complete adapter module."""

    try:
        get_adapter(name)
    except ReplayAdapterUnavailable:
        return False
    return True


def get_adapter(name: str) -> ModuleType:
    """Return the adapter module for backend ``name``.

    Raises
    ------
    ReplayAdapterUnavailable
        When ``name`` is not in :data:`ADAPTER_BACKENDS`, its module is not
        present yet, or the module lacks a contract function.
    """

    if name not in ADAPTER_BACKENDS:
        raise ReplayAdapterUnavailable(f"no replay adapter for backend {name!r}")
    module_name = f"{__name__}.{name}"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
        raise ReplayAdapterUnavailable(f"replay adapter for {name!r} is not built") from exc
    missing = [fn for fn in _ADAPTER_FUNCTIONS if not callable(getattr(module, fn, None))]
    if missing:
        raise ReplayAdapterUnavailable(f"replay adapter for {name!r} lacks {', '.join(missing)}")
    return module
