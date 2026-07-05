"""Callback integration namespace reserved for TorchLens 2.0."""

from __future__ import annotations

import importlib
from types import ModuleType

_CALLBACK_MODULES = {"lightning"}


def __getattr__(name: str) -> ModuleType:
    """Import callback integrations lazily.

    Parameters
    ----------
    name:
        Callback integration module name.

    Returns
    -------
    ModuleType
        Imported callback module.

    Raises
    ------
    AttributeError
        If ``name`` is not a known callback integration.
    """

    if name not in _CALLBACK_MODULES:
        raise AttributeError(f"module 'torchlens.callbacks' has no attribute {name!r}")
    module = importlib.import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    """Return visible callback namespace members.

    Returns
    -------
    list[str]
        Sorted callback module names plus module globals.
    """

    return sorted([*globals(), *_CALLBACK_MODULES])


__all__ = ["lightning"]
