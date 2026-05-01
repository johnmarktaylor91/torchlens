"""External-tool bridge namespace."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

_BRIDGE_MODULES = {
    "brain_score",
    "captum",
    "depyf",
    "dialz",
    "gradcam",
    "huggingface",
    "inseq",
    "lit",
    "nnsight",
    "profiler",
    "repeng",
    "rsatoolbox",
    "sae_lens",
    "shap",
    "steering_vectors",
}


def __getattr__(name: str) -> ModuleType:
    """Import bridge modules lazily.

    Parameters
    ----------
    name:
        Bridge module name.

    Returns
    -------
    ModuleType
        Imported bridge module.

    Raises
    ------
    AttributeError
        If ``name`` is not a known bridge module.
    """

    if name not in _BRIDGE_MODULES:
        raise AttributeError(f"module 'torchlens.bridge' has no attribute {name!r}")
    module = importlib.import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    """Return visible bridge namespace members.

    Returns
    -------
    list[str]
        Sorted bridge module names plus module globals.
    """

    return sorted([*globals(), *_BRIDGE_MODULES])


__all__ = [
    "brain_score",
    "captum",
    "depyf",
    "dialz",
    "gradcam",
    "huggingface",
    "inseq",
    "lit",
    "nnsight",
    "profiler",
    "repeng",
    "rsatoolbox",
    "sae_lens",
    "shap",
    "steering_vectors",
]
