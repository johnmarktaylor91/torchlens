"""Shared helpers used by two or more TorchLens packages.

Each helper lives in its own submodule; import it from there. The package
namespace re-exports only the convenience helpers in :mod:`._onramp`, which the
README documents at ``torchlens._utils``; they load on first use so the package's
submodules import without torch.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._onramp import (
        find_executable_save_set,
        flop_count,
        list_modules,
        list_ops,
        peek_graph,
        synthetic_input,
        trace_streaming,
    )

__all__ = [
    "find_executable_save_set",
    "flop_count",
    "list_modules",
    "list_ops",
    "peek_graph",
    "synthetic_input",
    "trace_streaming",
]


def __getattr__(name: str) -> Any:
    """Load the :mod:`._onramp` convenience helpers on first use.

    The helpers import torch; the package's own submodules (hashing, the
    record group table, the compatibility probes) stay importable without it,
    which the torch-free agent verbs rely on.
    """

    if name in __all__:
        return getattr(importlib.import_module("._onramp", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
