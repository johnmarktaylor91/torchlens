"""Shared helpers used by two or more TorchLens packages.

Each helper lives in its own submodule; import it from there. The package
namespace re-exports only the convenience helpers in :mod:`._onramp`, which the
README documents at ``torchlens._utils``.
"""

from __future__ import annotations

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
