"""The closed table of TorchLens-owned synthetic callables a runnable artifact may name.

Capture records two TorchLens-owned ops as computational calls:

* the module boundary mint (``_state._decorated_identity``: ``nn.Identity`` and
  pass-through module outputs) records ``torchlens._utils.display.identity``, a pure
  pass-through;
* a resolved foreign data round trip (numpy, DLPack or ``tolist`` out and back in plain
  forward code) records ``torchlens._utils._foreign_data_replay.foreigndata``, a pure
  function of the departed tensor and checked literals.

The table is keyed on the exact persisted registry key and resolves by in-memory
reference: no artifact-selected module is ever imported (``runnable_load``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal

from .._utils._foreign_data_replay import foreigndata
from .._utils.display import identity

SyntheticKey = tuple[Literal["custom"], str, Literal["function"], str]
"""``(namespace, qualname, kind, import_path)`` of a persisted ``FunctionRegistryKey``."""

BOUNDARY_IDENTITY_KEY: SyntheticKey = (
    "custom",
    "identity",
    "function",
    "torchlens._utils.display:identity",
)
"""Registry key capture mints for the module boundary identity op."""

FOREIGN_DATA_REPLAY_KEY: SyntheticKey = (
    "custom",
    "foreigndata",
    "function",
    "torchlens._utils._foreign_data_replay:foreigndata",
)
"""Registry key capture mints for a resolved foreign data round trip op."""

TORCHLENS_SYNTHETIC_CALLABLES: Mapping[SyntheticKey, tuple[str, Callable[..., Any]]] = {
    BOUNDARY_IDENTITY_KEY: ("torchlens._utils.display.identity", identity),
    FOREIGN_DATA_REPLAY_KEY: ("torchlens._utils._foreign_data_replay.foreigndata", foreigndata),
}
"""Persisted registry key fields to ``(resolved qualname, in-memory callable)``.

Plain tuples keep this module an import leaf; ``runnable_load`` builds the
``FunctionRegistryKey`` table from them.
"""
