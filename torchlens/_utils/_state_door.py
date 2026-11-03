"""Door from ``_utils`` per-call paths to the ``torchlens._state`` leaf.

``_state`` sits above ``_utils`` in the layer map, so a ``_utils`` module may
not import it at module top (``arch_spine_layer_lint``). A function-level
``from .. import _state`` keeps the edge deferred but runs an import statement
(``_handle_fromlist``) on every call, which per-op helpers such as the
tensor-size unwrap pay thousands of times per trace. :func:`state` holds the
one deferred import and caches the module, so those helpers read ``_state``
without an import statement after the first call. The import is a plain
``from`` statement so ``tools/layering/graph.py`` keeps seeing the edge.
"""

from __future__ import annotations

from functools import cache
from types import ModuleType

__all__ = ["state"]


@cache
def state() -> ModuleType:
    """Return the ``torchlens._state`` module, importing it on the first call only."""

    from .. import _state

    return _state
