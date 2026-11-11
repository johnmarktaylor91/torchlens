"""Proof that a model's module tree is the one its one-time preparation saw.

One-time preparation gives every submodule its address and class name, and a
model already prepared skips it on later traces. The skip is right only while
the module tree is the same one: a submodule added, replaced, renamed or moved
between two traces has no address yet, or a stale one. This module records the
tree a model was prepared with; when the next trace finds a different tree, or
one it cannot describe, the model is prepared in full again.
"""

from __future__ import annotations

import weakref
from typing import Any

from ... import _state
from ..._utils._model_walk import kept_tree_key, module_tree_key, same_module_tree

#: Returned by :func:`changed_module_tree` when the recorded tree still holds.
UNCHANGED: Any = object()

#: Model -> the tree key its last one-time preparation ran over. Weak in the
#: model, and the kept key holds modules and classes weakly, so nothing here
#: keeps a model alive.
_prepared_trees: weakref.WeakKeyDictionary[Any, tuple[Any, ...]] = weakref.WeakKeyDictionary()


def changed_module_tree(model: Any) -> Any:
    """Compare ``model``'s tree with the one recorded at its last preparation.

    Call before one-time preparation. When the recorded tree still holds,
    :data:`UNCHANGED` is returned and the prepared model keeps its fast path.
    Otherwise the model is dropped from the prepared set, so the preparation
    that follows runs in full, and the tree to record afterwards is returned
    (``None`` when it cannot be described).
    """

    current = module_tree_key(model)
    recorded = _prepared_trees.get(model)
    if current is not None and recorded is not None and same_module_tree(recorded, current):
        return UNCHANGED
    _state.release_model_prep(model, ())
    return current


def record_prepared_tree(model: Any, tree: tuple[Any, ...] | None) -> None:
    """Record ``tree`` as what ``model`` was just prepared over (``None`` forgets it)."""

    if tree is None:
        _prepared_trees.pop(model, None)
    else:
        _prepared_trees[model] = kept_tree_key(tree)


__all__ = ["UNCHANGED", "changed_module_tree", "module_tree_key", "record_prepared_tree"]
