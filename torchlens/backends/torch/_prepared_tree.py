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

#: Returned by :func:`changed_module_tree` when the recorded tree still holds.
UNCHANGED: Any = object()

#: Model -> the tree key its last one-time preparation ran over. Weak in the
#: model, and the key holds modules weakly, so nothing here keeps a model alive.
_prepared_trees: weakref.WeakKeyDictionary[Any, tuple[Any, ...]] = weakref.WeakKeyDictionary()


def module_tree_key(model: Any) -> tuple[Any, ...] | None:
    """Return the identity, class and child names of every module under ``model``.

    Every registration path is walked, parents before children, so the key
    fixes which modules the tree holds, under which names and in which order:
    all that one-time preparation reads. Modules are held through weak
    references, and a dead reference equals nothing but itself, so a new
    module that happens to be allocated where a replaced one lived still reads
    as a change. A child registered as ``None`` keeps its place in the key.

    Returns
    -------
    tuple[Any, ...] | None
        The key, or ``None`` when a module's registry cannot be walked or a
        module cannot be weakly referenced; the caller then prepares in full.
    """

    key: list[Any] = []
    pending: list[Any] = [model]
    try:
        for module in pending:
            if module is None:
                key += (None,)
                continue
            children = module._modules
            key += (weakref.ref(module), type(module), *children)
            pending += children.values()
    except Exception:
        # Not an error to report: an undescribable tree is prepared in full.
        return None
    return tuple(key)


def changed_module_tree(model: Any) -> Any:
    """Compare ``model``'s tree with the one recorded at its last preparation.

    Call before one-time preparation. When the recorded tree still holds,
    :data:`UNCHANGED` is returned and the prepared model keeps its fast path.
    Otherwise the model is dropped from the prepared set, so the preparation
    that follows runs in full, and the tree to record afterwards is returned
    (``None`` when it cannot be described).
    """

    current = module_tree_key(model)
    if current is not None and model in _state._prepared_models:
        try:
            if _prepared_trees.get(model) == current:
                return UNCHANGED
        except Exception:
            # A module whose ``__eq__`` raises cannot be proven unchanged.
            pass
    _state._prepared_models.discard(model)
    return current


def record_prepared_tree(model: Any, tree: tuple[Any, ...] | None) -> None:
    """Record ``tree`` as what ``model`` was just prepared over (``None`` forgets it)."""

    if tree is None:
        _prepared_trees.pop(model, None)
    else:
        _prepared_trees[model] = tree


__all__ = ["UNCHANGED", "changed_module_tree", "module_tree_key", "record_prepared_tree"]
