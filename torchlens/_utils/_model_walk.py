"""One model walk per module tree, reused while the tree is unchanged.

A capture session reads the model's module tree many times: session
preparation, the buffer tracker, the held-tensor scans, the robustness and
distributed checks and session cleanup each walked ``model.modules()`` or
``model.named_modules()`` on their own. The walk depends only on the tree
(which module objects are registered, under which names, in which order), so
this module records it once per tree and hands the recorded walk back for as
long as the tree is the same.

Every read proves the tree unchanged first with :func:`module_tree_key`, a
walk of the registries that makes no Python call, so a submodule added,
removed, replaced, renamed or moved since the last read is walked afresh at
that read, inside a session or between two of them. Parameters and buffers
are not recorded: they are read from the live registries at each call, so a
swapped parameter or a reassigned buffer is always current.

The record holds modules only through weak references and is keyed weakly on
the root, so nothing here keeps a model, a module, a tensor or a trace alive.
A class that overrides how torch walks its modules (``named_modules``,
``modules``, ``named_parameters``, ``parameters``, ``named_buffers``,
``buffers`` or ``_named_members``) makes the whole tree ineligible, and every
helper then defers to the live torch walk.
"""

from __future__ import annotations

import weakref
from typing import Any

from torch import nn

from ._torch_compat import get_module_named_members

#: Root module -> its recorded walk. Weak in the root; the value holds modules weakly.
_RECORDED: weakref.WeakKeyDictionary[nn.Module, tuple[Any, ...]] = weakref.WeakKeyDictionary()

#: The walk methods a class must inherit unchanged from ``nn.Module`` for the record to stand in.
_WALK_METHODS = (
    "named_modules",
    "modules",
    "named_parameters",
    "parameters",
    "named_buffers",
    "buffers",
    "_named_members",
)
#: ``_named_members`` comes through the compat layer: a fresh sentinel when torch lacks it,
#: so no class matches and every tree is walked live.
_STOCK = tuple(getattr(nn.Module, name) for name in _WALK_METHODS[:-1]) + (
    get_module_named_members() or object(),
)


def module_tree_key(model: Any) -> tuple[Any, ...] | None:
    """Return the identity, class and child names of every module under ``model``.

    Every registration path is walked, parents before children, so the key
    fixes which modules the tree holds, under which names and in which order:
    all that a module walk reads. Modules are held through weak references,
    and a dead reference equals nothing but itself, so a new module that
    happens to be allocated where a replaced one lived still reads as a
    change. A child registered as ``None`` keeps its place in the key.

    Returns
    -------
    tuple[Any, ...] | None
        The key, or ``None`` when a module's registry cannot be walked or a
        module cannot be weakly referenced; the caller then walks live.
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
            # Subscripts, not ``.values()``: the walk runs on every read and
            # is kept free of calls so the proof costs next to nothing.
            for name in children:
                pending += (children[name],)
    except Exception:
        # Not an error to report: an undescribable tree is walked live.
        return None
    return tuple(key)


def same_module_tree(old: tuple[Any, ...], new: tuple[Any, ...]) -> bool:
    """Return whether two :func:`module_tree_key` keys describe the same tree.

    Modules are compared by identity, never with ``==``: a module class with
    its own ``__eq__`` cannot make a replaced module read as the old one.
    """

    if len(old) != len(new):
        return False
    for index in range(len(old)):
        left = old[index]
        right = new[index]
        if left is right:
            continue
        if type(left) is weakref.ref and type(right) is weakref.ref:
            target = left()
            if target is None or target is not right():
                return False
            continue
        if type(left) is str and type(right) is str and left == right:
            continue
        return False
    return True


def _stock_walks(key: tuple[Any, ...]) -> bool:
    """Return whether every module class in ``key`` walks the tree the way ``nn.Module`` does.

    The classes come from the key, which reaches every registered module, so a
    module that hides itself from its own walk is still checked.
    """

    for cls in key:
        if not isinstance(cls, type):
            continue
        for name, stock in zip(_WALK_METHODS, _STOCK, strict=True):
            if getattr(cls, name, None) is not stock:
                return False
        if getattr(cls, "__eq__") is not object.__eq__ or (
            getattr(cls, "__hash__") is not object.__hash__
        ):
            # torch's walk de-duplicates through a set of modules.
            return False
    return True


def _record(model: nn.Module, key: tuple[Any, ...]) -> tuple[Any, ...]:
    """Walk ``model`` live and record the walk under ``key``.

    The record is ``(key, eligible, names, refs, path_names, path_refs)``; the
    last four are empty when a class in the tree walks its own way, so later
    reads of the same tree go straight to the live walk.
    """

    if not _stock_walks(key):
        record: tuple[Any, ...] = (key, False, (), (), (), ())
    else:
        named = tuple(model.named_modules())
        every_path = tuple(model.named_modules(remove_duplicate=False))
        record = (
            key,
            True,
            tuple(name for name, _module in named),
            tuple(weakref.ref(module) for _name, module in named),
            tuple(name for name, _module in every_path),
            tuple(weakref.ref(module) for _name, module in every_path),
        )
    _RECORDED[model] = record
    return record


def _current(model: Any) -> tuple[Any, ...] | None:
    """Return ``model``'s recorded walk when it stands in for the live one, else ``None``."""

    if not isinstance(model, nn.Module):
        return None
    key = module_tree_key(model)
    if key is None:
        _RECORDED.pop(model, None)
        return None
    record = _RECORDED.get(model)
    if record is None or not same_module_tree(record[0], key):
        record = _record(model, key)
    return record if record[1] else None


def current_tree_record(model: Any) -> tuple[Any, ...] | None:
    """Return the recorded walk standing for ``model``'s tree now, or ``None``.

    The same object comes back for as long as the tree is unchanged, and a new
    one after any change, so a cache keyed on this tree holds the record and
    compares it by identity. ``None`` means the tree is walked live (it cannot
    be described, or a class in it walks its own way), and nothing may be
    cached on it.
    """

    return _current(model)


def _paired(names: tuple[str, ...], refs: tuple[Any, ...]) -> tuple[tuple[str, Any], ...]:
    """Return ``(name, module)`` pairs from the record, dereferenced without a call per item."""

    out: list[Any] = [None] * len(refs)
    for index in range(len(refs)):
        out[index] = (names[index], refs[index]())
    return tuple(out)


def model_named_modules(
    model: nn.Module, remove_duplicate: bool = True
) -> tuple[tuple[str, nn.Module], ...]:
    """Return what ``tuple(model.named_modules(remove_duplicate=...))`` returns, walked once per tree."""

    record = _current(model)
    if record is None:
        if remove_duplicate:
            return tuple(model.named_modules())
        return tuple(model.named_modules(remove_duplicate=False))
    if remove_duplicate:
        return _paired(record[2], record[3])
    return _paired(record[4], record[5])


def model_modules(model: nn.Module) -> tuple[nn.Module, ...]:
    """Return what ``tuple(model.modules())`` returns, walked once per tree."""

    record = _current(model)
    if record is None:
        return tuple(model.modules())
    refs = record[3]
    out: list[Any] = [None] * len(refs)
    for index in range(len(refs)):
        out[index] = refs[index]()
    return tuple(out)


def _named_members(
    model: nn.Module, buffers: bool, remove_duplicate: bool
) -> tuple[tuple[str, Any], ...] | None:
    """Return torch's ``_named_members`` result for one registry, or ``None`` to walk live."""

    record = _current(model)
    if record is None:
        return None
    names, refs = (record[2], record[3]) if remove_duplicate else (record[4], record[5])
    seen: set[int] = set()
    out: list[Any] = []
    for index in range(len(refs)):
        prefix = names[index]
        module = refs[index]()
        members = module._buffers if buffers else module._parameters
        for name in members:
            value = members[name]
            if value is None:
                continue
            if remove_duplicate:
                if id(value) in seen:
                    continue
                seen.add(id(value))
            out += ((f"{prefix}.{name}" if prefix else name, value),)
    return tuple(out)


def model_named_parameters(
    model: nn.Module, remove_duplicate: bool = True
) -> tuple[tuple[str, nn.Parameter], ...]:
    """Return what ``tuple(model.named_parameters(remove_duplicate=...))`` returns, read live."""

    members = _named_members(model, False, remove_duplicate)
    if members is None:
        if remove_duplicate:
            return tuple(model.named_parameters())
        return tuple(model.named_parameters(remove_duplicate=False))
    return members


def model_parameters(model: nn.Module) -> tuple[nn.Parameter, ...]:
    """Return what ``tuple(model.parameters())`` returns, read live."""

    named = _named_members(model, False, True)
    if named is None:
        return tuple(model.parameters())
    out: list[Any] = [None] * len(named)
    for index in range(len(named)):
        out[index] = named[index][1]
    return tuple(out)


def model_named_buffers(
    model: nn.Module, remove_duplicate: bool = True
) -> tuple[tuple[str, Any], ...]:
    """Return what ``tuple(model.named_buffers(remove_duplicate=...))`` returns, read live."""

    members = _named_members(model, True, remove_duplicate)
    if members is None:
        if remove_duplicate:
            return tuple(model.named_buffers())
        return tuple(model.named_buffers(remove_duplicate=False))
    return members


def own_named_buffers(module: nn.Module) -> tuple[tuple[str, Any], ...]:
    """Return what ``tuple(module.named_buffers(recurse=False))`` returns, read live."""

    return _own_members(module, "_buffers", "named_buffers")


def own_named_parameters(module: nn.Module) -> tuple[tuple[str, Any], ...]:
    """Return what ``tuple(module.named_parameters(recurse=False))`` returns, read live."""

    return _own_members(module, "_parameters", "named_parameters")


def _own_members(module: nn.Module, registry: str, method: str) -> tuple[tuple[str, Any], ...]:
    """Read one module's own registry the way torch's non-recursive walk does."""

    cls = type(module)
    if getattr(cls, method, None) is not getattr(nn.Module, method) or (
        getattr(cls, "_named_members", None) is not _STOCK[-1]
    ):
        return tuple(getattr(module, method)(recurse=False))
    members = getattr(module, registry)
    out: list[Any] = []
    for name in members:
        value = members[name]
        if value is None:
            continue
        duplicate = False
        for _seen_name, seen in out:
            if seen is value:
                duplicate = True
                break
        if not duplicate:
            out += ((name, value),)
    return tuple(out)


__all__ = [
    "current_tree_record",
    "model_modules",
    "model_named_buffers",
    "model_named_modules",
    "model_named_parameters",
    "model_parameters",
    "module_tree_key",
    "own_named_buffers",
    "own_named_parameters",
    "same_module_tree",
]
