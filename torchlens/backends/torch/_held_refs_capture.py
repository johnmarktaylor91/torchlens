"""Capture-scoped rebinding of module-held pristine torch functions.

Under lazy wrapping a model is usually built BEFORE the process's first
capture, so a module that stores a torch function at construction holds the
pristine original (transformers' ``GELUActivation`` keeps ``F.gelu``;
``GELUTanh`` keeps ``functools.partial(F.gelu, approximate="tanh")``). Those
calls bypass the wrappers, so every capture disclosed a provenance gap and
paid a second (rescue) forward. Before each capture's forward this module
rebinds such references to the live wrappers, in the holder shapes the
stale-reference work covers: direct attributes, ``functools.partial``, closure
cells and default arguments of held functions and of the class ``forward``, and
exact builtin ``list``/``dict``/``tuple`` containers and namedtuples. Every
mutation is undone at session cleanup, so the user's objects are left exactly
as they were (the same objects in the same slots). References in any other
holder (module globals, custom objects, builtin subclasses) are untouched and
keep the disclosure-plus-rescue path.
"""

from __future__ import annotations

import functools
import types
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from torch import nn

from ... import _state
from ._held_refs import _is_namedtuple_instance, _live_counterpart

if TYPE_CHECKING:
    from ...data_classes.trace import Trace

__all__ = ["rebind_held_torch_refs", "restore_held_torch_refs"]

_UNDO_FIELD = "_held_torch_ref_rebinds"
# Container nesting the scan descends (the ``{"acts": [F.gelu]}`` shape is 2).
_MAX_DEPTH = 4
# Module roots whose functions are library code, never user holders.
_LIBRARY_ROOTS = ("torch", "torchlens")
# nn.Module bookkeeping slots: never user holders, skipped for cost.
_MODULE_INTERNAL_SLOTS = frozenset(
    {
        "_parameters",
        "_buffers",
        "_modules",
        "_non_persistent_buffers_set",
        "forward",
    }
)


def _wrapper_for(value: Any) -> Any | None:
    """Return the live wrapper for a ledgered pristine torch function, else ``None``."""

    if not callable(value):
        return None
    _decorated_to_orig, orig_to_decorated = _state.wrap_epoch_ledgers()
    if id(value) not in orig_to_decorated:
        return None
    return _live_counterpart(value)


def _is_library_function(fn: types.FunctionType) -> bool:
    """Torch's and TorchLens's own functions are never scanned.

    A wrapper's closure holds the pristine original it calls; rebinding that
    cell to the wrapper would make the wrapper call itself. Torch's Python
    functions are ledgered originals or library internals, never user holders.
    """

    decorated_to_orig, orig_to_decorated = _state.wrap_epoch_ledgers()
    if id(fn) in decorated_to_orig or id(fn) in orig_to_decorated:
        return True
    module_name = getattr(fn, "__module__", None) or ""
    return any(module_name == root or module_name.startswith(f"{root}.") for root in _LIBRARY_ROOTS)


class _Rebinder:
    """Collect rebinds for one capture; ``undo`` restores them in reverse."""

    def __init__(self) -> None:
        self.undo: list[Callable[[], None]] = []
        self._seen_functions: set[int] = set()

    def replacement(self, value: Any, depth: int) -> Any | None:
        """Return a rebound replacement for ``value``, or ``None``.

        Identity-bearing mutable holders (lists, dicts, function cells and
        defaults) are rebound in place with an undo entry and return ``None``;
        immutable holders return a rebuilt object the caller installs.
        """

        wrapper = _wrapper_for(value)
        if wrapper is not None:
            return wrapper
        if depth >= _MAX_DEPTH:
            return None
        value_type = type(value)
        if value_type is functools.partial:
            return self._partial(value, depth + 1)
        if value_type is types.FunctionType:
            self.function(value, depth + 1)
        elif value_type is list:
            self._list(value, depth + 1)
        elif value_type is dict:
            self._dict(value, depth + 1)
        elif value_type is tuple or _is_namedtuple_instance(value):
            return self._tuple(value, depth + 1)
        return None

    def function(self, fn: types.FunctionType, depth: int) -> None:
        """Rebind pristine refs in a function's closure cells and defaults."""

        if id(fn) in self._seen_functions or _is_library_function(fn):
            return
        self._seen_functions.add(id(fn))
        for cell in fn.__closure__ or ():
            try:
                contents = cell.cell_contents
            except ValueError:  # an empty cell
                continue
            new = self.replacement(contents, depth)
            if new is not None:
                cell.cell_contents = new
                self.undo.append(functools.partial(setattr, cell, "cell_contents", contents))
        defaults = fn.__defaults__
        if defaults:
            swapped = self._tuple(defaults, depth)
            if swapped is not None:
                fn.__defaults__ = swapped
                self.undo.append(functools.partial(setattr, fn, "__defaults__", defaults))
        kwdefaults = fn.__kwdefaults__
        if kwdefaults:
            swapped_kw = self._keywords(kwdefaults, depth)
            if swapped_kw is not None:
                fn.__kwdefaults__ = swapped_kw
                self.undo.append(functools.partial(setattr, fn, "__kwdefaults__", kwdefaults))

    def _partial(self, value: functools.partial[Any], depth: int) -> functools.partial[Any] | None:
        """Rebuild a partial whose ``func``, ``args`` or ``keywords`` hold a pristine ref."""

        func = self.replacement(value.func, depth)
        args = self._tuple(value.args, depth) if value.args else None
        keywords = self._keywords(value.keywords, depth) if value.keywords else None
        if func is None and args is None and keywords is None:
            return None
        rebuilt = functools.partial(
            value.func if func is None else func,
            *(value.args if args is None else args),
            **(value.keywords if keywords is None else keywords),
        )
        rebuilt.__dict__.update(value.__dict__)
        return rebuilt

    def _keywords(self, mapping: dict[str, Any], depth: int) -> dict[str, Any] | None:
        """Return a rebound copy of a keyword mapping, or ``None`` if nothing changed."""

        swapped = {key: self.replacement(val, depth) for key, val in mapping.items()}
        if all(val is None for val in swapped.values()):
            return None
        return {key: mapping[key] if new is None else new for key, new in swapped.items()}

    def _list(self, container: list[Any], depth: int) -> None:
        for index, item in enumerate(container):
            new = self.replacement(item, depth)
            if new is not None:
                container[index] = new
                self.undo.append(functools.partial(container.__setitem__, index, item))

    def _dict(self, container: dict[Any, Any], depth: int) -> None:
        for key, item in tuple(container.items()):
            new = self.replacement(item, depth)
            if new is not None:
                container[key] = new
                self.undo.append(functools.partial(container.__setitem__, key, item))

    def _tuple(self, values: tuple[Any, ...], depth: int) -> tuple[Any, ...] | None:
        swapped = [self.replacement(item, depth) for item in values]
        if all(item is None for item in swapped):
            return None
        items = [new if new is not None else old for new, old in zip(swapped, values, strict=True)]
        if type(values) is tuple:
            return tuple(items)
        try:
            return type(values)._make(items)  # type: ignore[attr-defined]
        except (TypeError, ValueError):  # an exotic ``_make``: leave it to the rescue
            return None

    def module(self, module: nn.Module) -> None:
        """Rebind one module's instance attributes and its class ``forward``."""

        slots = module.__dict__
        for name, value in tuple(slots.items()):
            if name in _MODULE_INTERNAL_SLOTS or name.startswith("_tl"):
                continue
            new = self.replacement(value, 0)
            if new is not None:
                slots[name] = new
                self.undo.append(functools.partial(_restore_slot, slots, name, value, new))
        class_forward = getattr(type(module), "forward", None)
        if type(class_forward) is types.FunctionType:
            self.function(class_forward, 1)


def _restore_slot(slots: dict[str, Any], name: str, original: Any, installed: Any) -> None:
    """Put ``original`` back unless the slot was reassigned after the rebind."""

    if slots.get(name) is installed:
        slots[name] = original


def rebind_held_torch_refs(trace: Trace, model: nn.Module) -> None:
    """Rebind module-held pristine torch functions to the live wrappers.

    Called last in per-session preparation, while the wrappers are installed.
    The undo list is parked on the trace for :func:`restore_held_torch_refs`;
    if the scan itself fails, everything it already rebound is restored before
    the error propagates, so the model is never left half rebound.
    """

    if not _state._is_decorated:
        return
    rebinder = _Rebinder()
    try:
        for module in model.modules():
            rebinder.module(module)
    except BaseException:
        _run_undo(rebinder.undo)
        raise
    if rebinder.undo:
        trace.__dict__[_UNDO_FIELD] = rebinder.undo


def restore_held_torch_refs(trace: Trace) -> None:
    """Undo this capture's rebinds, newest first, leaving the user's objects intact.

    Idempotent: the undo list is popped, so the failed-forward path may call it
    before its trace scrub and the shared session cleanup again after.
    """

    _run_undo(trace.__dict__.pop(_UNDO_FIELD, None) or ())


def _run_undo(undo: Sequence[Callable[[], None]]) -> None:
    for action in reversed(undo):
        action()
