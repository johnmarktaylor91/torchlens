"""Capture of legacy tensor constructors: ``torch.<dtype>Tensor(...)`` and ``Variable(t)``.

2017-2019 model code builds tensors through the legacy dtype classes
(``torch.FloatTensor(size).normal_()``, ``torch.LongTensor(data)``, a module
attribute bound at init such as ``self.FloatTensor = torch.FloatTensor``) and
wraps tensors in ``torch.autograd.Variable``. Their C constructors dispatch
``aten.empty`` / ``aten.detach`` (and friends) with no Python wrapper on the
stack and are invisible to ``TorchFunctionMode``, so such a forward left an
unowned dispatch and failed validation completeness.

Each class in ``constants.LEGACY_TENSOR_CONSTRUCTOR_SITES`` gets its
``__new__`` replaced IN PLACE (``utils/_type_new_slot.py``): the class object,
its identity, ``isinstance`` and ``x.type(torch.FloatTensor)`` are unchanged,
and any alias bound before the first capture reaches the same patched class.
The new ``__new__`` routes the call through ``torch_func_decorator``, so a
constructor call during a capture is one logged op (func name = class name,
``cuda_`` prefixed for ``torch.cuda`` classes) that owns its dispatches, and
outside a capture the decorator's fast path runs the original C constructor.
Installed with the other wrappers by ``wrap_torch``; ``unwrap_torch`` restores
the original ``tp_new`` slot and class ``__dict__`` exactly.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ...constants import LEGACY_TENSOR_CONSTRUCTOR_SITES
from ...utils import _torch_compat
from ...utils._type_new_slot import (
    TypeNewPatch,
    has_python_new_trampoline,
    install_new_override,
    original_new_caller,
    probe_type_new_slot_patch,
    restore_new_override,
)

__all__ = [
    "install_legacy_constructor_wrappers",
    "installed_legacy_constructor_classes",
    "skipped_legacy_constructor_classes",
    "uninstall_legacy_constructor_wrappers",
]


@dataclass(frozen=True)
class _InstalledLegacyConstructor:
    """One patched legacy constructor class and its undo record."""

    func_name: str
    patch: TypeNewPatch


#: id(class) -> installed record; ids are stable because torch keeps every
#: legacy class alive for the life of the process.
_INSTALLED: dict[int, _InstalledLegacyConstructor] = {}

#: func name -> class left unpatched because a Python-level ``__new__`` was
#: already in effect on it at install time (rebuilt on every install).
_SKIPPED_FOREIGN_NEW: dict[str, type] = {}

#: Classes already warned about, so a repeated ``wrap_torch`` warns once.
_WARNED_FOREIGN_NEW: set[str] = set()


def _legacy_func_name(namespace_name: str, class_name: str) -> str:
    """Return the recorded op name for one roster row."""

    if namespace_name == "torch.cuda":
        return f"cuda_{class_name}"
    return class_name


def _resolve_class(namespace_name: str, class_name: str) -> type | None:
    """Return the live roster class, or ``None`` when this torch lacks it."""

    namespace = _torch_compat.get_optional_torch_namespace(namespace_name)
    if namespace is None:
        return None
    cls = getattr(namespace, class_name, None)
    return cls if isinstance(cls, type) else None


def _build_new_impl(cls: type, func_name: str) -> Callable[..., Any]:
    """Return the ``__new__`` replacement that logs ``cls(...)`` calls.

    Parameters
    ----------
    cls:
        The legacy constructor class, still unpatched.
    func_name:
        Recorded op name.

    Returns
    -------
    Callable[..., Any]
        ``__new__(subtype, *args, **kwargs)``. Calls on ``cls`` itself go
        through the logging decorator; a Python subclass (``Variable`` admits
        them) calls the original constructor unchanged.
    """

    from .wrappers import torch_func_decorator

    call_original = original_new_caller(cls)

    def original(*args: Any, **kwargs: Any) -> Any:
        """Run the eager legacy constructor for ``cls``."""

        return call_original(cls, args, kwargs)

    original.__name__ = cls.__name__
    original.__qualname__ = cls.__qualname__
    original.__module__ = cls.__module__
    logged = torch_func_decorator(original, func_name)

    def legacy_new(subtype: type, *args: Any, **kwargs: Any) -> Any:
        """Dispatch a legacy constructor call through the TorchLens logging gate."""

        if subtype is cls:
            return logged(*args, **kwargs)
        return call_original(subtype, args, kwargs)

    legacy_new.__name__ = "__new__"
    legacy_new.__qualname__ = f"{cls.__qualname__}.__new__"
    return legacy_new


def _resolve_roster() -> list[tuple[str, type]]:
    """Return ``(func_name, class)`` for every roster row this torch provides."""

    resolved: list[tuple[str, type]] = []
    for namespace_name, class_name in LEGACY_TENSOR_CONSTRUCTOR_SITES:
        cls = _resolve_class(namespace_name, class_name)
        if cls is not None:
            resolved.append((_legacy_func_name(namespace_name, class_name), cls))
    return resolved


def _still_installed(cls: type) -> bool:
    """Return whether our patch on ``cls`` is still live; drop a stale record.

    A record whose override a third party replaced stays registered while
    that foreign ``__new__`` is in effect; once it is gone (the slot is no
    longer a Python trampoline) the record is dropped so the class is
    re-patched instead of being left uncaptured for good.
    """

    record = _INSTALLED.get(id(cls))
    if record is None:
        return False
    if cls.__dict__.get("__new__") is record.patch.installed_new:
        return True
    if has_python_new_trampoline(cls):
        return True
    del _INSTALLED[id(cls)]
    return False


def _skip_foreign_new(func_name: str, cls: type) -> None:
    """Record and disclose (once) a class left unpatched over a foreign ``__new__``."""

    _SKIPPED_FOREIGN_NEW[func_name] = cls
    if func_name in _WARNED_FOREIGN_NEW:
        return
    _WARNED_FOREIGN_NEW.add(func_name)
    from ..._errors import TorchLensWarning

    warnings.warn(
        f"TorchLens left the legacy constructor {cls.__module__}.{cls.__qualname__} "
        "uncaptured: its constructor slot already dispatches through a Python-level __new__ "
        "(another tool's patch on it or on a base). "
        "TorchLens never clobbers foreign patches, so calls to it during capture are not "
        "logged and tl.validate may report them as unowned dispatches. Remove the other "
        "patch before the first capture to capture it.",
        TorchLensWarning,
        stacklevel=4,
    )


def install_legacy_constructor_wrappers() -> None:
    """Patch every resolvable legacy constructor class, idempotently.

    A no-op when ``HAS_LEGACY_CONSTRUCTOR_NEW_PATCH`` is False (non-CPython or
    an unexpected type layout); legacy constructors then stay uncaptured, as
    before, and validation completeness reports them. Every class's layout is
    re-probed before anything is written: one failure flips the flag through
    ``mark_torch_capability_missing`` and patches nothing. A class already
    carrying a Python-level ``__new__`` (another tool's patch) is skipped with
    a one-time ``TorchLensWarning`` and listed by
    :func:`skipped_legacy_constructor_classes`; wrapping over it would make the
    saved "original" re-enter our own override forever.
    """

    if not _torch_compat.HAS_LEGACY_CONSTRUCTOR_NEW_PATCH:
        return
    roster = _resolve_roster()
    drifted = [name for name, cls in roster if not probe_type_new_slot_patch(cls)]
    if drifted:
        _torch_compat.mark_torch_capability_missing(
            "HAS_LEGACY_CONSTRUCTOR_NEW_PATCH",
            "the CPython type layout of "
            f"{', '.join(drifted)} does not match TorchLens's mirror, so legacy tensor "
            "constructors (torch.FloatTensor(...), Variable(...)) are not captured.",
        )
        return
    _SKIPPED_FOREIGN_NEW.clear()
    for func_name, cls in roster:
        if _still_installed(cls):
            continue
        if has_python_new_trampoline(cls):
            _skip_foreign_new(func_name, cls)
            continue
        patch = install_new_override(cls, _build_new_impl(cls, func_name))
        _INSTALLED[id(cls)] = _InstalledLegacyConstructor(func_name=func_name, patch=patch)


def uninstall_legacy_constructor_wrappers() -> None:
    """Restore every patched class's original constructor exactly.

    A class whose ``__new__`` was re-patched by a third party after ours is left
    as found (TorchLens never clobbers foreign patches) and stays registered,
    so a later uninstall can still restore it once the foreign patch is gone.
    """

    for key, record in list(_INSTALLED.items()):
        if restore_new_override(record.patch):
            del _INSTALLED[key]


def installed_legacy_constructor_classes() -> dict[str, type]:
    """Return the currently patched classes keyed by recorded op name.

    Returns
    -------
    dict[str, type]
        ``{func_name: class}`` for every installed legacy constructor.
    """

    return {record.func_name: record.patch.cls for record in _INSTALLED.values()}


def skipped_legacy_constructor_classes() -> dict[str, type]:
    """Return the classes the last install left unpatched over a foreign ``__new__``.

    Returns
    -------
    dict[str, type]
        ``{func_name: class}``; empty when every resolvable class was patched.
    """

    return dict(_SKIPPED_FOREIGN_NEW)
