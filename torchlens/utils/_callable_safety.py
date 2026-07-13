"""Security gate deciding which resolved callables are pure forward/tensor ops.

A loaded ``.tlspec`` bundle is UNTRUSTED input. Its per-op callable registry
keys are resolved by ``getattr`` over the torch, ``torch.Tensor``,
``torch.nn.functional`` and ``operator`` namespaces (plus enumerated torch
tensor-op submodules). That surface *also* exposes side-effecting callables --
most dangerously ``torch.load`` / ``torch.save`` (both live in
``torch.serialization``), which unpickle attacker files (RCE) or write attacker
tensors to arbitrary paths during ``Trace.run``. Filtering only by
``callable(...)`` -- as the resolvers historically did -- therefore let a crafted
bundle op keyed ``("torch", "load")`` or ``("torch", "save")`` resolve and
execute before any downstream faithfulness check could fire.

Gating on the wrapped-op inventory (``get_orig_torch_funcs``) is INSUFFICIENT
because ``("torch", "load")`` is itself a wrapped op. This module instead admits
a resolved callable only when its REAL module identity (after unwrapping any
TorchLens capture wrapper) is a pure tensor-op module. The primary gate is a
positive allowlist (default-deny); a curated denylist of side-effecting module
identities is checked first as belt-and-suspenders, so a future widening of the
allowlist can never silently re-admit ``os`` / ``pickle`` / ``torch.serialization``
and friends.

The reachable-callable universe was enumerated across every allowlisted root and
enumerated namespace: every legitimate forward/tensor op clusters in the small,
stable module set below, while ``torch.load`` / ``torch.save`` are the *only*
``torch.serialization`` callables reachable. The allowlist is therefore complete
for the reachable forward-op surface, and the two proven exploits are unreachable.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

# Module identities (exact name or dotted-prefix ancestor) whose callables are
# pure forward/tensor ops safe to resolve from an untrusted bundle key. Derived
# by enumerating every callable reachable through the resolvers' allowlisted
# roots and torch tensor-op namespaces.
_ALLOWED_FORWARD_OP_MODULES: frozenset[str] = frozenset(
    {
        "torch",
        "torch.functional",
        "torch.nn.functional",
        "torch._tensor",
        "torch._VF",
        "torch.fft",
        "torch.linalg",
        "torch.special",
        "torch.nested",
        "torch._C",
        "torch._C._nn",
        "torch._C._fft",
        "torch._C._linalg",
        "torch._C._nested",
        "torch._C._special",
        "torch._C._VariableFunctions",
        "torch._C._VariableFunctionsClass",
        "torch._C._TensorBase",
        "torch._C.TensorBase",
        "operator",
        "_operator",
    }
)

# Side-effecting / dangerous module identities that must NEVER resolve from an
# untrusted bundle key, regardless of the allowlist. Checked first. Covers
# serialization (unpickle / arbitrary write), code execution, process spawning,
# imports, and disk / OS I/O.
_DENIED_MODULES: frozenset[str] = frozenset(
    {
        # Serialization: unpickle (RCE) and arbitrary-path tensor writes.
        "torch.serialization",
        "torch.jit",
        "torch.package",
        "torch.hub",
        "torch.storage",
        "torch.multiprocessing",
        "torch.distributed",
        "torch._utils_internal",
        "pickle",
        "_pickle",
        "marshal",
        # Code execution / imports.
        "builtins",
        "importlib",
        "runpy",
        "code",
        "codeop",
        "ctypes",
        # Process / OS / filesystem I/O.
        "os",
        "posix",
        "nt",
        "sys",
        "subprocess",
        "shutil",
        "socket",
        "pty",
        "signal",
        "threading",
        "multiprocessing",
        "glob",
        "tempfile",
        "pathlib",
    }
)


def _tensor_method_owners() -> frozenset[type]:
    """Return the class objects that own genuine C-level tensor method descriptors."""

    owners: set[type] = {torch.Tensor}
    for name in ("TensorBase", "_TensorBase"):
        candidate = getattr(torch._C, name, None)
        if isinstance(candidate, type):
            owners.add(candidate)
    return frozenset(owners)


_TENSOR_METHOD_OWNERS = _tensor_method_owners()


def _unwrap_capture_wrapper(func: Callable[..., Any]) -> Callable[..., Any]:
    """Translate a TorchLens capture wrapper to its original callable.

    Wrapped torch ops report ``__module__ == 'torchlens.backends.torch.wrappers'``;
    the security decision must be made on the REAL underlying callable, so this
    walks the decorated->original map. Safe and idempotent for already-unwrapped
    callables. Imported lazily to avoid any import-order coupling.
    """

    try:
        from .. import _state
    except Exception:  # pragma: no cover - defensive; _state always imports here.
        return func
    current = func
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        original = _state._decorated_to_orig.get(id(current))
        if original is None:
            break
        current = original
    return current


def _matches(module: str, patterns: frozenset[str]) -> bool:
    """Return whether ``module`` equals or is nested under any pattern."""

    return any(module == pattern or module.startswith(pattern + ".") for pattern in patterns)


def _is_tensor_method_descriptor(func: Callable[..., Any]) -> bool:
    """Return whether a module-less callable is a genuine ``torch.Tensor`` method.

    C-level tensor method descriptors report ``__module__ is None``. They are
    admitted only when bound to a Tensor class via ``__objclass__``; any other
    module-less callable is denied.
    """

    objclass = getattr(func, "__objclass__", None)
    if not isinstance(objclass, type):
        return False
    if objclass in _TENSOR_METHOD_OWNERS:
        return True
    return issubclass(objclass, torch.Tensor)


def is_pure_forward_callable(func: Callable[..., Any]) -> bool:
    """Return whether a resolved callable is a pure, side-effect-free forward op.

    The callable is unwrapped to its real identity, then admitted only if its
    module is on the positive allowlist and off the side-effecting denylist.
    Module-less C tensor method descriptors are admitted when bound to a Tensor
    class. Anything else -- notably ``torch.load`` / ``torch.save`` and any
    ``os`` / ``pickle`` / ``subprocess`` callable -- is refused.
    """

    real = _unwrap_capture_wrapper(func)
    module = str(getattr(real, "__module__", "") or "")
    if module == "":
        return _is_tensor_method_descriptor(real)
    if _matches(module, _DENIED_MODULES):
        return False
    return _matches(module, _ALLOWED_FORWARD_OP_MODULES)


def unsafe_callable_reason(func: Callable[..., Any]) -> str:
    """Return a short, stable description of a refused callable's real module.

    Used only to populate a typed diagnostic; never returned to untrusted code.
    """

    real = _unwrap_capture_wrapper(func)
    module = str(getattr(real, "__module__", "") or "<none>")
    qualname = str(getattr(real, "__qualname__", getattr(real, "__name__", "?")))
    return f"{module}:{qualname}"
