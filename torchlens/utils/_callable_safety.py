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
``torch.serialization`` callables reachable.

MODULE granularity is NOT ENOUGH (round-5). The allowlist admits the whole
``torch`` namespace by prefix, but ``torch`` *also* hosts side-effecting builtins
whose real ``__module__`` is plainly ``"torch"`` -- most dangerously
``torch.from_file`` (a ``_VariableFunctionsClass`` factory that CREATES/TRUNCATES a
file with ``shared=True`` or READS an arbitrary file into a tensor with
``shared=False``), plus ``import_ir_module`` / ``PyTorchFileWriter`` /
``_load_global_deps`` and friends. A module-only denylist missed them (that is the
round-3 gap ``from_file`` slipped through). We therefore add a QUALNAME-level guard
on top of the module gate: an audited denylist of the exact side-effecting
callables reachable in the allowed namespace, PLUS a structural terminal-name guard
(``file`` / ``import`` / ``serial`` / ``pickle`` / ``marshal`` substrings and exact
``save`` / ``load`` / ``dump`` names) so a *future* sibling I/O/serialization gadget
is denied by NAME even if not enumerated.

A complete POSITIVE allowlist of every pure op was assessed and rejected as
infeasible: ``_VariableFunctionsClass`` alone exposes thousands of dynamically
generated builtins that drift across torch versions, so an exact-name allowlist
would either be perpetually incomplete (breaking legitimate ops on a new torch) or
impossible to keep current. The residual risk of the denylist+structural approach
is a future side-effecting torch callable whose name matches NONE of the structural
patterns and is not enumerated; network/process/import side effects are additionally
covered by the module denylist, and the structural guard covers the entire file-I/O
/ serialization / import class by name -- the class ``from_file`` belongs to.

The high-signal substrings were chosen to have ZERO overlap with the pure forward-op
surface (verified by enumeration): pure ops never contain ``file`` / ``import`` /
``serial`` / ``pickle`` / ``marshal``. Bare ``load`` / ``save`` substrings are
deliberately NOT used (they would wrongly deny ``Tensor.module_load`` /
``_overload``); the ``load``-suffixed siblings (``_load_global_deps`` /
``_preload_cuda_deps``) are enumerated exactly instead. ``from_numpy`` and
``frombuffer`` are pure and stay resolvable (they match none of the patterns).
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


# Exact terminal callable names that are side-effecting even though they live in an
# allowlisted module (chiefly ``torch``, which also hosts every pure tensor op).
# Enumerated by auditing every callable reachable through the allowlisted roots /
# tensor-op namespaces whose module PASSES the gate above: FILE I/O, SERIALIZATION,
# IMPORT, and dynamic-library-load callables. None collide with a pure forward-op
# name.
_DENIED_CALLABLE_NAMES: frozenset[str] = frozenset(
    {
        # File-I/O tensor factory (the round-5 exploit) + jit file-reader/writer /
        # serializer classes reachable at ``torch.*``.
        "from_file",
        "PyTorchFileReader",
        "PyTorchFileWriter",
        "FileCheck",
        "ScriptModuleSerializer",
        "SerializationStorageContext",
        "DeserializationStorageContext",
        "get_file_path",
        # jit module (de)serialization from disk / buffer.
        "import_ir_module",
        "import_ir_module_from_buffer",
        # Arbitrary import / dynamic-library load side effects.
        "_import_device_backends",
        "_import_dotted_name",
        "_load_global_deps",
        "_preload_cuda_deps",
        # Belt-and-suspenders: the two serialization entry points, in case a torch
        # version ever re-exports them with ``__module__ == "torch"`` (their real
        # module ``torch.serialization`` is already module-denied).
        "save",
        "load",
    }
)

# Exact terminal names that are serialization/dump entry points across common APIs.
# Kept EXACT (not substring) so ``Tensor.module_load`` / ``_overload`` -- pure,
# in-memory ops whose names merely contain "load" -- stay resolvable.
_DENIED_EXACT_NAMES: frozenset[str] = frozenset(
    {"save", "load", "loads", "dump", "dumps", "load_all", "dump_all", "safe_load"}
)

# Structural terminal-name substrings that mark file-I/O / serialization / import
# side effects. Every substring was verified to have NO overlap with the reachable
# pure forward-op surface, so a future sibling gadget (e.g. a new ``*_from_file``
# factory) is denied by name even if it is not yet enumerated above.
_DENIED_NAME_SUBSTRINGS: tuple[str, ...] = (
    "file",
    "import",
    "serial",
    "pickle",
    "unpickle",
    "marshal",
    "shelve",
)


def _terminal_callable_name(func: Callable[..., Any]) -> str:
    """Return a callable's terminal name (last ``__qualname__`` component fallback)."""

    name = getattr(func, "__name__", None)
    if not name:
        qualname = str(getattr(func, "__qualname__", "") or "")
        name = qualname.rsplit(".", maxsplit=1)[-1]
    return str(name or "")


def _is_side_effecting_callable_name(func: Callable[..., Any]) -> bool:
    """Return whether a callable's name marks it as side-effecting.

    This is the QUALNAME-level guard layered on top of the module gate: it denies
    file-I/O / serialization / import callables that live inside an allowlisted
    module (``torch.from_file`` and its audited siblings), which a module-only
    policy cannot distinguish from the pure tensor ops sharing that module.
    """

    name = _terminal_callable_name(func)
    if name in _DENIED_CALLABLE_NAMES or name in _DENIED_EXACT_NAMES:
        return True
    lowered = name.lower()
    return any(substring in lowered for substring in _DENIED_NAME_SUBSTRINGS)


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

    The callable is unwrapped to its real identity, then admitted only if (a) its
    terminal NAME is not a side-effecting file-I/O / serialization / import callable
    and (b) its module is on the positive allowlist and off the side-effecting
    denylist. Module-less C tensor method descriptors are admitted when bound to a
    Tensor class. Anything else -- notably ``torch.load`` / ``torch.save`` /
    ``torch.from_file`` and any ``os`` / ``pickle`` / ``subprocess`` callable -- is
    refused. The name guard runs FIRST so a side-effecting builtin whose real module
    is the allowlisted ``torch`` namespace (``from_file`` lives there) cannot slip
    the module-granular gate.
    """

    real = _unwrap_capture_wrapper(func)
    if _is_side_effecting_callable_name(real):
        return False
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
