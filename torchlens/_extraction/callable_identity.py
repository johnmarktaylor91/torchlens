"""Callable identity for extraction resume (extract memo D8 + item 15).

The classifier digests everything a Python callable can OBSERVE — bytecode,
signature facts, default values BY VALUE, closure cells BY VALUE, and every
global name the bytecode actually loads — and refuses exactly where it is
blind. The classification vocabulary is closed:

* ``complete`` — nothing opaque referenced; resume compares digests and a
  mismatch refuses typed.
* ``partial`` — at least one opaque reference, NAMED in the record (e.g.
  ``fn.os:module(os)``); resume refuses unless ``pipeline_id=`` attests the
  callable, recorded as ASSERTED, not measured.
* ``asserted`` — a caller-supplied ``pipeline_id`` claim; a policy level
  applied at resume time, never produced by this classifier.

Value rules (merged v4 spec): tensors/arrays fold shape+dtype+complete
contiguous bytes (never ``repr`` — repr truncates and collides); scalars and
strings fold ``repr``; list/tuple/dict recurse with SIZE and DEPTH bounds and
a CYCLE guard (an over-bound container or cycle DEMOTES to partial and is
named — never ``RecursionError``); set/frozenset demote (unordered); pure
modules fold ``module@version`` for an allowlisted namespace; functions and
``functools.partial`` recurse; bound methods digest ``__self__`` FIRST
(``nn.Module`` via the D6 state-digest route, other instances via sorted
``vars()``), then ``__func__``; ``nn.Module`` callables fold
``type(fn).forward`` plus the D6 digest of their own state; callable objects
fold ``type(fn).__call__`` plus sorted ``vars(fn)``.

The launch pure-module allowlist is exactly the tested set (stdlib-pure
namespaces + torch + numpy); anything else classifies partial, with
:func:`register_pure_module` as the documented, manifest-recorded escape —
an audited assertion, not an invisible trust grant.

Every spelling here is DOCUMENTED-UNSTABLE pending the naming sprint.
"""

from __future__ import annotations

import dis
import functools
import hashlib
import platform
import types
from typing import Any

import torch
from torch import nn

from .._errors import InvalidArgumentError
from .dtype_policy import tensor_payload_bytes

__tl_layer__ = "L5"

__all__ = [
    "CALLABLE_IDENTITY_ALGORITHM_ID",
    "CALLABLE_IDENTITY_ALGORITHM_VERSION",
    "classify_callable",
    "register_pure_module",
    "registered_pure_modules",
]

#: Pinned algorithm id recorded beside every callable-identity digest.
CALLABLE_IDENTITY_ALGORITHM_ID = "tl_callable_identity_blake2b"

#: Pinned encoding version of the digest fold. v2 folds nested code objects
#: structurally (v1 repr'd ``co_consts``, which embedded a process address for
#: every nested code object -- audit 2.10c); v1 and v2 digests are
#: INCOMPARABLE, which the resume rules disclose instead of reading as a
#: behavior change.
CALLABLE_IDENTITY_ALGORITHM_VERSION = 2

#: Launch pure-module allowlist: stdlib-pure namespaces plus torch + numpy —
#: exactly the tested set (extract D8) — plus torchlens itself (the
#: intervention-spec precedent: TorchLens-owned ``torchlens.*`` callables
#: always resolve; the package version rides every record). Growth is
#: post-launch repo-owner curation through :func:`register_pure_module`.
_BUILTIN_PURE_MODULES: frozenset[str] = frozenset(
    {"builtins", "math", "operator", "itertools", "functools", "torch", "numpy", "torchlens"}
)

#: User-registered pure-module namespaces (audited assertions; recorded in
#: every classification record that consults them).
_REGISTERED_PURE_MODULES: dict[str, str] = {}

#: Container recursion bounds: crossing either demotes to partial, named.
_MAX_CONTAINER_ITEMS = 256
_MAX_DEPTH = 8

#: Bytecode opcodes whose argument names are loads the callable can observe.
_GLOBAL_LOAD_OPS = frozenset({"LOAD_GLOBAL", "LOAD_NAME", "LOAD_DEREF"})


def register_pure_module(name: str) -> None:
    """Register a module namespace as pure for callable classification (D8).

    Registration is an AUDITED ASSERTION, not an invisible trust grant: the
    module name and its installed version are recorded in every
    classification record that folds it, and those records ride the
    extraction manifest.

    Parameters
    ----------
    name:
        Top-level module name (e.g. ``"einops"``).

    Raises
    ------
    torchlens.errors.InvalidArgumentError
        ``extraction_pure_module_invalid`` when the name is empty, dotted,
        or not an importable installed module.
    """

    import importlib

    if not isinstance(name, str) or not name or "." in name:
        raise InvalidArgumentError(
            f"register_pure_module() takes one TOP-LEVEL module name, got "
            f"{name!r}; purity is asserted per namespace, never per submodule.",
            code="extraction_pure_module_invalid",
            remedy='pass a top-level installed module name, e.g. "einops"',
            name=repr(name),
        )
    try:
        module = importlib.import_module(name)
    except ImportError as exc:
        raise InvalidArgumentError(
            f"register_pure_module({name!r}) could not import the module "
            f"({exc}); an unimportable namespace cannot be asserted pure.",
            code="extraction_pure_module_invalid",
            remedy="install the module first, then register it",
            name=name,
        ) from exc
    _REGISTERED_PURE_MODULES[name] = _module_version(module)


def registered_pure_modules() -> dict[str, str]:
    """Return a snapshot of the user-registered pure-module namespaces.

    Returns
    -------
    dict[str, str]
        ``{module_name: version}`` for every registered namespace.
    """

    return dict(_REGISTERED_PURE_MODULES)


def _module_version(module: types.ModuleType) -> str:
    """Return a module's version string, falling back to the Python version.

    Parameters
    ----------
    module:
        Imported module.

    Returns
    -------
    str
        ``__version__`` when present (third-party convention), else the
        interpreter version (stdlib modules version with the interpreter).
    """

    version = getattr(module, "__version__", None)
    return str(version) if version else f"python-{platform.python_version()}"


class _Fold:
    """One classification pass: a hash fold plus the opacity ledger."""

    def __init__(self) -> None:
        """Initialize an empty fold."""

        self.hasher = hashlib.blake2b(digest_size=32)
        self.opaque: list[str] = []
        self.pure_modules: dict[str, str] = {}
        self._seen: set[int] = set()
        # Function memo: every reachable function folds ONCE per pass;
        # re-encounters fold a stable back-reference ordinal instead. This
        # is what keeps the loaded-global walk LINEAR in unique functions
        # (the naive walk re-folds shared helpers along every path, which
        # is exponential in real modules) and terminates mutual recursion.
        self._fn_memo: dict[int, int] = {}

    def feed(self, tag: str, payload: bytes | str) -> None:
        """Fold one tagged payload into the digest.

        Parameters
        ----------
        tag:
            Structural tag keeping distinct fields from colliding.
        payload:
            Bytes or text to fold.
        """

        data = payload.encode("utf-8", "backslashreplace") if isinstance(payload, str) else payload
        self.hasher.update(tag.encode())
        self.hasher.update(len(data).to_bytes(8, "little"))
        self.hasher.update(data)

    def demote(self, path: str, why: str) -> None:
        """Record one named opaque reference (classification -> partial).

        Parameters
        ----------
        path:
            Dotted path of the opaque reference inside the callable.
        why:
            Short kind label (``module(os)``, ``cycle``, ``set``, ...).
        """

        self.opaque.append(f"{path}:{why}")
        self.feed("opaque", f"{path}:{why}")


def _pure_module_namespaces() -> dict[str, str]:
    """Return the active allowlist: builtins plus registered namespaces.

    Returns
    -------
    dict[str, str]
        ``{top_level_name: version}`` for every allowed namespace.
    """

    active: dict[str, str] = {}
    import importlib

    for name in _BUILTIN_PURE_MODULES:
        try:
            active[name] = _module_version(importlib.import_module(name))
        except ImportError:
            continue  # stdlib/torch import everywhere this package runs
    active.update(_REGISTERED_PURE_MODULES)
    return active


def _fold_leaf(fold: _Fold, value: Any, path: str) -> bool:
    """Fold one leaf value (scalars, tensors, arrays, torch scalars).

    Parameters
    ----------
    fold:
        The active fold.
    value:
        Candidate leaf value.
    path:
        Opacity path.

    Returns
    -------
    bool
        Whether the value was a leaf (folded or demoted here).
    """

    if value is None or isinstance(value, (bool, int, float, complex, str, bytes)):
        fold.feed("scalar", repr(value))
        return True
    if isinstance(value, torch.Tensor):
        detached = value.detach()
        if detached.is_meta:
            fold.demote(path, "meta_tensor")
            return True
        fold.feed("tensor", f"{tuple(detached.shape)}|{detached.dtype}")
        fold.feed("tensor_bytes", tensor_payload_bytes(detached))
        return True
    if type(value).__module__ == "numpy" and hasattr(value, "tobytes"):
        fold.feed("ndarray", f"{getattr(value, 'shape', None)}|{getattr(value, 'dtype', None)}")
        fold.feed("ndarray_bytes", value.tobytes())
        return True
    if isinstance(value, (torch.dtype, torch.device)):
        fold.feed("torch_scalar", repr(value))
        return True
    return False


def _fold_container(fold: _Fold, value: Any, path: str, depth: int) -> None:
    """Fold one ordered container with the size/depth/cycle bounds (D8).

    Parameters
    ----------
    fold:
        The active fold.
    value:
        list/tuple/dict container.
    path:
        Opacity path.
    depth:
        Remaining depth.
    """

    marker = id(value)
    if marker in fold._seen:
        fold.demote(path, "cycle")
        return
    if len(value) > _MAX_CONTAINER_ITEMS:
        fold.demote(path, f"container_over_{_MAX_CONTAINER_ITEMS}")
        return
    fold._seen.add(marker)
    try:
        if isinstance(value, dict):
            fold.feed("dict", str(len(value)))
            for key, item in value.items():
                _fold_value(fold, key, f"{path}[key]", depth - 1)
                _fold_value(fold, item, f"{path}[{key!r}]", depth - 1)
        else:
            fold.feed("seq", f"{type(value).__name__}:{len(value)}")
            for index, item in enumerate(value):
                _fold_value(fold, item, f"{path}[{index}]", depth - 1)
    finally:
        fold._seen.discard(marker)


def _fold_value(fold: _Fold, value: Any, path: str, depth: int) -> None:
    """Fold one observed value under the D8 value rules.

    Parameters
    ----------
    fold:
        The active classification fold.
    value:
        Observed value (default, closure cell, loaded global, ...).
    path:
        Dotted path for opacity naming.
    depth:
        Remaining recursion depth; exhaustion demotes, never raises.
    """

    if depth <= 0:
        fold.demote(path, "depth")
        return
    if _fold_leaf(fold, value, path):
        return
    if isinstance(value, (set, frozenset)):
        fold.demote(path, "set_unordered")
        return
    if isinstance(value, (list, tuple, dict)):
        _fold_container(fold, value, path, depth)
        return
    if _fold_namespace_identity(fold, value, path):
        return
    if _fold_callable_like(fold, value, path, depth):
        return
    fold.demote(path, f"opaque({type(value).__name__})")


def _fold_callable_like(fold: _Fold, value: Any, path: str, depth: int) -> bool:
    """Fold one callable-shaped value through its matching D8 rule.

    Parameters
    ----------
    fold:
        The active fold.
    value:
        Candidate callable-shaped value.
    path:
        Opacity path.
    depth:
        Remaining depth.

    Returns
    -------
    bool
        Whether the value was callable-shaped (folded or demoted here).
    """

    if isinstance(value, functools.partial):
        _fold_callable(fold, value.func, f"{path}.func", depth - 1)
        _fold_value(fold, tuple(value.args), f"{path}.args", depth - 1)
        _fold_value(fold, dict(value.keywords), f"{path}.keywords", depth - 1)
        return True
    if isinstance(value, nn.Module):
        _fold_module_callable(fold, value, path, depth)
        return True
    if isinstance(value, types.MethodType):
        _fold_bound_method(fold, value, path, depth)
        return True
    if isinstance(value, (types.FunctionType, types.BuiltinFunctionType)):
        _fold_callable(fold, value, path, depth - 1)
        return True
    if callable(value):
        _fold_callable_object(fold, value, path, depth)
        return True
    return False


def _fold_namespace_identity(fold: _Fold, value: Any, path: str) -> bool:
    """Fold module and CLASS values by allowlisted namespace identity.

    A class never folds by class-dict crawl (dataclass ``Field`` objects and
    descriptors are framework noise, not observable transform semantics).

    Parameters
    ----------
    fold:
        The active fold.
    value:
        Candidate module or class value.
    path:
        Opacity path.

    Returns
    -------
    bool
        Whether the value was a module/class (folded or demoted here).
    """

    if isinstance(value, types.ModuleType):
        top = value.__name__.split(".")[0]
        allowed = _pure_module_namespaces()
        if top in allowed:
            fold.feed("module", f"{value.__name__}@{allowed[top]}")
            fold.pure_modules[value.__name__] = allowed[top]
        else:
            fold.demote(path, f"module({value.__name__})")
        return True
    if isinstance(value, type):
        module = getattr(value, "__module__", "") or ""
        top = module.split(".")[0]
        allowed = _pure_module_namespaces()
        if top in allowed:
            fold.feed("class", f"{module}.{value.__qualname__}@{allowed[top]}")
        else:
            fold.demote(path, f"class({module}.{value.__qualname__})")
        return True
    return False


def _fold_module_callable(fold: _Fold, module: nn.Module, path: str, depth: int) -> None:
    """Fold an ``nn.Module`` value: forward code plus the D6 state digest.

    Parameters
    ----------
    fold:
        The active fold.
    module:
        The module value.
    path:
        Opacity path.
    depth:
        Remaining depth.
    """

    from .._data_substrate import compute_model_identity

    forward = type(module).forward
    if isinstance(forward, types.FunctionType):
        _fold_code(fold, forward, f"{path}.forward", depth - 1)
    else:
        fold.feed("module_forward", repr(forward))  # C-implemented forward
    identity = compute_model_identity(module, level="measured")
    if identity.get("level") == "measured":
        fold.feed("module_state", str(identity.get("digest")))
    else:
        fold.demote(path, "module_state_unmeasurable")


def _fold_bound_method(fold: _Fold, method: types.MethodType, path: str, depth: int) -> None:
    """Fold a bound method: ``__self__`` FIRST, then ``__func__`` (D8).

    Two bound methods differing only in instance state must separate, so the
    receiver is digested before the code object.

    Parameters
    ----------
    fold:
        The active fold.
    method:
        Bound method value.
    path:
        Opacity path.
    depth:
        Remaining depth.
    """

    receiver = method.__self__
    if isinstance(receiver, nn.Module):
        _fold_module_callable(fold, receiver, f"{path}.__self__", depth)
    elif isinstance(receiver, types.ModuleType):
        _fold_value(fold, receiver, f"{path}.__self__", depth - 1)
    else:
        try:
            state = vars(receiver)
        except TypeError:
            fold.feed("receiver", repr(type(receiver)))
        else:
            fold.feed("receiver_type", type(receiver).__qualname__)
            for name in sorted(state):
                _fold_value(fold, state[name], f"{path}.__self__.{name}", depth - 1)
    func = method.__func__
    if isinstance(func, types.FunctionType):
        _fold_code(fold, func, path, depth - 1)
    else:
        fold.feed("method_func", repr(func))


def _fold_callable_object(fold: _Fold, obj: Any, path: str, depth: int) -> None:
    """Fold a callable object: ``type(obj).__call__`` plus sorted ``vars``.

    Parameters
    ----------
    fold:
        The active fold.
    obj:
        Callable non-function object.
    path:
        Opacity path.
    depth:
        Remaining depth.
    """

    call = type(obj).__call__
    if isinstance(call, types.FunctionType):
        _fold_code(fold, call, f"{path}.__call__", depth - 1)
    else:
        fold.feed("call", repr(call))
    try:
        state = vars(obj)
    except TypeError:
        state = {}
    fold.feed("obj_type", type(obj).__qualname__)
    for name in sorted(state):
        _fold_value(fold, state[name], f"{path}.{name}", depth - 1)


def _loaded_global_names(code: types.CodeType) -> list[str]:
    """Return the names the bytecode actually loads as globals (D8).

    ``dis.get_instructions`` over LOAD_GLOBAL / LOAD_NAME — not every
    ``co_names`` entry, which falsely refuses ``t.float()`` against the
    builtin ``float``. LOAD_DEREF cells are folded by value separately.

    Parameters
    ----------
    code:
        Code object to scan.

    Returns
    -------
    list[str]
        Loaded global names in first-load order, deduplicated.
    """

    seen: list[str] = []
    for instruction in dis.get_instructions(code):
        if instruction.opname in ("LOAD_GLOBAL", "LOAD_NAME"):
            name = str(instruction.argval)
            if name not in seen:
                seen.append(name)
    return seen


def _fold_code_object(fold: _Fold, code: types.CodeType) -> None:
    """Fold one code object's process-stable facts, recursing into nested code.

    ``repr(code.co_consts)`` was folded historically; a nested code object
    (generator expression, nested lambda, comprehension on Python < 3.12)
    reprs as ``<code object <genexpr> at 0x7f...>`` -- an ADDRESS -- so the
    digest changed on every process and every resume with such a transform
    refused ``extraction_resume_callable_mismatch`` (audit 2.10c). Nested
    code objects fold structurally instead; only address-free consts repr.

    Parameters
    ----------
    fold:
        The active fold.
    code:
        The code object.
    """

    fold.feed("co_code", code.co_code)
    fold.feed(
        "co_facts",
        repr(
            (
                code.co_names,
                code.co_varnames,
                code.co_freevars,
                code.co_cellvars,
                code.co_argcount,
                code.co_kwonlyargcount,
                code.co_flags,
            )
        ),
    )
    fold.feed("co_consts_len", str(len(code.co_consts)))
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            fold.feed("nested_code", const.co_name)
            _fold_code_object(fold, const)
        else:
            fold.feed("const", repr(const))


def _fold_code(fold: _Fold, fn: types.FunctionType, path: str, depth: int) -> None:
    """Fold one Python function: code facts, defaults, cells, loaded globals.

    Parameters
    ----------
    fold:
        The active fold.
    fn:
        The function.
    path:
        Opacity path prefix.
    depth:
        Remaining depth.
    """

    import builtins

    marker = id(fn)
    if marker in fold._fn_memo:
        fold.feed("fn_ref", str(fold._fn_memo[marker]))
        return
    fold._fn_memo[marker] = len(fold._fn_memo)
    code = fn.__code__
    _fold_code_object(fold, code)
    for index, default in enumerate(fn.__defaults__ or ()):
        _fold_value(fold, default, f"{path}.__defaults__[{index}]", depth)
    for name, default in sorted((fn.__kwdefaults__ or {}).items()):
        _fold_value(fold, default, f"{path}.__kwdefaults__[{name}]", depth)
    free_names = code.co_freevars
    for name, cell in zip(free_names, fn.__closure__ or (), strict=False):
        try:
            cell_value = cell.cell_contents
        except ValueError:
            fold.demote(f"{path}.<cell:{name}>", "unbound_cell")
            continue
        _fold_value(fold, cell_value, f"{path}.<cell:{name}>", depth)
    for name in _loaded_global_names(code):
        if name in fn.__globals__:
            _fold_value(fold, fn.__globals__[name], f"{path}.{name}", depth)
        elif hasattr(builtins, name):
            fold.feed("builtin", name)
        else:
            fold.demote(f"{path}.{name}", "unresolvable_global")


def _fold_callable(fold: _Fold, fn: Any, path: str, depth: int) -> None:
    """Fold any callable kind through its matching rule.

    Parameters
    ----------
    fold:
        The active fold.
    fn:
        The callable.
    path:
        Opacity path.
    depth:
        Remaining depth.
    """

    if depth <= 0:
        fold.demote(path, "depth")
        return
    if isinstance(fn, (types.FunctionType, types.BuiltinFunctionType)):
        module = getattr(fn, "__module__", None) or (
            "builtins" if isinstance(fn, types.BuiltinFunctionType) else ""
        )
        top = module.split(".")[0]
        allowed = _pure_module_namespaces()
        if top in allowed:
            # An allowlisted-namespace function folds as its PINNED identity
            # (module.qualname@version), never by code walk: the namespace's
            # version pins its semantics, and the fold stays identical
            # whether TorchLens has wrapped the torch function or not (the
            # wrapper is a FunctionType whose globals hold mutable capture
            # registries — walking it would make the digest unstable).
            fold.feed("pure_fn", f"{module}.{fn.__qualname__}@{allowed[top]}")
            fold.pure_modules[module] = allowed[top]
            return
        if isinstance(fn, types.FunctionType):
            _fold_code(fold, fn, path, depth)
        else:
            fold.demote(path, f"builtin({module}.{fn.__qualname__})")
        return
    _fold_value(fold, fn, path, depth)


def classify_callable(fn: Any, *, slot: str = "callable") -> dict[str, Any]:
    """Classify one user callable for the extraction resume signature (D8).

    Parameters
    ----------
    fn:
        The callable to classify.
    slot:
        Slot name used as the opacity path root (``"collate"``,
        ``"transform[k][0]"``, ...) so refusals name the exact reference.

    Returns
    -------
    dict[str, Any]
        JSON-portable record: ``classification`` (``"complete"`` |
        ``"partial"``), ``digest`` (``"blake2b:..."`` — computed over the
        visible surface in both classes), ``opaque_references`` (named, e.g.
        ``"fn.os:module(os)"``), ``pure_modules`` folded, ``qualname``, and
        the pinned algorithm id + version.
    """

    fold = _Fold()
    _fold_callable(fold, fn, slot, _MAX_DEPTH)
    return {
        "classification": "partial" if fold.opaque else "complete",
        "digest": f"blake2b:{fold.hasher.hexdigest()}",
        "opaque_references": sorted(fold.opaque),
        "pure_modules": dict(sorted(fold.pure_modules.items())),
        "qualname": getattr(fn, "__qualname__", type(fn).__qualname__),
        "algorithm_id": CALLABLE_IDENTITY_ALGORITHM_ID,
        "algorithm_version": CALLABLE_IDENTITY_ALGORITHM_VERSION,
    }
