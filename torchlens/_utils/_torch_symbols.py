"""Single sanctioned spelling for resolving a top-level ``torch`` attribute on the
load / decode / exec path (r42 secC_1 / r45 secC_1 / r47 secD_1).

The top-level ``torch`` module implements PEP 562 ``__getattr__``: a bare
``getattr(torch, name)`` on an attacker-derived manifest / literal string can trigger an
UNREQUESTED lazy submodule import (``torch.onnx`` / ``torch._dynamo`` / ``torch._inductor`` /
``torch._export``), fire a deprecated-attribute ``replacement()`` shim (``torch.has_cuda`` ->
``torch.backends.cuda.is_built()``), or leak a raw ``ImportError`` outside TorchLens' typed
error vocabulary -- all BEFORE any value gate runs. Reading ``torch.__dict__`` directly never
invokes ``torch.__getattr__``, and every real dtype / layout / memory_format / qscheme / ``Size``
symbol (plus every real top-level class such as ``UntypedStorage`` / ``TypedStorage`` and every
public free-function alias such as ``equal`` / ``allclose`` / ``is_nonzero`` / ``add``) is a
genuine ``torch.__dict__`` entry, so it still resolves.

r47 secD_1 / secF_1: this is the CANONICAL helper home. It lives under ``torchlens._utils`` -- a
low-level package that imports only ``torch`` -- so low-level importers (``utils/_torch_compat``,
``backends/torch/*``, ``_runnable_state``, ``capture/*``, ``fastlog/*``, ``intervention/*``,
``postprocess/*``) resolve it WITHOUT pulling ``_io/__init__.py`` (avoiding an import cycle).
``torchlens/_io/_torch_symbols.py`` re-exports :func:`torch_attr` unchanged for the load/decode
importers already spelling ``from ._torch_symbols import torch_attr``. The public contract is one
helper, not a physical module path.

Every attacker-derived top-level ``torch``-name resolution anywhere in the package routes through
:func:`torch_attr`; a whole-package AST immunizer (``tests/test_r45_torch_symbol_decode.py``) fails
on any bare ``getattr(torch, <non-literal>)`` OR ``hasattr(torch, <non-literal>)`` in
``torchlens/**/*.py`` so a future site cannot reintroduce the lazy-import / deprecated-replacement
side effect. Class roots (``torch.Tensor`` / ``torch._C`` / ``torch.backends``) and literal-name
``getattr(torch, "...")`` module-layout constants carry no lazy hazard and stay out of scope.

:func:`shadowed_torch_submodule` is the dotted-path companion: it recovers an imported torch
submodule that a same-named package attribute hides from attribute resolution.

:func:`tensor_read` and :func:`tensor_reads` resolve a ``torch.Tensor`` METHOD for TorchLens's
own bookkeeping reads (storage identity, element counts, layout flags), made tens of thousands of
times per capture under ``pause_logging``. Each read used to re-enter the per-op wrapper only to
take its logging-off fast path and call the original. The returned callable (called with the
receiver as its only argument) is the saved original, the unwrap ledger's entry for the TorchLens
wrapper on the receiver's class, only when that wrapper's fast path would be a pure pass-through:

* the backend registered its pass-through predicate and unwrap ledger (:data:`_INTERNAL_READS`)
  and the predicate holds: logging is off, no guarded fast re-run collector is observing wrapped
  calls, and the call fingerprint would not count the call;
* the receiver's class is exactly ``torch.Tensor`` or ``torch.nn.Parameter``, so no subclass
  override or ``__torch_function__`` decline redispatch can apply;
* the name is in :data:`ROUTED_TENSOR_READS`, whose wrappers carry no fast-path side effect (no
  clone-on-write materialization, device injection, subclass construction or detached-activation
  propagation; ``tests/test_internal_call_fastpath.py`` reads each wrapper's closure to prove it).

Otherwise the callable is ``operator.methodcaller(name)``, the exact spelling the call site used
before (``t.untyped_storage()``), so the wrapper runs as it always did. The class attribute is
compared by identity on every call, so a re-wrap, an unwrap or a peer patch over the wrapper is
honoured immediately. Callers keep their ``pause_logging`` scope.
"""

from __future__ import annotations

import operator
import sys
import types
from collections.abc import Callable, Mapping
from typing import Any, cast

import torch


def torch_attr(name: str) -> Any | None:
    """Resolve one top-level ``torch`` attribute without firing ``torch.__getattr__``.

    Parameters
    ----------
    name:
        Bare top-level ``torch`` attribute name (no ``torch.`` prefix, no dotted path).

    Returns
    -------
    Any | None
        The value from ``torch.__dict__`` when present, otherwise ``None``. A dotted or
        non-identifier ``name`` (never a real top-level torch symbol) returns ``None``
        without touching ``torch.__dict__`` -- so no lazy import, deprecated
        ``replacement()``, or raw ``ImportError`` can escape this call.
    """

    if "." in name or not name.isidentifier():
        return None
    return torch.__dict__.get(name)


def _resolve_dtype(dtype: object | None) -> object | None:
    """Convert serialized torch dtype names back to dtype objects.

    Parameters
    ----------
    dtype
        Dtype object or serialized dtype name from an event ref.

    Returns
    -------
    object | None
        Runtime dtype when resolvable, otherwise the original value.
    """

    if dtype is None or not isinstance(dtype, str):
        return dtype
    if dtype.startswith("torch."):
        dtype_name = dtype.split(".", 1)[1]
        # r47 secD_1: ``torch_attr`` reads ``torch.__dict__`` (no lazy ``torch.__getattr__``);
        # fall back to the original string when the name is not a real top-level torch symbol.
        return torch_attr(dtype_name) or dtype
    return dtype


def torch_dtype_named(name: str) -> torch.dtype | None:
    """Resolve a bare dtype name (``"float32"``) to a ``torch.dtype``, else ``None``.

    The one dtype-name parser: it reads through :func:`torch_attr` (no lazy
    ``torch.__getattr__``) and keeps only a real ``torch.dtype``, so an
    attacker-chosen name such as ``"load"`` never resolves a callable. Callers
    own any ``torch.`` prefix handling and their refusal error.

    Parameters
    ----------
    name:
        Bare top-level dtype name.

    Returns
    -------
    torch.dtype | None
        The dtype, or ``None`` when ``name`` does not name one.
    """

    value = torch_attr(name)
    return value if isinstance(value, torch.dtype) else None


def torch_dtype_from_string(dtype_name: str | None) -> torch.dtype | None:
    """Return a ``torch.dtype`` for canonical dtype strings when possible."""

    if dtype_name is None:
        return None
    if dtype_name.startswith("torch."):
        return torch_dtype_named(dtype_name.removeprefix("torch."))
    return None


def torch_device_from_string(device_name: Any) -> torch.device | None:
    """Return a ``torch.device`` from a string-like field when possible."""

    if device_name is None:
        return None
    try:
        return torch.device(str(device_name))
    except (TypeError, RuntimeError):
        return None


def shadowed_torch_submodule(namespace_name: str, resolved: Any) -> types.ModuleType | None:
    """Return the imported submodule that a same-named package attribute shadows.

    A package attribute can shadow an imported submodule of the same name:
    ``torch.nn.utils.weight_norm`` resolves by attribute to the function, while a
    roster row naming that dotted path means the module that holds the
    ``_weight_norm`` alias.

    Parameters
    ----------
    namespace_name:
        Dotted torch namespace, as written in the wrapper roster.
    resolved:
        What attribute resolution of ``namespace_name`` returned (``None`` if absent).

    Returns
    -------
    types.ModuleType | None
        ``sys.modules[namespace_name]`` when attribute resolution found a non-module
        object and that submodule is imported; otherwise ``None``.
    """

    submodule = sys.modules.get(namespace_name)
    if (
        resolved is not None
        and not isinstance(resolved, types.ModuleType)
        and isinstance(submodule, types.ModuleType)
    ):
        return submodule
    return None


# SECURITY BOUNDARY: the one table of in-memory roots a saved callable key may name. The
# sparse-run resolver (``torchlens._io.runnable_load``) and the load-time function-key
# verifier (``torchlens.intervention._resolver_registry``) both read these, so the loader
# never admits a namespace the replay resolver refuses or the reverse. Every hit is still
# held to ``is_pure_forward_callable`` by the caller; nothing here imports by name.
ALLOWED_EXACT_ROOTS: Mapping[str, Any] = {
    "torch": torch,
    "torch.Tensor": torch.Tensor,
    "torch.nn.functional": torch.nn.functional,
    "operator": operator,
}
ENUMERATED_TORCH_NAMESPACES = frozenset(
    {
        "torch.fft",
        "torch.linalg",
        "torch.nested",
        "torch.special",
        "torch.nn.functional",
        "torch.Tensor",
        "torch._C._nn",
        # ``torch.fft.<name>`` / ``torch.linalg.<name>`` ARE ``torch._C._fft.fft_<name>`` /
        # ``torch._C._linalg.linalg_<name>`` (bound by ``_add_docstr``), so captured keys
        # record the private module. Enumerating it resolves the recorded callable itself
        # on every torch version, instead of relying on the producer-version-bounded
        # private-to-public alias rows.
        "torch._C._fft",
        "torch._C._linalg",
        "torch._C._special",
        "torch._C._VariableFunctions",
        "torch._C._VariableFunctionsClass",
        "torch._C._TensorBase",
        "torch._C.TensorBase",
        "torch._VF",
    }
)


def torch_namespace(namespace: str) -> Any | None:
    """Traverse an enumerated torch namespace without importing modules.

    Parameters
    ----------
    namespace:
        Dotted ``torch.<...>`` namespace from :data:`ENUMERATED_TORCH_NAMESPACES`.

    Returns
    -------
    Any | None
        The namespace object, or ``None`` when a hop is absent.
    """

    parts = namespace.removeprefix("torch.").split(".")
    # r49 secF_1: the FIRST hop is off the TOP-LEVEL torch module, whose PEP-562
    # ``__getattr__`` can fire a lazy submodule import; resolve it through ``torch_attr``.
    # Later hops are off submodule/class roots that carry no lazy-import hazard.
    current: Any = torch_attr(parts[0])
    for part in parts[1:]:
        if current is None:
            return None
        current = getattr(current, part, None)
    return current


def getattr_allowlisted(namespace: str, qualname: str) -> Callable[..., Any] | None:
    """Read a callable from one explicitly allowlisted in-memory namespace.

    Parameters
    ----------
    namespace:
        A key of :data:`ALLOWED_EXACT_ROOTS` or a member of
        :data:`ENUMERATED_TORCH_NAMESPACES`; any other namespace returns ``None``.
    qualname:
        One attribute name on that namespace.

    Returns
    -------
    Callable[..., Any] | None
        The callable, or ``None`` when absent, not callable, or the namespace is not listed.
    """

    if namespace in ALLOWED_EXACT_ROOTS:
        root = ALLOWED_EXACT_ROOTS[namespace]
    elif namespace in ENUMERATED_TORCH_NAMESPACES:
        root = torch_namespace(namespace)
    else:
        return None
    if root is None:
        return None
    # r42 secC_1: the PEP-562 lazy-submodule hazard lives ONLY on the top-level ``torch``
    # module, so it is read through its ``__dict__``; class roots and the proxying
    # ``torch._VF`` module expose inherited/proxied callables only through ``getattr``.
    value = torch_attr(qualname) if root is torch else getattr(root, qualname, None)
    return cast(Callable[..., Any], value) if callable(value) else None


def enumerated_torch_callable(stock_path: str) -> tuple[Callable[..., Any], str] | None:
    """Resolve a dotted ``torch.<namespace>.<name>`` path on an enumerated namespace.

    Parameters
    ----------
    stock_path:
        Dotted stock torch path; its owner (left of the last dot) must be listed in
        :data:`ENUMERATED_TORCH_NAMESPACES`.

    Returns
    -------
    tuple[Callable[..., Any], str] | None
        The callable and its resolved dotted name, or ``None``.
    """

    namespace, _, qualname = stock_path.rpartition(".")
    if namespace not in ENUMERATED_TORCH_NAMESPACES:
        return None
    func = getattr_allowlisted(namespace, qualname)
    if func is None:
        return None
    return func, f"{namespace}.{qualname}"


ROUTED_TENSOR_READS: frozenset[str] = frozenset(
    {
        "data_ptr",
        "element_size",
        "is_conj",
        "is_contiguous",
        "is_inference",
        "is_neg",
        "nelement",
        "numel",
        "storage_offset",
        "untyped_storage",
    }
)
"""Zero-argument tensor reads whose wrapper fast path is a pure pass-through."""

_SPELLED: dict[str, Callable[[Any], Any]] = {
    name: operator.methodcaller(name) for name in ROUTED_TENSOR_READS
}
_TENSOR = torch.Tensor
_PARAMETER = torch.nn.Parameter

_INTERNAL_READS: tuple[Callable[[], bool], Mapping[int, Callable[..., Any]]] | None = None
"""The backend's pass-through predicate and unwrap ledger; ``None`` routes nothing."""


def register_internal_reads(
    pass_through: Callable[[], bool], ledger: Mapping[int, Callable[..., Any]]
) -> None:
    """Install the predicate and unwrap ledger :func:`tensor_read` consults.

    Parameters
    ----------
    pass_through:
        Returns whether a wrapped pure read would only pass through its wrapper now.
    ledger:
        The append-only ``id(wrapper) -> original`` unwrap ledger (read only).
    """

    global _INTERNAL_READS
    _INTERNAL_READS = (pass_through, ledger)


def _resolve(name: str, attr: Any, ledger: Mapping[int, Callable[..., Any]]) -> Callable[..., Any]:
    """Resolve and remember the callable for class attribute ``attr`` under routed ``name``.

    Parameters
    ----------
    name:
        A name in :data:`ROUTED_TENSOR_READS`.
    attr:
        The receiver class's current attribute for ``name``.
    ledger:
        The unwrap ledger.

    Returns
    -------
    Callable[..., Any]
        The ledger's original for ``attr``, else the method-call spelling.
    """

    original = ledger.get(id(attr))
    resolved = _SPELLED[name] if original is None else original
    _RESOLVED[name] = (attr, resolved)
    return resolved


# Last resolution per name, keyed by the class attribute it was resolved from (held by
# identity, so a re-wrap, unwrap or peer patch misses and resolves again). Reads stay free
# of calls on a hit: the receiver class is read as ``__class__`` and the attribute through
# the class ``__dict__``, which is what ``getattr`` on these two classes returns for a
# patched name; a name absent from ``torch.Tensor.__dict__`` is not wrapped, so it keeps
# the method call.
_RESOLVED: dict[str, tuple[Any, Callable[..., Any]]] = {
    name: (None, spelled) for name, spelled in _SPELLED.items()
}
_TENSOR_ATTRS = _TENSOR.__dict__
_PARAMETER_ATTRS = _PARAMETER.__dict__


def tensor_read(tensor: Any, name: str) -> Callable[[Any], Any]:
    """Return the callable for one routed read of ``tensor`` (call it with ``tensor``).

    Parameters
    ----------
    tensor:
        The receiver of the read.
    name:
        A name in :data:`ROUTED_TENSOR_READS`.

    Returns
    -------
    Callable[[Any], Any]
        The saved original when the wrapper would pass through, else the method call.
    """

    bound = _INTERNAL_READS
    cls = tensor.__class__
    if bound is None or (cls is not _TENSOR and cls is not _PARAMETER) or not bound[0]():
        return _SPELLED[name]
    attrs = _PARAMETER_ATTRS if name in _PARAMETER_ATTRS and cls is _PARAMETER else _TENSOR_ATTRS
    if name not in attrs:
        return _SPELLED[name]
    attr = attrs[name]
    seen = _RESOLVED[name]
    return seen[1] if seen[0] is attr else _resolve(name, attr, bound[1])


def tensor_reads(tensor: Any, *names: str) -> tuple[Callable[[Any], Any], ...]:
    """Return the callables for several routed reads of ``tensor``, in ``names`` order.

    One gate check covers every name, so a site reading several properties of one tensor
    fetches them once per scope.

    Parameters
    ----------
    tensor:
        The receiver of the reads.
    *names:
        Names in :data:`ROUTED_TENSOR_READS`.

    Returns
    -------
    tuple[Callable[[Any], Any], ...]
        One callable per name, each as :func:`tensor_read` would return it.
    """

    # Complexity C (11) on purpose: hot path. The gate and the per-name memo check stay
    # inline so a hit makes no call beyond the one gate predicate per scope.
    bound = _INTERNAL_READS
    cls = tensor.__class__
    resolved: tuple[Callable[[Any], Any], ...] = ()
    if bound is None or (cls is not _TENSOR and cls is not _PARAMETER) or not bound[0]():
        for name in names:
            resolved += (_SPELLED[name],)
        return resolved
    for name in names:
        attrs = (
            _PARAMETER_ATTRS if name in _PARAMETER_ATTRS and cls is _PARAMETER else _TENSOR_ATTRS
        )
        if name not in attrs:
            resolved += (_SPELLED[name],)
            continue
        attr = attrs[name]
        seen = _RESOLVED[name]
        resolved += (seen[1] if seen[0] is attr else _resolve(name, attr, bound[1]),)
    return resolved
