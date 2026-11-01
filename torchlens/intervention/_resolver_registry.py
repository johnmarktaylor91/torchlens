"""Function registry keys and their trust-gated resolution at the execution boundary.

Split from :mod:`torchlens._selection.resolver`, which re-exports every name here that callers
import from it.
"""

from __future__ import annotations

import importlib
import operator
from collections.abc import Callable, Collection
from typing import Any, Literal, cast

import torch

from .._errors import InvalidArgumentError
from .._utils._callable_safety import (
    _DENIED_MODULES,
    _matches,
    _unwrap_capture_wrapper,
    is_denied_operator_gadget,
    is_denied_stdlib_or_builtin_module,
    is_inert_first_party_callable,
    is_pure_forward_callable,
    real_callable_module,
    unsafe_callable_reason,
)
from .._utils._torch_compat import (
    get_variable_functions_class,
    resolve_runnable_torch_alias,
)
from .._utils._torch_symbols import torch_attr
from .errors import (
    ReplayPreconditionError,
    UntrustedCallableError,
)
from .types import FunctionRegistryKey

_TORCH_INTERNAL_BUILTIN_NAMESPACE = "torch._C._VariableFunctionsClass"


def _internal_torch_builtin_key(
    func: Callable[..., Any],
    name: str,
    dispatch_kind: Literal["function", "dunder"],
) -> FunctionRegistryKey | None:
    """Return a replay key for an internal torch builtin, when ``func`` is one.

    The public ``torch`` Python wrappers sometimes lower their calls to the
    underlying ``_VariableFunctionsClass`` builtin with a different argument
    convention. Sparse runnable recipes record that lowered convention, so
    replay must keep the builtin identity instead of resolving its public
    wrapper. Direct builtin public exports retain their existing public keys.

    Parameters
    ----------
    func:
        Captured callable after TorchLens wrapper unwrapping.
    name:
        Callable's terminal name.
    dispatch_kind:
        Portable dispatch category for the callable.

    Returns
    -------
    FunctionRegistryKey | None
        Stock internal-builtin key when the callable is owned by that namespace,
        otherwise ``None``.
    """

    # r-b4 R26-2: routed through _torch_compat (HAS_VARIABLE_FUNCTIONS_CLASS) --
    # namespace drift used to silently downgrade the recorded replay key to the
    # public torch wrapper (a different argument convention); it now flips the
    # named flag so the downgrade is visible in doctor/compat.
    internal = getattr(get_variable_functions_class(), name, None)
    # r47 secD_1: resolve the public alias through ``torch_attr`` so an attacker callable ``name``
    # reads ``torch.__dict__`` directly and never fires the PEP-562 lazy ``torch.__getattr__``.
    # While capture wrappers are installed, ``torch.__dict__`` holds the TorchLens wrapper,
    # whose provenance stamp (B8-1a pickle fix) claims ``__module__ == "torch"`` for every
    # module-namespace wrapper -- the direct-builtin-export test must read the ORIGINAL.
    public = torch_attr(name)
    if callable(public):
        public = _unwrap_capture_wrapper(public)
    if internal is not func or getattr(public, "__module__", None) == "torch":
        return None
    return FunctionRegistryKey(
        "custom",
        name,
        dispatch_kind,
        import_path=f"{_TORCH_INTERNAL_BUILTIN_NAMESPACE}:{name}",
    )


def _resolve_internal_torch_builtin_key(
    key: FunctionRegistryKey,
) -> Callable[..., Any] | None:
    """Resolve a captured internal torch-builtin key without importing a module.

    Parameters
    ----------
    key:
        Saved function registry key.

    Returns
    -------
    Callable[..., Any] | None
        The in-memory builtin for a canonical internal key, or ``None`` for all
        other key shapes.
    """

    if key.namespace != "custom" or key.import_path is None:
        return None
    module_name, separator, qualname = key.import_path.partition(":")
    if (
        separator != ":"
        or module_name != _TORCH_INTERNAL_BUILTIN_NAMESPACE
        or qualname != key.qualname
    ):
        return None
    resolved = getattr(get_variable_functions_class(), qualname, None)
    return cast(Callable[..., Any], resolved) if callable(resolved) else None


def function_registry_key_from_callable(func: Callable[..., Any]) -> FunctionRegistryKey:
    """Infer a portable registry key from a captured callable.

    Parameters
    ----------
    func:
        Callable captured during tracing.

    Returns
    -------
    FunctionRegistryKey
        Registry key using known namespaces where possible and import refs for
        custom callables.
    """

    module = getattr(func, "__module__", "") or ""
    qualname = getattr(func, "__qualname__", None) or getattr(func, "__name__", None) or repr(func)
    name = getattr(func, "__name__", qualname.rsplit(".", maxsplit=1)[-1])
    dispatch_kind: Literal["function", "dunder"] = (
        "dunder" if str(name).startswith("__") and str(name).endswith("__") else "function"
    )

    if module == "torch":
        internal_key = _internal_torch_builtin_key(func, str(name), dispatch_kind)
        if internal_key is not None:
            return internal_key
        return FunctionRegistryKey("torch", str(name), dispatch_kind)
    if module == "torch.nn.functional":
        return FunctionRegistryKey("torch.nn.functional", str(name), dispatch_kind)
    if module == "operator":
        return FunctionRegistryKey("operator", str(name), dispatch_kind)
    stock_alias = resolve_runnable_torch_alias(f"{module}.{name}", str(torch.__version__))
    if stock_alias is not None:
        namespace, alias_qualname, _provenance = stock_alias
        alias_dispatch: Literal["function", "method", "dunder"] = (
            "method" if namespace == "torch.Tensor" else dispatch_kind
        )
        return FunctionRegistryKey(
            cast(Any, namespace),
            alias_qualname,
            alias_dispatch,
        )
    if module in {"torch._tensor", "torch.Tensor"} or (
        hasattr(torch.Tensor, str(name)) and "Tensor" in str(qualname)
    ):
        return FunctionRegistryKey("torch.Tensor", str(name), "method")

    import_path = f"{module}:{qualname}" if module else None
    return FunctionRegistryKey("custom", str(qualname), dispatch_kind, import_path=import_path)


# Extras-gated "appliance" subpackages whose ``__init__`` imports heavy FOREIGN
# third-party dependencies at IMPORT TIME (rsatoolbox / brainscore_core for
# ``torchlens.neuro``, IPython / jupyter_client for ``torchlens._notebook``).
# Mirrors ``torchlens._io._safe_unpickle._TORCHLENS_APPLIANCE_MODULES`` --
# duplicated here rather than imported to keep this security boundary free of
# cross-module import-order coupling. A bundle-supplied ``custom`` key naming
# one of these must be EXCLUDED from the "torchlens is our own code, safe to
# import + inspect" fast path in ``resolve_function_registry_key`` and instead
# receive the exact same deny-by-default / trust-opt-in treatment as a
# genuinely foreign import path: denied before import, resolved only under an
# explicit trust opt-in, decided by NAME alone -- never imported to decide it.
_TORCHLENS_APPLIANCE_MODULES: frozenset[str] = frozenset({"torchlens.neuro", "torchlens._notebook"})


def _is_torchlens_appliance_module(module: str) -> bool:
    """Return whether ``module`` is (nested under) an extras-gated appliance package."""

    return any(
        module == appliance or module.startswith(appliance + ".")
        for appliance in _TORCHLENS_APPLIANCE_MODULES
    )


def resolve_function_registry_key(
    key: FunctionRegistryKey,
    *,
    trust_custom_callables: bool = False,
    allowed_custom_callable_modules: Collection[str] | None = None,
) -> Callable[..., Any]:
    """Resolve a saved function registry key at the execution boundary.

    Loading a saved spec may retain an untrusted foreign custom key for safe
    analysis without importing it. Resolving that key for execution denies the
    import unless the caller opts into broad trust or supplies a matching
    module allowlist. TorchLens-owned custom callables are always trusted.

    Parameters
    ----------
    key:
        Saved function registry key.
    trust_custom_callables:
        Explicit execution-time permission to import a foreign custom callable
        when no allowlist is supplied. Only enable for specs from a trusted
        source.
    allowed_custom_callable_modules:
        Optional allowlist of custom callable module names. When supplied,
        custom imports must be listed even if ``trust_custom_callables=True``.

    Returns
    -------
    Callable[..., Any]
        Resolved callable.

    Raises
    ------
    InvalidArgumentError
        If a custom key is missing its ``import_path``
        (``code="custom_callable_import_path_missing"``).
    ReplayPreconditionError
        If the namespace or qualified name cannot be resolved.
    UntrustedCallableError
        If execution attempts to resolve a foreign custom callable import that
        has not been explicitly trusted.
    """

    _fixed_roots: dict[str, Any] = {
        "torch": torch,
        "torch.Tensor": torch.Tensor,
        "torch.nn.functional": torch.nn.functional,
        "operator": operator,
    }
    try:
        internal_builtin = _resolve_internal_torch_builtin_key(key)
        if internal_builtin is not None:
            if not is_pure_forward_callable(internal_builtin):
                raise UntrustedCallableError(
                    "Refusing to resolve bundle-supplied callable "
                    f"{key.import_path} ({unsafe_callable_reason(internal_builtin)}); "
                    "it is not a pure forward/tensor op and can execute side effects.",
                    code="custom_callable_not_pure",
                    import_path=key.import_path,
                )
            return internal_builtin
        if key.namespace in _fixed_roots:
            # r49 secF_1: the top-level ``torch`` root must resolve through ``torch_attr``
            # (identifier-only ``torch.__dict__`` read) so an attacker qualname (``onnx`` /
            # ``_dynamo`` / ``has_cuda``) cannot fire torch's PEP-562 lazy ``__getattr__``
            # (unrequested submodule import / deprecated ``replacement()`` shim) BEFORE the
            # purity gate below rejects it -- the co-located sibling of the runnable-load site.
            # A genuinely-missing torch attr raises ``AttributeError`` exactly as the prior
            # bare ``getattr`` did; non-torch fixed roots carry no lazy hazard.
            _root = _fixed_roots[key.namespace]
            if _root is torch:
                _resolved = torch_attr(key.qualname)
                if _resolved is None:
                    raise AttributeError(f"module 'torch' has no attribute {key.qualname!r}")
                resolved = cast(Callable[..., Any], _resolved)
            else:
                resolved = cast(Callable[..., Any], getattr(_root, key.qualname))
            # SECURITY BOUNDARY (tripwire). These fixed namespaces also expose
            # side-effecting callables -- above all torch.load / torch.save (both
            # in torch.serialization), which unpickle attacker files (RCE) or
            # write to arbitrary paths. A bundle-supplied key is UNTRUSTED, so
            # only pure, side-effect-free forward/tensor ops may resolve. Gating
            # on the wrapped-op inventory would NOT suffice (torch.load is in it).
            if not is_pure_forward_callable(resolved):
                raise UntrustedCallableError(
                    "Refusing to resolve bundle-supplied callable "
                    f"{key.namespace}.{key.qualname} ({unsafe_callable_reason(resolved)}); "
                    "it is not a pure forward/tensor op and can execute side effects.",
                    code="custom_callable_not_pure",
                    import_path=f"{key.namespace}.{key.qualname}",
                )
            return resolved
        if key.namespace == "custom":
            if not key.import_path:
                # SF-07: a malformed spec refusal is a typed configuration door
                # (sibling of resolve_import_ref's ``import_path_invalid``), not a
                # raw AttributeError laundered through the resolution wrapper.
                # InvalidArgumentError is not in the wrap-except tuple below, so
                # it propagates with its code intact.
                raise InvalidArgumentError(
                    "custom function registry key is missing import_path",
                    code="custom_callable_import_path_missing",
                    remedy=(
                        "supply the custom callable's import reference as "
                        "import_path='module:qualname' on the saved function "
                        "registry key entry"
                    ),
                    namespace=key.namespace,
                    qualname=key.qualname,
                )
            module_name, _, qualname = key.import_path.partition(":")
            # SECURITY BOUNDARY (tripwire). A bundle-supplied custom key is FOREIGN
            # arbitrary code and default-denies. The only auto-trusted custom
            # callables are TorchLens's OWN built-in intervention helpers (e.g.
            # zero_ablate/scale), keyed "custom" because they live outside the torch
            # namespaces.
            #
            # Trust is decided by the RESOLVED CALLABLE's real ``__module__``, NEVER
            # by the import-PATH string prefix. ~75 torchlens modules do
            # ``import os`` / ``import sys`` / ``import subprocess`` / ``import
            # importlib`` / ``import builtins`` at top level, so a malicious key like
            # ``torchlens._io.tlspec:os.system`` reaches ``os.system`` by walking
            # attributes off a torchlens module. That callable's real
            # ``__module__`` is ``"os"``, so it is NOT torchlens-owned and must be
            # denied. Checking only the path prefix (as an earlier version did) let
            # such a key bypass BOTH the default-deny AND an explicit strict
            # allowlist -- the round-2 RCE this guard closes.
            #
            # Importing a ``torchlens.*`` module is itself safe (it runs only our
            # already-installed code), so we may import + inspect it without a trust
            # gate to discover the resolved callable's true owner. A genuinely
            # FOREIGN module is never imported until the trust gate passes, because
            # importing it executes its top-level code.
            #
            # EXCEPTION: the extras-gated appliance packages (``torchlens.neuro`` /
            # ``torchlens._notebook``) import FOREIGN third-party dependencies at
            # import time, so a path naming one of them is NOT safe to import
            # unconditionally -- it must receive the same treatment as a
            # genuinely foreign import path below (denied before import, resolved
            # only under an explicit trust opt-in).
            path_claims_torchlens = (
                module_name == "torchlens" or module_name.startswith("torchlens.")
            ) and not _is_torchlens_appliance_module(module_name)

            def _walk_qualname(root: Any) -> Any:
                """Resolve ``qualname`` off an already-imported module root."""

                obj: Any = root
                for part in qualname.split("."):
                    obj = getattr(obj, part)
                return obj

            def _is_torchlens_owned(obj: Any) -> bool:
                """Return whether a resolved object genuinely lives under torchlens."""

                owner = str(getattr(obj, "__module__", "") or "")
                return owner == "torchlens" or owner.startswith("torchlens.")

            def _enforce_foreign_trust(resolved_module: str) -> None:
                """Default-deny a foreign callable by its REAL module identity.

                A hard denylist of dangerous modules (os / sys / subprocess /
                builtins / importlib / ctypes / shutil / socket / ... via
                ``_DENIED_MODULES``) is refused UNCONDITIONALLY -- even on the
                trust-satisfied path. Trust means "run this user recipe", NEVER
                "import os": a satisfied ``trust_custom_callables`` (or a matching
                allowlist entry) must not be able to resolve ``os:system`` and hand
                back a live ``os.system`` callable. This closes the trust-path leg of
                the r23 ``LazyImportRef(import_path="os:system", trust=True)`` RCE.
                """

                if _matches(resolved_module, _DENIED_MODULES):
                    raise UntrustedCallableError(
                        "Refusing to resolve bundle-supplied custom callable "
                        f"{key.import_path!r} from dangerous module {resolved_module!r}; "
                        "process / OS / serialization / import / dynamic-library modules "
                        "are DENIED even under trust_custom_callables or an explicit "
                        "module allowlist. Trust never authorizes importing these modules.",
                        code="custom_callable_module_denied",
                        module=resolved_module,
                        import_path=key.import_path,
                    )
                # STRUCTURAL close of the denylist-completeness class (r31): DENY any
                # STANDARD-LIBRARY / BUILTIN module (keyed on the resolved real
                # top-level package), regardless of trust or allowlist. This closes
                # the whole class the explicit denylist above kept chasing one module
                # at a time (io / _imp / zipimport / linecache / gc / mmap / ...). The
                # pure-forward ``operator`` root and non-stdlib torch / torchlens /
                # user packages are carved out inside the detector.
                if is_denied_stdlib_or_builtin_module(resolved_module):
                    raise UntrustedCallableError(
                        "Refusing to resolve bundle-supplied custom callable "
                        f"{key.import_path!r} from standard-library / builtin module "
                        f"{resolved_module!r}; stdlib and builtin modules are DENIED even "
                        "under trust_custom_callables or an explicit module allowlist. "
                        "Trust authorizes running a user recipe, never importing a "
                        "stdlib/builtin module.",
                        code="custom_callable_module_denied",
                        module=resolved_module,
                        import_path=key.import_path,
                    )
                if allowed_custom_callable_modules is not None:
                    if resolved_module not in allowed_custom_callable_modules:
                        raise UntrustedCallableError(
                            "Refusing to resolve bundle-supplied custom callable "
                            f"{key.import_path!r} from module {resolved_module!r}; it is "
                            "not in allowed_custom_callable_modules. Resolving a foreign "
                            "callable can execute arbitrary code.",
                            code="custom_callable_module_not_allowlisted",
                            module=resolved_module,
                            import_path=key.import_path,
                        )
                elif not trust_custom_callables:
                    # Names the denied import (R65): with several custom callables in
                    # one spec, the user cannot build the recommended
                    # allowed_custom_callable_modules allowlist without knowing WHICH
                    # module was denied here.
                    raise UntrustedCallableError(
                        "Refusing to resolve bundle-supplied custom callable "
                        f"{key.import_path!r} (module {resolved_module!r}) because "
                        "importing/resolving it can execute arbitrary code. Pass "
                        "trust_custom_callables=True only for a trusted spec, or supply "
                        f"allowed_custom_callable_modules={{{resolved_module!r}}}.",
                        code="custom_callable_untrusted",
                        module=resolved_module,
                        import_path=key.import_path,
                    )

            if path_claims_torchlens:
                # Safe to import + inspect: a torchlens module is our own code.
                module = importlib.import_module(module_name)
                obj = _walk_qualname(module)
                if not _is_torchlens_owned(obj):
                    # Reached a non-torchlens callable (e.g. os.system) by walking
                    # attributes off a torchlens module. Deny by the callable's REAL
                    # module -- the torchlens import path grants it nothing.
                    _enforce_foreign_trust(str(getattr(obj, "__module__", "") or module_name))
                    # PURITY PARITY (secE-1). ``_enforce_foreign_trust`` gates on the
                    # RESOLVED-owner STRING, but that string is blind to two
                    # side-effecting families reachable by walking off a torchlens
                    # module: (a) a torch builtin (``torch.from_file`` /
                    # ``torch.compile``) whose real ``__module__`` is the bare,
                    # non-denied ``"torch"``, and (b) a C tensor method
                    # (``Tensor.apply_`` / ``resize_`` / ``set_``) whose
                    # ``__module__ is None`` so the ``or module_name`` fallback lands
                    # on the (benign) torchlens import path. The sibling
                    # genuinely-foreign branch below routes torch owners through
                    # ``is_pure_forward_callable``; mirror that gate here on the REAL
                    # object identity (NOT the fallback string) so a foreign callable
                    # walked off a torchlens path is held to the SAME purity contract
                    # even under trust. ``is_pure_forward_callable`` covers torch
                    # name/purity, the operator name-allowlist, the stdlib/denylist,
                    # and the ``__module__ is None`` tensor-method case -- closing the
                    # ``or module_name`` fallback loophole and collapsing both foreign
                    # sub-branches onto one purity gate.
                    if not is_pure_forward_callable(obj):
                        raise UntrustedCallableError(
                            "Refusing bundle-supplied custom callable "
                            f"{module_name}:{qualname}: it walks off a torchlens module "
                            f"onto a non-torchlens callable ({unsafe_callable_reason(obj)}) "
                            "that is not a pure forward/tensor op; only pure "
                            "forward/tensor ops resolve from a torchlens-path walk onto "
                            "a foreign callable, even under trust.",
                            code="custom_callable_not_pure",
                            module=module_name,
                            import_path=f"{module_name}:{qualname}",
                        )
                elif not is_inert_first_party_callable(obj):
                    # Defense-in-depth (mirrors the r21 bundle-unpickler narrowing):
                    # a genuinely torchlens-owned callable is auto-trusted ONLY if it
                    # is a vetted-inert first-party symbol (public facet recipe /
                    # transform / intervention helper). A PRIVATE torchlens util --
                    # notably the ``torchlens._utils:_module_is_installed`` import
                    # gadget -- or any I/O / import / exec / spawn callable is refused
                    # rather than resolved to a live callable.
                    raise UntrustedCallableError(
                        "Refusing to resolve torchlens-owned callable "
                        f"{module_name}:{qualname}; only public, side-effect-free "
                        "first-party callables (facet recipes / transforms / "
                        "intervention helpers) are auto-trusted. Private utilities "
                        "and I/O / import / exec callables are denied.",
                        code="custom_callable_private_first_party",
                        module=module_name,
                        import_path=f"{module_name}:{qualname}",
                    )
            else:
                # Genuinely foreign import path: gate BEFORE importing, because the
                # import itself executes the untrusted module's top-level code.
                _enforce_foreign_trust(module_name)
                module = importlib.import_module(module_name)
                obj = _walk_qualname(module)
                # RE-ENFORCE the DENYLIST on the RESOLVED callable's REAL module
                # identity, NEVER the import-PATH string. A DOTTED qualname
                # attribute-walks off the imported module and can land on a callable
                # from a DIFFERENT, denied module: ``torch:os.system`` passes the
                # pre-import gate on the non-denied root ``torch`` yet resolves
                # ``os.system`` (real module ``posix``), and ``torch:serialization.load``
                # resolves ``torch.load`` (real module ``torch.serialization``). Both
                # are hard-denied here on the resolved owner. We re-enforce the
                # DENYLIST (never hand back a process / OS / serialization / import
                # callable, even under trust) but NOT the allowlist: the allowlist
                # governs which MODULES may be IMPORTED (already enforced pre-import on
                # ``module_name``), and the resolved ``__module__`` is an implementation
                # detail -- ``operator:neg`` legitimately resolves ``operator.neg`` whose
                # real module is the C accelerator ``_operator``, so re-checking the
                # allowlist on the resolved owner would wrongly deny it.
                resolved_owner = str(getattr(obj, "__module__", "") or module_name)
                if _matches(resolved_owner, _DENIED_MODULES):
                    raise UntrustedCallableError(
                        "Refusing bundle-supplied custom callable whose RESOLVED real "
                        f"module {resolved_owner!r} is a dangerous (process / OS / "
                        "serialization / import) module reached by attribute-walking a "
                        f"dotted qualname off {module_name!r}; denied even under trust.",
                        code="custom_callable_module_denied",
                        module=resolved_owner,
                        import_path=f"{module_name}:{qualname}",
                    )
                # STRUCTURAL stdlib/builtin close (r31): a dotted qualname can walk OFF
                # a permitted user/torch module and land on a stdlib/builtin callable
                # (e.g. a trusted ``mymod:io.open`` resolving ``io.open``, real module
                # ``io``). ``_DENIED_MODULES`` above never enumerates every such owner;
                # deny the whole stdlib/builtin class on the resolved real module.
                if is_denied_stdlib_or_builtin_module(resolved_owner):
                    raise UntrustedCallableError(
                        "Refusing bundle-supplied custom callable whose RESOLVED real "
                        f"module {resolved_owner!r} is a standard-library / builtin "
                        f"module reached by attribute-walking a dotted qualname off "
                        f"{module_name!r}; denied even under trust.",
                        code="custom_callable_module_denied",
                        module=resolved_owner,
                        import_path=f"{module_name}:{qualname}",
                    )
                # PURITY PARITY (secE-1 / secE-r36-1). A callable that walked BACK into
                # the torch namespace via a dotted qualname -- OR a module-less C tensor
                # method (``resize_`` / ``set_`` / ``apply_`` / ``map_``) -- must be a
                # PURE forward op: ``torch`` also hosts side-effecting builtins
                # (``torch.from_file``) and the tensor-method family
                # rebinds/reallocates storage or runs an arbitrary callable per element.
                # This gate keys on the callable's REAL (capture-unwrapped) module, NEVER
                # ``resolved_owner``: that fallback string is spoofed two ways -- it lands
                # on the trusted ``module_name`` for a module-less tensor method (real
                # ``__module__ is None``), and on ``"torchlens.backends.torch.wrappers"``
                # for ANY torch op TorchLens has capture-wrapped (the near-universal live
                # state) -- so a raw string torch/prefix check misses a wrapped
                # ``resize_`` / ``torch.load`` (the r36 hole). Mirror the r35
                # torchlens-walk fix on the REAL identity: when the real owner is
                # module-less OR torch, hold it to ``is_pure_forward_callable`` (which
                # unwraps + covers the tensor-method-descriptor case, the storage-unsafe /
                # ``apply_`` / ``map_`` name guards, and torch name/purity). Genuinely
                # foreign (non-torch) trusted recipes carry a real, non-torch
                # ``__module__`` and are NOT subject to this gate -- trust means "run this
                # user recipe".
                real_owner = real_callable_module(obj) if callable(obj) else ""
                if (
                    callable(obj)
                    and (
                        real_owner == "" or real_owner == "torch" or real_owner.startswith("torch.")
                    )
                    and not is_pure_forward_callable(obj)
                ):
                    raise UntrustedCallableError(
                        "Refusing bundle-supplied custom callable "
                        f"{module_name}:{qualname}: it resolves to a torch-namespace or "
                        f"module-less callable ({unsafe_callable_reason(obj)}) -- chiefly a "
                        "C-level tensor method such as resize_/set_/apply_/map_ or a "
                        "side-effecting torch builtin -- that is not a pure forward/tensor "
                        "op; only pure forward/tensor ops resolve from the torch namespace "
                        "or a module-less C callable, even under trust.",
                        code="custom_callable_not_pure",
                        module=real_owner or module_name,
                        import_path=f"{module_name}:{qualname}",
                    )

            # OPERATOR GADGET name-scope (r33, A-R32-1). The ``operator`` /
            # ``_operator`` root is carved out of the stdlib denial so ``operator:neg``
            # survives, but that carve-out must NOT re-admit the generic operator
            # gadgets (``attrgetter`` / ``methodcaller`` / ``call`` / ``getitem`` /
            # ``setitem`` / ``delitem`` / the in-place ``iadd`` / ``imul`` / ...
            # mutators) that enable an RCE chain (``attrgetter('__globals__')`` ->
            # ``__import__`` -> ``os.system``). Applied on ALL foreign sub-branches
            # above (torchlens-walk-to-foreign AND genuinely-foreign), where the
            # ``is_pure_forward_callable`` gate is otherwise only applied to the torch
            # namespace. When the RESOLVED real module is ``operator`` / ``_operator``,
            # require the terminal name in the pure-forward operator allowlist.
            if callable(obj) and is_denied_operator_gadget(obj):
                raise UntrustedCallableError(
                    "Refusing bundle-supplied custom callable "
                    f"{module_name}:{qualname}: it resolves to a generic operator gadget "
                    f"({unsafe_callable_reason(obj)}); only the pure arithmetic / "
                    "comparison / bitwise / index operators resolve from "
                    "operator/_operator, even under trust.",
                    code="custom_callable_not_pure",
                    module="operator",
                    import_path=f"{module_name}:{qualname}",
                )
            if not callable(obj):
                raise TypeError(f"{key.import_path!r} resolved to non-callable {obj!r}")
            return cast(Callable[..., Any], obj)
    except (AttributeError, ImportError, TypeError) as exc:
        raise ReplayPreconditionError(f"Could not resolve function registry key {key!r}") from exc

    raise ReplayPreconditionError(f"Unknown function registry namespace {key.namespace!r}")


def _import_ref_registry_key(module_name: str, qualname: str) -> FunctionRegistryKey:
    """Route an ``module:qualname`` import ref onto the shared registry-key model.

    Mirrors ``function_registry_key_from_callable`` namespace selection from the
    STRING form so a ``torch`` / ``torch.nn.functional`` / ``operator`` /
    ``torch.Tensor`` import ref lands on the always-available fixed namespaces
    (still purity-gated inside ``resolve_function_registry_key``), while any other
    module is treated as a FOREIGN ``custom`` import that default-denies.

    Parameters
    ----------
    module_name:
        Module component of the import reference (left of ``:``).
    qualname:
        Qualified-name component of the import reference (right of ``:``).

    Returns
    -------
    FunctionRegistryKey
        Registry key whose namespace decides trust the same way a callable-derived
        key would.
    """

    terminal = qualname.rsplit(".", 1)[-1]
    dispatch_kind: Literal["function", "dunder"] = (
        "dunder" if terminal.startswith("__") and terminal.endswith("__") else "function"
    )
    # Only a bare single-attribute name maps onto a fixed root (which resolves via a
    # single ``getattr`` + purity gate). A dotted qualname stays ``custom`` so it is
    # decided by the deny-by-default foreign-import gate, never smuggled onto a fixed
    # root by attribute-walking.
    if "." not in qualname:
        if module_name == "torch":
            return FunctionRegistryKey("torch", terminal, dispatch_kind)
        if module_name == "torch.nn.functional":
            return FunctionRegistryKey("torch.nn.functional", terminal, dispatch_kind)
        # Route BOTH ``operator`` and its C accelerator ``_operator`` onto the fixed,
        # name-allowlisted operator root (r33, A-R32-1): the fixed root purity-gate
        # restricts operator to ``_ALLOWED_OPERATOR_NAMES``, so ``_operator:neg`` still
        # resolves while ``_operator:attrgetter`` / ``_operator:setitem`` are DENIED --
        # instead of ``_operator:*`` falling through to the foreign ``custom`` tail
        # (which, before r33, applied no operator name filter).
        if module_name in {"operator", "_operator"}:
            return FunctionRegistryKey("operator", terminal, dispatch_kind)
    if module_name in {"torch._tensor", "torch.Tensor"} and "." not in qualname:
        return FunctionRegistryKey("torch.Tensor", terminal, "method")
    return FunctionRegistryKey(
        "custom", qualname, dispatch_kind, import_path=f"{module_name}:{qualname}"
    )


def resolve_import_ref(
    import_path: str,
    *,
    trust_custom_callables: bool = False,
    allowed_custom_callable_modules: Collection[str] | None = None,
) -> Callable[..., Any]:
    """Resolve a ``module:qualname`` import reference through the SAME trust gate.

    This is the single trust-gated resolution path for bundle-supplied import
    references. It routes the reference onto ``resolve_function_registry_key`` so it
    obeys exactly one contract: the fixed ``torch`` / ``torch.Tensor`` /
    ``torch.nn.functional`` / ``operator`` namespaces (and TorchLens-owned custom
    helpers) always resolve WITHOUT trust but are purity-gated; a genuinely FOREIGN
    module import default-denies with a typed :class:`UntrustedCallableError` and is
    NEVER imported unless the caller opts in via ``trust_custom_callables=True`` or a
    matching ``allowed_custom_callable_modules`` entry.

    Parameters
    ----------
    import_path:
        Import reference in ``module:qualname`` form.
    trust_custom_callables:
        Explicit execution-time permission to import a foreign custom callable when
        no allowlist is supplied. Enable only for a trusted spec.
    allowed_custom_callable_modules:
        Optional allowlist of custom callable module names. When supplied, foreign
        imports must be listed even if ``trust_custom_callables=True``.

    Returns
    -------
    Callable[..., Any]
        Resolved callable.

    Raises
    ------
    ValueError
        If the import reference is malformed.
    UntrustedCallableError
        If resolving a foreign custom callable has not been explicitly trusted.
    ReplayPreconditionError
        If the namespace or qualified name cannot be resolved.
    """

    module_name, separator, qualname = import_path.partition(":")
    if not separator or not module_name or not qualname:
        raise InvalidArgumentError(
            f"Invalid import path {import_path!r}",
            code="import_path_invalid",
            remedy="use the 'module:qualname' import reference form",
            import_path=import_path,
        )
    key = _import_ref_registry_key(module_name, qualname)
    return resolve_function_registry_key(
        key,
        trust_custom_callables=trust_custom_callables,
        allowed_custom_callable_modules=allowed_custom_callable_modules,
    )
