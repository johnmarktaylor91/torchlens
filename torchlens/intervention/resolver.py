"""Resolve TorchLens intervention selectors against model logs."""

from __future__ import annotations

import importlib
import operator
import warnings
from collections.abc import Callable, Collection, Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast

import torch

from .._errors import InvalidArgumentError
from ..ir.selector_eval import (
    ensure_supported,
    evaluate,
    normalize_selector_like,
    walk_selector,
)
from ..utils._callable_safety import (
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
from ..utils._torch_compat import resolve_runnable_torch_alias
from ..utils._torch_symbols import torch_attr
from ._module_alias_guard import refuse_trace_alias_spellings
from .errors import (
    MultiMatchWarning,
    ReplayPreconditionError,
    SiteAmbiguityError,
    SiteResolutionError,
    UntrustedCallableError,
)
from .selectors import (
    BaseSelector,
    CompositeSelector,
    NotSelector,
    _classify_selector_direction,
)
from .types import FrozenTargetSpec, FunctionRegistryKey, TargetSpec

if TYPE_CHECKING:
    import pandas as pd

    from torchlens.data_classes.grad_fn import GradFn
    from torchlens.data_classes.layer import Layer
    from torchlens.data_classes.op import Op
    from torchlens.data_classes.trace import Trace

SelectorInput = BaseSelector | TargetSpec | FrozenTargetSpec | str
if TYPE_CHECKING:
    Site: TypeAlias = Op | Layer | GradFn
else:
    Site: TypeAlias = Any
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

    from ..utils._torch_compat import get_variable_functions_class

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
    from ..utils._torch_compat import get_variable_functions_class

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
# ``torchlens.neuro``, IPython / jupyter_client for ``torchlens.notebook``).
# Mirrors ``torchlens._io._safe_unpickle._TORCHLENS_APPLIANCE_MODULES`` --
# duplicated here rather than imported to keep this security boundary free of
# cross-module import-order coupling. A bundle-supplied ``custom`` key naming
# one of these must be EXCLUDED from the "torchlens is our own code, safe to
# import + inspect" fast path in ``resolve_function_registry_key`` and instead
# receive the exact same deny-by-default / trust-opt-in treatment as a
# genuinely foreign import path: denied before import, resolved only under an
# explicit trust opt-in, decided by NAME alone -- never imported to decide it.
_TORCHLENS_APPLIANCE_MODULES: frozenset[str] = frozenset({"torchlens.neuro", "torchlens.notebook"})


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
            # ``torchlens.notebook``) import FOREIGN third-party dependencies at
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
                    # notably the ``torchlens.utils:_module_is_installed`` import
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


@dataclass(frozen=True)
class SiteTable:
    """Result of resolving a selector; ordered, indexable, dataframe-convertible."""

    _sites: tuple[Site, ...]
    query: Any | None = None

    def __len__(self) -> int:
        """Return the number of resolved sites.

        Returns
        -------
        int
            Number of layer-pass records in the table.
        """

        return len(self._sites)

    def __iter__(self) -> Iterator[Site]:
        """Iterate through resolved sites in execution order.

        Returns
        -------
        Iterator[Op]
            Iterator over layer-pass records.
        """

        return iter(self._sites)

    def __getitem__(self, idx: int | slice) -> Site | SiteTable:
        """Return one site or a sliced site table.

        Parameters
        ----------
        idx:
            Integer index or slice.

        Returns
        -------
        Op | SiteTable
            Single layer pass for integer indexes, table for slices.
        """

        if isinstance(idx, slice):
            return SiteTable(self._sites[idx], query=self.query)
        return self._sites[idx]

    def __repr__(self) -> str:
        """Return a compact table representation.

        Returns
        -------
        str
            Summary including count and first labels.
        """

        count = len(self)
        if count == 0:
            return "SiteTable(0 sites)"
        labels = self.labels()
        if count == 1:
            return f"SiteTable(1 site: {labels[0]})"
        if count <= 3:
            return f"SiteTable({count} sites: {', '.join(labels)})"
        prefix = ", ".join(labels[:3])
        return f"SiteTable({count} sites: {prefix}, ... {labels[-1]})"

    def where(self, predicate: Callable[[Site], bool]) -> SiteTable:
        """Filter the table with a predicate.

        Parameters
        ----------
        predicate:
            Callable receiving each layer-pass record.

        Returns
        -------
        SiteTable
            Filtered table in original execution order.
        """

        return SiteTable(tuple(site for site in self._sites if predicate(site)), query=self.query)

    def first(self) -> Site:
        """Return the first resolved site.

        Returns
        -------
        Op
            First layer-pass record.

        Raises
        ------
        SiteResolutionError
            If the table is empty.
        """

        if not self._sites:
            raise SiteResolutionError("SiteTable is empty; no first site is available.")
        return self._sites[0]

    def labels(self) -> tuple[str, ...]:
        """Return execution-order labels for resolved sites.

        Returns
        -------
        tuple[str, ...]
            Stable layer labels.
        """

        return tuple(
            str(
                getattr(
                    site,
                    "layer_label",
                    getattr(site, "label", "<unknown>"),
                )
            )
            for site in self._sites
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Return a pandas table describing resolved sites.

        Returns
        -------
        pd.DataFrame
            DataFrame with one row per resolved site.
        """
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for this feature. Install with `pip install torchlens[tabular]`."
            ) from e

        rows = [
            {
                "label": getattr(site, "label", None),
                "layer_label": getattr(site, "layer_label", None),
                "call_index": getattr(site, "call_index", None),
                "step_index": getattr(site, "step_index", None),
                "raw_index": getattr(site, "raw_index", None),
                "func_name": getattr(site, "func_name", getattr(site, "name", None)),
                "module": getattr(site, "module", None),
                "modules": getattr(site, "modules", ()),
                "output_of_module_calls": getattr(site, "output_of_module_calls", ()),
            }
            for site in self._sites
        ]
        return pd.DataFrame(rows)


def multipass_bare_label_message(layer_label: str, pass_indices: Sequence[int]) -> str:
    """Return the teaching refusal message for a bare multi-pass layer label.

    Parameters
    ----------
    layer_label:
        Bare layer label the caller supplied.
    pass_indices:
        Pass indices recorded for the layer, in execution order.

    Returns
    -------
    str
        Message naming the layer, its pass count, and every pass-qualified
        spelling the caller can address instead.
    """

    spellings = ", ".join(f"'{layer_label}:{index}'" for index in pass_indices)
    return (
        f"bare label {layer_label!r} is ambiguous on this trace: layer "
        f"{layer_label!r} ran {len(pass_indices)} passes, and each pass is a "
        "distinct op with its own activation. Bare layer labels address only "
        "single-pass layers. Remedy: address one pass with a pass-qualified "
        f"label ({spellings}), or select every pass explicitly with the Layer "
        f"selection (log[{layer_label!r}].__selection__())."
    )


def _refuse_bare_multipass_label(selector: BaseSelector, matched: Sequence[Site]) -> None:
    """Refuse an exact-label query that spans several passes of one layer.

    A label selector addresses ONE op; when its spelling is layer-wide on a
    multi-pass (recurrence-grouped) layer it matches every pass, and any
    single-op consumer would have to guess a pass — the exact silent
    wrong-pass corruption the replay engine refuses. Predicate selectors
    (``tl.func``, module selectors, ...) keep their fan-out semantics.

    Raises
    ------
    SiteAmbiguityError
        With ``fields["code"] == "multipass_bare_label_ambiguous"``.
    """

    selector_kind = getattr(selector, "selector_kind", None)
    if len(matched) <= 1 or selector_kind not in ("label", "contains"):
        return
    layer_labels = {getattr(site, "layer_label", None) for site in matched}
    if len(layer_labels) != 1:
        return
    layer_label = next(iter(layer_labels))
    if not isinstance(layer_label, str):
        return
    if selector_kind == "contains" and getattr(selector, "selector_value", None) != layer_label:
        # A genuine substring pattern keeps its documented fan-out semantics;
        # only the exact bare layer label is the pass-ambiguous address.
        return
    pass_indices = sorted(int(getattr(site, "pass_index", 1) or 1) for site in matched)
    raise SiteAmbiguityError(
        multipass_bare_label_message(layer_label, pass_indices),
        code="multipass_bare_label_ambiguous",
        layer_label=layer_label,
        pass_indices=tuple(pass_indices),
    )


def resolve_sites(
    log: Trace,
    query: SelectorInput,
    *,
    strict: bool = False,
    max_fanout: int = 8,
) -> SiteTable:
    """Resolve a selector or accepted shorthand against a model log.

    Parameters
    ----------
    log:
        Model log whose captured layer-pass records should be searched.
    query:
        Selector, target spec, frozen target spec, or non-strict bare string.
    strict:
        Whether to reject non-portable query forms.
    max_fanout:
        Maximum number of sites allowed. ``None`` is rejected in this MVP.

    Returns
    -------
    SiteTable
        Ordered table of resolved sites.

    Raises
    ------
    SiteAmbiguityError
        If more sites resolve than ``max_fanout`` allows.
    SiteResolutionError
        If no site resolves or the query shape is unsupported.
    """

    if max_fanout is None:
        raise SiteResolutionError("max_fanout=None is not supported; pass an explicit integer.")
    if max_fanout < 1:
        raise SiteAmbiguityError("max_fanout must be at least 1.")
    if strict and isinstance(query, str):
        raise SiteResolutionError(
            "Bare strings are non-portable in strict mode; use tl.label(...), "
            "tl.func(...), or another typed selector."
        )

    selector = _normalize_query(query)
    _guard_episode_step_query(log, selector)
    refuse_trace_alias_spellings(log, selector)
    direction = _selector_resolution_direction(selector)
    sites = tuple(_iter_sites(log, direction))
    matched = _resolve_unchecked(sites, selector, strict=strict)
    if not matched:
        raise SiteResolutionError(
            f"selector {query!r} matched 0 sites. Use log.find_sites(...) to discover labels."
        )
    _refuse_bare_multipass_label(selector, matched)
    if len(matched) > max_fanout:
        raise SiteAmbiguityError(
            f"site {query!r} matched {len(matched)} sites, exceeding max_fanout={max_fanout}. "
            "Pass a larger max_fanout explicitly or use a narrower selector."
        )
    if len(matched) > 1:
        warnings.warn(
            _multi_match_message(query, matched),
            MultiMatchWarning,
            stacklevel=2,
        )
    return SiteTable(matched, query=query)


def _multi_match_message(query: SelectorInput, matched: Sequence[Site]) -> str:
    """Return the multi-match warning text for a resolved site set.

    A set holding a synthetic model-output alias (``output_N``) together with
    the op it aliases does not fan out: a value edit applied at both sites
    compounds on the one returned value, so the warning says so.

    Parameters
    ----------
    query:
        Selector input as the caller spelled it.
    matched:
        Resolved sites, in execution order.

    Returns
    -------
    str
        Warning message.
    """

    matched_labels = {getattr(site, "layer_label", None) for site in matched}
    pairs = [
        (str(getattr(site, "layer_label", None)), str(parent))
        for site in matched
        if getattr(site, "is_output", False)
        for parent in (getattr(site, "parents", ()) or ())
        if parent in matched_labels
    ]
    if not pairs:
        return f"selector {query!r} matched {len(matched)} sites and will fan out."
    named = ", ".join(f"{alias!r} aliases {producer!r}" for alias, producer in pairs)
    return (
        f"selector {query!r} matched {len(matched)} sites, including a model-output alias "
        f"and the op it aliases ({named}); a value edit applied at both compounds on the "
        "same returned value instead of reaching independent sites. Narrow the selector to "
        "the producing op."
    )


def find_sites(
    log: Trace,
    query: SelectorInput,
    *,
    strict: bool = False,
    max_fanout: int = 8,
) -> SiteTable:
    """Find matching sites and return a table.

    Parameters
    ----------
    log:
        Model log whose captured layer-pass records should be searched.
    query:
        Selector, target spec, frozen target spec, or non-strict bare string.
    strict:
        Whether to reject non-portable query forms.
    max_fanout:
        Maximum number of sites allowed.

    Returns
    -------
    SiteTable
        Ordered table of resolved sites.
    """

    if max_fanout is None:
        raise SiteResolutionError("max_fanout=None is not supported; pass an explicit integer.")
    if max_fanout < 1:
        raise SiteAmbiguityError("max_fanout must be at least 1.")
    if strict and isinstance(query, str):
        raise SiteResolutionError(
            "Bare strings are non-portable in strict mode; use tl.label(...), "
            "tl.func(...), or another typed selector."
        )

    selector = _normalize_query(query)
    _guard_episode_step_query(log, selector)
    refuse_trace_alias_spellings(log, selector)
    direction = _selector_resolution_direction(selector)
    sites = tuple(_iter_sites(log, direction))
    matched = _resolve_unchecked(sites, selector, strict=strict)
    if len(matched) > max_fanout:
        raise SiteAmbiguityError(
            f"site {query!r} matched {len(matched)} sites, exceeding max_fanout={max_fanout}. "
            "Pass a larger max_fanout explicitly or use a narrower selector."
        )
    return SiteTable(matched, query=query)


def _guard_episode_step_query(log: Trace, selector: Any) -> None:
    """Door guard for post-hoc episode-step queries (lane F42).

    A step-qualified selector resolving against a product with no step axis
    would match NOTHING silently -- the inverse of the fires-at-every-step
    wrongness the qualifier exists to close -- so the door refuses typed at
    the point of failure instead: plain captures have no steps to qualify,
    and pre-F42 episode artifacts carry no ``Op.episode_step`` stamps.

    Parameters
    ----------
    log:
        Model log the query resolves against.
    selector:
        Normalized query selector.

    Raises
    ------
    SiteResolutionError
        ``episode_step_selector_without_episode`` on plain captures;
        ``episode_step_unstamped`` on stamp-less episode artifacts.
    """

    from ..ir.selector_eval import selector_contains_kind

    if not isinstance(selector, BaseSelector) or not selector_contains_kind(
        selector, "episode_step", unwrap=True
    ):
        return
    from ..capture._episode_ledger import capture_kind_for

    if capture_kind_for(log) != "episode":
        raise SiteResolutionError(
            "at_step(...) names an episode step, but this product carries no "
            "episode declaration: there are no steps to qualify. Re-capture "
            "with tl.trace(model, x, episode=tl.options.EpisodeSpec(...)), or "
            "drop the step qualifier.",
            code="episode_step_selector_without_episode",
            remedy="re-capture with episode=, or drop at_step()",
        )
    if not any(getattr(op, "episode_step", None) is not None for op in log.layer_list):
        raise SiteResolutionError(
            "at_step(...) needs the per-op episode-step stamps "
            "(Op.episode_step), but no op on this episode product carries "
            "one -- a pre-stamping artifact. Re-capture (or re-save from a "
            "fresh capture) with this TorchLens version to mint the stamps.",
            code="episode_step_unstamped",
            remedy="re-capture the episode with a stamping TorchLens version",
        )


def _iter_layer_ops(log: Trace) -> Sequence[Op]:
    """Return final layer ops from a completed model log.

    Parameters
    ----------
    log:
        Model log to inspect.

    Returns
    -------
    Sequence[Op]
        Execution-order layer-pass records.

    Raises
    ------
    SiteResolutionError
        If the model log has not completed postprocessing.
    """

    if not getattr(log, "_tracing_finished", False):
        raise SiteResolutionError("Sites can only be resolved after the forward pass is complete.")
    return log.layer_list


def _iter_sites(log: Trace, direction: Literal["forward", "backward"]) -> Sequence[Site]:
    """Return candidate sites for the requested graph direction.

    Parameters
    ----------
    log:
        Model log to inspect.
    direction:
        Site universe to return.

    Returns
    -------
    Sequence[Site]
        Forward op sites or backward grad_fn_handle sites.
    """

    if direction == "forward":
        return _iter_layer_ops(log)
    if not getattr(log, "has_backward_pass", False):
        raise SiteResolutionError("Backward selectors require log_backward() to run first.")
    return tuple(log.grad_fn_logs.values())


def _resolve_unchecked(
    sites: Sequence[Site],
    query: SelectorInput,
    *,
    strict: bool,
) -> tuple[Site, ...]:
    """Resolve without zero/multi fanout validation.

    Parameters
    ----------
    sites:
        Layer-pass records to search.
    query:
        Selector input.
    strict:
        Whether strict portability rules are active.

    Returns
    -------
    tuple[Op, ...]
        Matching sites in execution order.
    """

    selector = _normalize_query(query)
    if strict and any(
        isinstance(node, BaseSelector) and node.selector_kind == "predicate"
        for node in walk_selector(selector)
    ):
        raise SiteResolutionError(
            "tl.where(...) predicate selectors are non-portable in strict mode."
        )
    ensure_supported(selector, lifecycle="site")
    return tuple(site for site in sites if evaluate(selector, site, lifecycle="site"))


def _selector_resolution_direction(query: SelectorInput) -> Literal["forward", "backward"]:
    """Pick the resolver search universe for a selector query.

    Parameters
    ----------
    query:
        Selector query to inspect recursively.

    Returns
    -------
    Literal["forward", "backward"]
        Direction whose site universe should be searched.
    """

    selector = _normalize_query(query)

    def _walk(sel: BaseSelector) -> tuple[bool, bool]:
        """Return whether a selector tree contains backward or forward selectors."""

        if isinstance(sel, CompositeSelector):
            has_back = False
            has_forward = False
            for child in sel.selectors:
                child_selector = _normalize_query(child)
                child_back, child_forward = _walk(child_selector)
                has_back = has_back or child_back
                has_forward = has_forward or child_forward
            return has_back, has_forward
        if isinstance(sel, NotSelector):
            return _walk(_normalize_query(sel.selector))
        direction = _classify_selector_direction(sel)
        return direction == "backward", direction == "forward"

    has_backward, has_forward = _walk(selector)
    if has_backward:
        return "backward"
    if has_forward:
        return "forward"
    return "forward"


def _normalize_query(query: SelectorInput) -> BaseSelector:
    """Normalize a query object to a selector.

    Parameters
    ----------
    query:
        Supported selector input.

    Returns
    -------
    BaseSelector
        Normalized selector object.

    Raises
    ------
    SiteResolutionError
        If the query shape is unsupported.
    """

    return normalize_selector_like(query, lifecycle="site")


__all__ = ["SiteTable", "_selector_resolution_direction", "find_sites", "resolve_sites"]
