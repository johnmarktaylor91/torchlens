"""Resolve TorchLens intervention selectors against model logs."""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterator, Sequence
from dataclasses import dataclass
import importlib
import operator
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast
import warnings

import torch

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
    in_module,
)
from .types import FrozenTargetSpec, FunctionRegistryKey, TargetSpec
from ..ir.container import DataclassField, DictKey, HFKey, NamedField, TupleIndex
from ..ir.container_registry import Role
from ..utils._callable_safety import (
    _DENIED_MODULES,
    _matches,
    is_denied_stdlib_or_builtin_module,
    is_inert_first_party_callable,
    is_pure_forward_callable,
    unsafe_callable_reason,
)
from ..utils._torch_compat import resolve_runnable_torch_alias

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
DIRECTION_AGNOSTIC_KINDS = frozenset(
    {
        "label",
        "func",
        "in_module",
        "module",
        "contains",
        "regex",
        "predicate",
        "func_transform",
        # These three describe a forward op's output/input container position and
        # carry no direction of their own; the selector layer
        # (``_classify_selector_direction``) already treats them as agnostic, so
        # they must be listed here too. Otherwise a backward (GradFn) site never
        # bridges to its paired forward op for these kinds and a validly-composed
        # selector like ``output_at(0) & grad_input()`` silently resolves to 0
        # sites instead of intersecting meaningfully.
        "output",
        "output_at",
        "input_at",
    }
)

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

    internal = getattr(getattr(torch._C, "_VariableFunctionsClass", None), name, None)
    public = getattr(torch, name, None)
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
    resolved = getattr(getattr(torch._C, "_VariableFunctionsClass", None), qualname, None)
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
                    "it is not a pure forward/tensor op and can execute side effects."
                )
            return internal_builtin
        if key.namespace in _fixed_roots:
            resolved = cast(Callable[..., Any], getattr(_fixed_roots[key.namespace], key.qualname))
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
                    "it is not a pure forward/tensor op and can execute side effects."
                )
            return resolved
        if key.namespace == "custom":
            if not key.import_path:
                raise AttributeError("custom key is missing import_path")
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
                        "Refusing to resolve bundle-supplied custom callable from "
                        f"dangerous module {resolved_module!r}; process / OS / "
                        "serialization / import / dynamic-library modules are DENIED "
                        "even under trust_custom_callables or an explicit module "
                        "allowlist. Trust never authorizes importing these modules."
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
                        "Refusing to resolve bundle-supplied custom callable from "
                        f"standard-library / builtin module {resolved_module!r}; stdlib "
                        "and builtin modules are DENIED even under trust_custom_callables "
                        "or an explicit module allowlist. Trust authorizes running a "
                        "user recipe, never importing a stdlib/builtin module."
                    )
                if allowed_custom_callable_modules is not None:
                    if resolved_module not in allowed_custom_callable_modules:
                        raise UntrustedCallableError(
                            "Refusing to resolve bundle-supplied custom callable "
                            f"from module {resolved_module!r}; it is not in "
                            "allowed_custom_callable_modules. Resolving a foreign "
                            "callable can execute arbitrary code."
                        )
                elif not trust_custom_callables:
                    raise UntrustedCallableError(
                        "Refusing to resolve bundle-supplied custom callable because "
                        "importing/resolving it can execute arbitrary code. Pass "
                        "trust_custom_callables=True only for a trusted spec, or "
                        "supply allowed_custom_callable_modules."
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
                        "and I/O / import / exec callables are denied."
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
                        f"dotted qualname off {module_name!r}; denied even under trust."
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
                        f"{module_name!r}; denied even under trust."
                    )
                # A callable that walked BACK into the torch namespace via a dotted
                # qualname must be a PURE forward op: ``torch`` also hosts
                # side-effecting builtins (``torch.from_file``) whose real module is
                # the bare, non-denied ``"torch"``. Parity with
                # ``runnable_load._finalize_resolution``'s ``is_pure_forward_callable``
                # gate. Genuinely foreign (non-torch) trusted callables are NOT
                # subject to this gate -- trust means "run this user recipe".
                if callable(obj) and (
                    resolved_owner == "torch" or resolved_owner.startswith("torch.")
                ):
                    if not is_pure_forward_callable(obj):
                        raise UntrustedCallableError(
                            "Refusing bundle-supplied custom callable "
                            f"{module_name}:{qualname}: it resolves to a side-effecting "
                            f"torch callable ({unsafe_callable_reason(obj)}) reached via "
                            "a dotted qualname; only pure forward/tensor ops resolve "
                            "from the torch namespace."
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
        if module_name == "operator":
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
        raise ValueError(f"Invalid import path {import_path!r}")
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

    def __getitem__(self, idx: int | slice) -> "Site | SiteTable":
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

    def where(self, predicate: Callable[[Site], bool]) -> "SiteTable":
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

    def to_dataframe(self) -> "pd.DataFrame":
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


def resolve_sites(
    log: "Trace",
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
    direction = _selector_resolution_direction(selector)
    sites = tuple(_iter_sites(log, direction))
    matched = _resolve_unchecked(sites, selector, strict=strict)
    if not matched:
        raise SiteResolutionError(
            f"selector {query!r} matched 0 sites. Use log.find_sites(...) to discover labels."
        )
    if len(matched) > max_fanout:
        raise SiteAmbiguityError(
            f"site {query!r} matched {len(matched)} sites, exceeding max_fanout={max_fanout}. "
            "Pass a larger max_fanout explicitly or use a narrower selector."
        )
    if len(matched) > 1:
        warnings.warn(
            f"selector {query!r} matched {len(matched)} sites and will fan out.",
            MultiMatchWarning,
            stacklevel=2,
        )
    return SiteTable(matched, query=query)


def find_sites(
    log: "Trace",
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
    direction = _selector_resolution_direction(selector)
    sites = tuple(_iter_sites(log, direction))
    matched = _resolve_unchecked(sites, selector, strict=strict)
    if len(matched) > max_fanout:
        raise SiteAmbiguityError(
            f"site {query!r} matched {len(matched)} sites, exceeding max_fanout={max_fanout}. "
            "Pass a larger max_fanout explicitly or use a narrower selector."
        )
    return SiteTable(matched, query=query)


def _iter_layer_ops(log: "Trace") -> Sequence["Op"]:
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


def _iter_sites(log: "Trace", direction: Literal["forward", "backward"]) -> Sequence[Site]:
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
    kind = selector.selector_kind
    value = selector.selector_value

    if strict and kind == "predicate":
        raise SiteResolutionError(
            "tl.where(...) predicate selectors are non-portable in strict mode."
        )

    if kind == "and" and isinstance(selector, CompositeSelector):
        left, right = selector.selectors
        left_site_ids = {id(site) for site in _resolve_unchecked(sites, left, strict=strict)}
        right_site_ids = {id(site) for site in _resolve_unchecked(sites, right, strict=strict)}
        return tuple(
            site for site in sites if id(site) in left_site_ids and id(site) in right_site_ids
        )
    if kind == "or" and isinstance(selector, CompositeSelector):
        left, right = selector.selectors
        left_site_ids = {id(site) for site in _resolve_unchecked(sites, left, strict=strict)}
        right_site_ids = {id(site) for site in _resolve_unchecked(sites, right, strict=strict)}
        return tuple(
            site for site in sites if id(site) in left_site_ids or id(site) in right_site_ids
        )
    if kind == "not" and isinstance(selector, NotSelector):
        excluded_site_ids = {
            id(site) for site in _resolve_unchecked(sites, selector.selector, strict=strict)
        }
        return tuple(site for site in sites if id(site) not in excluded_site_ids)

    if kind == "label":
        return tuple(site for site in sites if _resolve_site_kind(site, kind, value))
    if kind == "func":
        return tuple(site for site in sites if _resolve_site_kind(site, kind, value))
    if kind == "func_transform":
        return tuple(site for site in sites if _resolve_site_kind(site, kind, value))
    if kind == "module":
        return tuple(site for site in sites if _resolve_site_kind(site, kind, value))
    if kind == "output":
        return tuple(site for site in sites if _resolve_site_kind(site, kind, value))
    if kind == "output_at":
        return tuple(site for site in sites if _resolve_site_kind(site, kind, value))
    if kind == "input_at":
        return tuple(site for site in sites if _resolve_site_kind(site, kind, value))
    if kind == "contains":
        return tuple(site for site in sites if _resolve_site_kind(site, kind, value))
    if kind == "regex":
        return tuple(site for site in sites if _resolve_site_kind(site, kind, value))
    if kind == "in_module":
        return tuple(site for site in sites if _resolve_site_kind(site, kind, value))
    if kind == "predicate":
        return tuple(site for site in sites if _resolve_site_kind(site, kind, value))
    if kind in {
        "grad_fn",
        "grad_fn_handle",
        "intervening",
        "without_op",
        "label",
        "grad_kind",
        "backward_pass",
    }:
        return tuple(site for site in sites if _resolve_site_kind(site, kind, value))

    raise SiteResolutionError(f"Unsupported selector kind {kind!r}.")


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


def _resolve_site_kind(site: Site, kind: str, value: Any) -> bool:
    """Return whether one site matches a simple selector kind.

    Parameters
    ----------
    site:
        Candidate forward or backward site.
    kind:
        Selector kind.
    value:
        Selector payload.

    Returns
    -------
    bool
        Whether the selector matches.
    """

    from torchlens.data_classes.grad_fn import GradFn

    if isinstance(site, GradFn):
        if kind in DIRECTION_AGNOSTIC_KINDS:
            return _resolve_grad_fn_forward_kind(site, kind, value)
        return _resolve_grad_fn_kind(site, kind, value)

    if kind == "label":
        return _label_matches(site, str(value))
    if kind == "func":
        if isinstance(value, dict):
            return site.func_name == value.get("name") and _output_matches(
                site, value.get("output")
            )
        return site.func_name == value
    if kind == "func_transform":
        if not bool(getattr(site, "is_transform", False)):
            return False
        if value is None:
            return True
        transform_kind = getattr(site, "transform_kind", None)
        if transform_kind is None:
            return False
        normalized_site = str(transform_kind).lower().replace("_", "").replace(".", "")
        normalized_value = str(value).lower().replace("_", "").replace(".", "")
        return normalized_site == normalized_value
    if kind == "output":
        return _output_matches(site, value)
    if kind == "output_at":
        return _output_path_matches(tuple(getattr(site, "container_path", ()) or ()), tuple(value))
    if kind == "input_at":
        return _input_path_matches(site, tuple(value))
    if kind == "module":
        return _module_output_matches(site, str(value))
    if kind == "contains":
        return str(value).lower() in str(site.layer_label).lower()
    if kind == "regex":
        import re as _re

        return _re.search(str(value), str(site.layer_label)) is not None
    if kind == "in_module":
        return bool(in_module(site, str(value)))
    if kind == "predicate":
        predicate, _name_hint = _predicate_payload(value)
        return bool(predicate(site))
    return False


def _resolve_grad_fn_forward_kind(site: "GradFn", kind: str, value: Any) -> bool:
    """Return whether a GradFn matches a forward selector through its op aliases.

    Parameters
    ----------
    site:
        Candidate grad_fn_handle log.
    kind:
        Direction-agnostic selector kind.
    value:
        Selector payload.

    Returns
    -------
    bool
        Whether the selector matches the paired op or a synthetic boundary alias
        sharing the same autograd ``grad_fn`` identity.
    """

    if site.op is not None and _resolve_site_kind(site.op, kind, value):
        return True
    for alias in _grad_fn_boundary_aliases(site):
        if _resolve_site_kind(alias, kind, value):
            return True
    return False


def _grad_fn_boundary_aliases(site: "GradFn") -> tuple[Site, ...]:
    """Return input/output alias ops sharing a GradFn identity with ``site``.

    Parameters
    ----------
    site:
        Candidate grad_fn_handle log.

    Returns
    -------
    tuple[Site, ...]
        Synthetic input/output ops whose ``grad_fn_object_id`` equals the
        GradFn's object id, excluding the already-paired op.
    """

    trace = site.source_trace
    if trace is None:
        return ()
    paired_label = site.op_label
    aliases: list[Site] = []
    for layer in getattr(trace, "layer_list", ()):
        if getattr(layer, "layer_label", None) == paired_label:
            continue
        if getattr(layer, "grad_fn_object_id", None) != site.grad_fn_object_id:
            continue
        if not (getattr(layer, "is_input", False) or getattr(layer, "is_output", False)):
            continue
        aliases.append(layer)
    return tuple(aliases)


def _resolve_grad_fn_kind(site: "GradFn", kind: str, value: Any) -> bool:
    """Return whether one grad_fn_handle site matches a backward-only selector.

    Parameters
    ----------
    site:
        Candidate grad_fn_handle log.
    kind:
        Backward selector kind.
    value:
        Selector payload.

    Returns
    -------
    bool
        Whether the selector matches.
    """

    if kind in {"intervening", "without_op"}:
        return not site.has_op
    if kind == "label":
        return site.label == str(value)
    if kind == "grad_fn":
        payload = value if isinstance(value, dict) else {}
        type = payload.get("type")
        label_pattern = payload.get("grad_fn_label_pattern")
        is_custom = payload.get("is_custom")
        if type is not None and not _grad_fn_type_matches(site, str(type)):
            return False
        if label_pattern is not None and str(label_pattern) not in site.label:
            return False
        if is_custom is not None and bool(site.is_custom) is not bool(is_custom):
            return False
        return True
    if kind == "grad_kind":
        return _grad_fn_has_saved_grad_kind(site, str(value))
    if kind == "backward_pass":
        return _grad_fn_has_backward_pass(site, int(value))
    return False


def _grad_fn_has_saved_grad_kind(site: "GradFn", grad_kind: str) -> bool:
    """Return whether a grad_fn has calls with the requested saved grad tuple.

    Parameters
    ----------
    site:
        Candidate grad_fn_handle log.
    grad_kind:
        ``"grad_input"`` or ``"grad_output"``.

    Returns
    -------
    bool
        Whether at least one call has the requested tuple saved.
    """

    field_name = "grad_inputs" if grad_kind == "grad_input" else "grad_outputs"
    return any(getattr(call, field_name, None) is not None for call in _grad_fn_call_values(site))


def _grad_fn_has_backward_pass(site: "GradFn", pass_index: int) -> bool:
    """Return whether a grad_fn participates in a backward pass.

    Parameters
    ----------
    site:
        Candidate grad_fn_handle log.
    pass_index:
        One-based global backward pass number.

    Returns
    -------
    bool
        Whether the grad_fn has a call in the requested pass.
    """

    return any(
        getattr(call, "backward_pass_index", None) == pass_index
        for call in _grad_fn_call_values(site)
    )


def _grad_fn_call_values(site: "GradFn") -> tuple[Any, ...]:
    """Return GradFnCall values from dict or accessor-backed call storage.

    Parameters
    ----------
    site:
        Candidate grad_fn_handle log.

    Returns
    -------
    tuple[Any, ...]
        Recorded GradFnCall values.
    """

    calls = site.calls
    if isinstance(calls, dict):
        return tuple(calls.values())
    return tuple(calls._list)


def _output_matches(site: Site, value: Any) -> bool:
    """Return whether a forward site matches an output index or role.

    Parameters
    ----------
    site:
        Candidate forward site.
    value:
        Output index or semantic role.

    Returns
    -------
    bool
        Whether the site matches the requested output.
    """

    if isinstance(value, int):
        return getattr(site, "multi_output_index", None) == value
    return getattr(site, "multi_output_name", None) == str(value)


def _output_path_matches(saved_path: tuple[Any, ...], requested_path: tuple[Any, ...]) -> bool:
    """Return whether a saved typed path matches a user path.

    Parameters
    ----------
    saved_path:
        Captured typed output path.
    requested_path:
        User path using plain indices/keys/field names.

    Returns
    -------
    bool
        Whether the paths address the same output leaf.
    """

    if len(saved_path) != len(requested_path):
        return False
    return all(
        _output_path_component_matches(saved_component, requested_component)
        for saved_component, requested_component in zip(saved_path, requested_path)
    )


def _output_path_component_matches(saved_component: Any, requested_component: Any) -> bool:
    """Return whether one typed path component matches a user component.

    Parameters
    ----------
    saved_component:
        Captured typed path component.
    requested_component:
        User path component.

    Returns
    -------
    bool
        Whether the components are equivalent.
    """

    if isinstance(saved_component, TupleIndex):
        return saved_component.index == requested_component
    if isinstance(saved_component, (DictKey, HFKey)):
        return saved_component.key == requested_component
    if isinstance(saved_component, (NamedField, DataclassField)):
        return saved_component.name == requested_component
    return saved_component == requested_component


def _input_path_matches(site: Site, requested_path: tuple[Any, ...]) -> bool:
    """Return whether a forward site consumes an input at ``requested_path``.

    Parameters
    ----------
    site:
        Candidate forward site.
    requested_path:
        User path using plain indices, keys, or field names.

    Returns
    -------
    bool
        Whether any input-container leaf occurrence matches the path.
    """

    trace = getattr(site, "source_trace", None)
    site_labels = {
        label
        for label in (
            getattr(site, "layer_label", None),
            getattr(site, "layer_label_raw", None),
            getattr(site, "_layer_label_raw", None),
        )
        if label is not None
    }
    for record in getattr(trace, "_containers", {}).values():
        for snapshot in getattr(record, "snapshots", ()) or ():
            if getattr(snapshot, "role", None) != Role.MODEL_INPUT:
                continue
            for occurrence in getattr(snapshot, "leaf_occurrences", ()) or ():
                if occurrence.producer_op_label not in site_labels:
                    continue
                if _output_path_matches(tuple(occurrence.path), requested_path):
                    return True
    for container in getattr(site, "input_containers", ()) or ():
        for occurrence in getattr(container, "leaf_occurrences", ()) or ():
            if _output_path_matches(tuple(occurrence.path), requested_path):
                return True
    return False


def _grad_fn_type_matches(site: "GradFn", requested: str) -> bool:
    """Return whether a grad_fn_handle type matches user spelling flexibly.

    Parameters
    ----------
    site:
        Candidate grad_fn_handle log.
    requested:
        Requested type string.

    Returns
    -------
    bool
        Whether class name or normalized type matches.
    """

    lowered = requested.lower()
    normalized = lowered.removesuffix("backward0").removesuffix("backward")
    candidates = {
        site.class_name.lower(),
        site.type.lower(),
        site.label.lower(),
    }
    return lowered in candidates or normalized in candidates


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

    if isinstance(query, BaseSelector):
        return query
    if isinstance(query, str):
        from .selectors import contains

        return contains(query)
    if isinstance(query, TargetSpec):
        return _selector_from_spec(query.selector_kind, query.selector_value, query.metadata)
    if isinstance(query, FrozenTargetSpec):
        return _selector_from_spec(
            query.selector_kind,
            query.selector_value,
            dict(query.metadata),
        )
    raise SiteResolutionError(f"Unsupported site query {query!r}.")


def _selector_from_spec(kind: str, value: Any, metadata: dict[str, Any]) -> BaseSelector:
    """Build a selector from a target spec payload.

    Parameters
    ----------
    kind:
        Selector kind from a target spec.
    value:
        Selector payload.
    metadata:
        Selector metadata.

    Returns
    -------
    BaseSelector
        Selector matching the target spec.
    """

    from .selectors import (
        contains,
        facet,
        func,
        func_transform,
        grad_fn,
        grad_input,
        grad_output,
        head,
        in_backward_pass,
        input_at,
        label,
        module,
        output,
        output_at,
        regex,
        where,
        without_op,
    )

    if kind == "label":
        return label(str(value))
    if kind == "func":
        if isinstance(value, dict):
            return func(str(value.get("name")), output=value.get("output"))
        return func(str(value))
    if kind == "func_transform":
        return func_transform(None if value is None else str(value))
    if kind == "module":
        return module(str(value))
    if kind == "output":
        return output(value)
    if kind == "output_at":
        return output_at(value)
    if kind == "input_at":
        if isinstance(value, Sequence) and not isinstance(value, str):
            return input_at(*value)
        return input_at(value)
    if kind == "contains":
        return contains(str(value))
    if kind == "regex":
        return regex(str(value))
    if kind == "in_module":
        from .selectors import in_module as make_in_module

        selector = make_in_module(str(value))
        if isinstance(selector, BaseSelector):
            return selector
    if kind == "facet":
        if isinstance(value, dict):
            name = value.get("name")
            head_index = value.get("head_index")
            if head_index is not None:
                return head(int(head_index), None if name is None else str(name))
            if name is not None:
                return facet(str(name))
        raise SiteResolutionError(f"Unsupported facet selector payload {value!r}.")
    if kind == "predicate" and callable(value):
        return where(value, name_hint=metadata.get("name_hint"))
    if kind == "grad_fn":
        payload = dict(value)
        return grad_fn(
            payload.get("type"),
            label=payload.get("grad_fn_label_pattern"),
            is_custom=payload.get("is_custom"),
        )
    if kind in {"intervening", "without_op"}:
        return without_op()
    if kind == "label":
        return label(str(value))
    if kind == "not":
        nested = _normalize_query(value)
        return ~nested
    if kind in {"and", "or"}:
        if not isinstance(value, Sequence) or len(value) != 2:
            raise SiteResolutionError(f"{kind!r} target specs require two nested selectors.")
        left, right = value
        return CompositeSelector(
            cast("Literal['and', 'or']", kind),
            (_normalize_query(left), _normalize_query(right)),
        )
    if kind == "grad_kind":
        return grad_input() if value == "grad_input" else grad_output()
    if kind == "backward_pass":
        if not isinstance(value, int):
            raise SiteResolutionError("backward_pass target specs require an integer pass index.")
        return in_backward_pass(value)
    raise SiteResolutionError(f"Unsupported target spec selector kind {kind!r}.")


def _label_matches(site: Any, label: str) -> bool:
    """Return whether a layer-pass record has a requested label.

    Parameters
    ----------
    site:
        Layer-pass record.
    label:
        Label to match.

    Returns
    -------
    bool
        Whether the label matches any exact label field or lookup key.
    """

    candidate_labels = (
        getattr(site, "layer_label", None),
        getattr(site, "label", None),
        getattr(site, "layer_label", None),
        getattr(site, "layer_label_short", None),
        getattr(site, "label_short", None),
        getattr(site, "layer_label_short", None),
        getattr(site, "_layer_label_raw", None),
    )
    return label in candidate_labels or label in getattr(site, "lookup_keys", ())


def _module_output_matches(site: Any, address: str) -> bool:
    """Return whether a layer pass is the output boundary for a module.

    Parameters
    ----------
    site:
        Layer-pass record.
    address:
        Module address or pass-qualified module label.

    Returns
    -------
    bool
        Whether the site exits the requested module.
    """

    module_ops = getattr(site, "output_of_module_calls", ())
    return any(_module_label_matches(module_pass, address) for module_pass in module_ops)


def _module_label_matches(module_pass: str, address: str) -> bool:
    """Return whether a module pass label matches an address.

    Parameters
    ----------
    module_pass:
        Pass-qualified module label.
    address:
        Requested module address or pass label.

    Returns
    -------
    bool
        Whether the labels refer to the same module boundary.
    """

    module_address = module_pass.rsplit(":", 1)[0]
    return module_pass == address or module_address == address


def _predicate_payload(value: Any) -> tuple[Callable[[Any], bool], str | None]:
    """Validate and unpack a predicate selector payload.

    Parameters
    ----------
    value:
        Stored predicate payload.

    Returns
    -------
    tuple[Callable[[Op], bool], str | None]
        Predicate and optional name hint.

    Raises
    ------
    SiteResolutionError
        If the predicate payload is malformed.
    """

    if isinstance(value, tuple) and len(value) == 2 and callable(value[0]):
        predicate = value[0]
        name_hint = value[1] if value[1] is None or isinstance(value[1], str) else None
        return predicate, name_hint
    if callable(value):
        return value, None
    raise SiteResolutionError("tl.where(...) requires a callable predicate.")


__all__ = ["SiteTable", "_selector_resolution_direction", "find_sites", "resolve_sites"]
