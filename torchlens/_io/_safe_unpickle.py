"""Restricted unpickler for TorchLens portable-bundle metadata.

A loaded ``.tlspec`` bundle is UNTRUSTED input. Its ``metadata.pkl`` payload was
historically read with an *unrestricted* :class:`pickle.Unpickler`, so a crafted
bundle whose ``metadata.pkl`` carried a ``__reduce__`` / ``os.system`` gadget
executed arbitrary code at ``pickle.load`` time -- BEFORE the resolver / callable
safety defenses (:mod:`torchlens.utils._callable_safety` and the runnable
resolver) could ever fire. That is a load-time RCE sitting in front of the entire
prior hardening effort.

This module closes that front door with the standard safe-unpickle defense: a
class allowlist (default-deny), mirroring the intent of ``torch.load(...,
weights_only=True)``. Only the exact set of classes / values that legitimately
appear in TorchLens ``Trace`` metadata is admitted; every other global -- and in
particular any ``os`` / ``sys`` / ``subprocess`` / ``posix`` / ``nt`` /
``builtins`` code-exec callable, ``importlib``, or ``torch.serialization``
loader -- raises :class:`pickle.UnpicklingError` and executes nothing.

Allowlist policy (fail-closed):

* Objects resolved from the ``torchlens`` package are admitted when they are a
  ``type`` (the metadata dataclasses / enums / accessors -- reconstructed by normal
  unpickling, never INVOKED with attacker args), or a VETTED-INERT first-party
  callable: a PUBLIC, side-effect-free ``torchlens.*`` facet recipe / transform /
  helper (e.g. ``torchlens.utils.display.identity``), decided by
  ``torchlens.utils._callable_safety.is_inert_first_party_callable``. Trust is
  decided by the resolved object's real module, never the pickled path, so a key
  that walks attributes off a torchlens module to reach ``os.system`` (real module
  ``posix``) is denied. Crucially, a pickle ``REDUCE`` INVOKES the admitted callable
  with attacker args, so the pre-r21 blanket "any torchlens-owned callable" policy
  was a load-time RCE: it trusted the PRIVATE ``torchlens.utils._module_is_installed``
  (which ``importlib.import_module``s its argument). The narrowed gate is
  deterministic (public-name + side-effect-name guard, no live-registry lookup) and
  admits our vetted first-party code without admitting that gadget.
* A FOREIGN facet-recipe callable (a user/test ``@tl.facets.register`` recipe
  embedded by identity in a ``FacetRegistrySnapshot``) is handled DETERMINISTICALLY
  by the same trust model the resolver uses. Known-dangerous modules (``os`` /
  ``sys`` / ``subprocess`` / ``pickle`` / ``builtins`` / ``importlib`` / ...) are
  hard-denied unconditionally. Otherwise, under an explicit trust opt-in
  (``trust_custom_callables`` / ``allowed_custom_callable_modules``) the module is
  imported and the real callable resolved; by default it is LOAD-TOLERATED as an
  inert ``_DeferredForeignCallable`` placeholder -- the foreign module is NEVER
  imported at unpickle, the trace still loads structurally, and any attempt to CALL
  the recipe (facet computation, or a ``__reduce__`` gadget) fails closed. Admission
  depends only on ``(module, name)`` + trust flags, never on process state (an
  import-time / live-registry snapshot was the ordering-dependent load-failure bug).
* A curated set of safe stdlib data/value types (``collections.OrderedDict``, the
  pure-data builtin container types plus ``object``, and the NumPy numeric-array
  reconstruction helpers).
* Any object resolved from the ``torch`` package that is a DATA ``type`` (tensor /
  Module / storage classes) or a ``torch`` dtype / layout / memory_format
  *instance*, plus the pure ``torch._utils._rebuild_*`` reconstructors. Resolving
  a global only references it; constructing a torch data type executes no attacker
  code. Torch's I/O-capable types are the exception -- constructing
  ``torch._C.PyTorchFileReader`` / ``PyTorchFileWriter`` / ``FileCheck`` /
  ``torch.package.PackageImporter`` / ``torch.serialization.*`` / ``torch.jit.*``
  opens/reads/writes an attacker-named filesystem path (arbitrary file access +
  zip-bomb DoS) at unpickle time, so these TYPES are denied by a module-prefix +
  structural-name guard aligned with torch's own ``weights_only`` exclusions.
  Non-type torch callables (e.g. ``torch.serialization.load``) are denied.
* Preview-backend (MLX / Paddle / JAX) numeric array/dtype ``type`` objects and
  framework dtype-like instances.
* ``builtins.getattr`` is legitimately used to reconstruct references to torch C
  tensor methods (``getattr(TensorBase, "sum")``). It is admitted only through a
  *safe* wrapper that restricts the target object to the enumerated torch C
  callable-holder classes, which structurally blocks the classic
  ``getattr(obj, "__globals__")`` pivot to ``eval`` / ``exec``.
* ``torch.storage._load_from_bytes`` legitimately reconstructs small embedded
  tensors (e.g. a control-flow predicate value) but internally calls
  ``torch.load`` with the *unsafe* default, i.e. a nested unrestricted unpickler.
  It is admitted only through a wrapper that forces ``weights_only=True`` so the
  nested load routes through torch's own restricted unpickler.

If a future legitimate class is not yet covered, loading it fails closed with a
clear error; we NEVER admit a code-exec gadget to make a load succeed.
"""

from __future__ import annotations

import io
import pickle
from dataclasses import dataclass
from typing import Any, BinaryIO, Collection, Mapping

import torch

__all__ = ["SafeBundleUnpickler"]


# Known side-effecting / code-exec module identities that must NEVER resolve from
# an untrusted bundle, regardless of trust flags. This mirrors
# ``torchlens.utils._callable_safety._DENIED_MODULES`` (the resolver's denylist)
# and is duplicated here -- rather than imported -- to keep this security
# front-door free of any import-order coupling (it is imported very early during
# ``torchlens`` startup). Serialization, code execution, imports, and process /
# OS / filesystem I/O are all hard-denied here BEFORE the deferred-reference
# fallback below can ever apply.
_DENIED_FOREIGN_MODULES: frozenset[str] = frozenset(
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


def _module_denied(module: str) -> bool:
    """Return whether ``module`` is (nested under) a hard-denied dangerous module."""

    return any(
        module == denied or module.startswith(denied + ".") for denied in _DENIED_FOREIGN_MODULES
    )


# torch I/O / serialization / packaging / jit TYPE denylist.
#
# The torch-``type`` branch below admits any resolved ``torch.*`` ``type`` on the
# premise that "resolving/constructing a torch data type executes no attacker
# code". That premise HOLDS for the tensor / storage / Size / Parameter / dtype /
# nn-module classes legit metadata references, but it is FALSE for torch's
# I/O-capable types: constructing one opens/reads/writes a filesystem PATH taken
# from the pickle stream at ``tl.load`` time -- e.g.
# ``torch._C.PyTorchFileReader('/etc/passwd')`` (arbitrary-path read + zip-bomb
# DoS) or ``torch.package.PackageImporter(path)`` (directory traversal read).
# torch's own ``weights_only`` allowlist deliberately EXCLUDES these types; we
# align with that intent and deny them here so they are never constructed.
#
# Denial uses two independent signals, because ``__module__`` alone is
# insufficient: ``torch._C.PyTorchFileReader.__module__`` is ``"torch"`` -- the
# SAME module string as ``torch.Size`` / ``torch.Tensor`` -- so a module test
# cannot separate the file reader from a benign shape type. Hence:
#   (a) a MODULE-PREFIX denylist over the pickled path AND the resolved real
#       ``__module__`` (catches ``torch.serialization`` / ``torch.jit`` /
#       ``torch.package`` / ``torch.hub`` families), plus
#   (b) a STRUCTURAL TYPE-NAME guard (the load-bearing one) that denies any type
#       whose pickled name or real ``__qualname__`` contains a file / reader /
#       writer / package / directory / zip / (file)check / importer / pickler
#       token -- this is what stops ``PyTorchFileReader`` / ``PyTorchFileWriter``
#       / ``FileCheck`` / ``PackageImporter`` / ``DirectoryReader``.
#
# Residual (stated honestly): a positive allowlist of the torch DATA types is
# impractical because legit metadata may reference ANY ``nn.Module`` subclass
# (an open-ended set) as a ``module_type``. So this is a denylist: a
# hypothetical future torch type whose construction does I/O yet whose name
# contains none of the tokens and whose module is outside the denied families
# would still be admitted. The token + module families cover the entire known
# torch file-I/O / serialization / packaging / jit surface (audited against
# torch's ``weights_only`` exclusions), so the residual is bounded and small.
_DENIED_TORCH_TYPE_MODULE_PREFIXES: tuple[str, ...] = (
    "torch.serialization",
    "torch.jit",
    "torch.package",
    "torch.hub",
    "torch.utils.tensorboard",
    "torch.utils.cpp_extension",
    "torch.utils.bottleneck",
    "torch.distributed",
    "torch.multiprocessing",
)

_DENIED_TORCH_TYPE_NAME_TOKENS: tuple[str, ...] = (
    "file",  # PyTorchFileReader / PyTorchFileWriter / *FileStore
    "reader",  # DirectoryReader / *Reader
    "writer",  # *Writer
    "package",  # PackageImporter / PackageExporter
    "importer",
    "exporter",
    "directory",
    "zip",  # *ZipFile / zip archive helpers
    "check",  # FileCheck
    "pickler",  # Pickler / Unpickler
    "unpickler",
)


def _torch_type_denied(pickled_module: str, pickled_name: str, obj: type) -> bool:
    """Return whether a resolved torch ``type`` is an I/O / serialization gadget.

    Denies torch's file-reader / file-writer / packaging / serialization / jit
    types whose construction is NOT inert (opens/reads/writes a filesystem path
    from the pickle stream). See ``_DENIED_TORCH_TYPE_MODULE_PREFIXES`` /
    ``_DENIED_TORCH_TYPE_NAME_TOKENS`` for the policy rationale.

    Parameters
    ----------
    pickled_module:
        Attacker-controlled pickled module path (e.g. ``"torch._C"``).
    pickled_name:
        Attacker-controlled pickled global name (e.g. ``"PyTorchFileReader"``).
    obj:
        The resolved torch ``type`` object.

    Returns
    -------
    bool
        ``True`` if the type must be denied (never constructed).
    """

    real_module = str(getattr(obj, "__module__", "") or "")
    for candidate in (pickled_module, real_module):
        if any(
            candidate == prefix or candidate.startswith(prefix + ".")
            for prefix in _DENIED_TORCH_TYPE_MODULE_PREFIXES
        ):
            return True
    real_name = str(getattr(obj, "__qualname__", "") or getattr(obj, "__name__", "") or "")
    haystack = f"{pickled_name} {real_name}".lower()
    return any(token in haystack for token in _DENIED_TORCH_TYPE_NAME_TOKENS)


@dataclass(frozen=True)
class _DeferredForeignCallable:
    """Inert placeholder for a foreign recipe callable retained on an untrusted load.

    A saved ``FacetRegistrySnapshot`` may reference a user/test-registered facet
    recipe by identity (``@tl.facets.register``). Importing that foreign module at
    unpickle time would execute its top-level code -- exactly the load-time RCE
    surface this module closes. So under a default (untrusted) load the reference
    is LOAD-TOLERATED as this placeholder instead of imported: the trace loads
    structurally (the recipe reference is preserved for inspection / re-save), but
    the foreign module is NEVER imported and any attempt to CALL the recipe (facet
    computation, or a ``__reduce__`` construction gadget) fails closed.

    This mirrors the resolver's load-tolerate / execute-deny least-privilege split
    (``resolve_function_registry_key``): a foreign custom callable is retained as an
    unresolved reference on load and only imported+resolved under an explicit trust
    opt-in (``trust_custom_callables`` / ``allowed_custom_callable_modules``).

    Admission here is DETERMINISTIC: it depends only on the pickled ``(module,
    name)`` and the caller's trust flags -- never on process state (whether the
    module happens to be imported, or whether the recipe is currently in the live
    registry), which was the ordering-dependent load-failure bug.
    """

    module: str
    qualname: str

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Fail closed: an untrusted foreign recipe is never executed.

        Raises
        ------
        pickle.UnpicklingError
            Always. If invoked during unpickle (a ``__reduce__`` gadget) the load
            fails closed; if invoked later (facet computation) it signals that a
            trust opt-in is required to run the foreign recipe.
        """

        raise pickle.UnpicklingError(
            "Refusing to execute foreign custom callable "
            f"{self.module}:{self.qualname} resolved from an untrusted bundle. "
            "Pass trust_custom_callables=True or allowed_custom_callable_modules "
            "to import and run a trusted foreign recipe."
        )


def _is_torchlens_owned(obj: Any) -> bool:
    """Return whether a resolved object genuinely lives under the ``torchlens`` package.

    Trust for a callable is decided by the RESOLVED object's real ``__module__``,
    never by the pickled import path. This closes the "walk ``os.system`` off a
    torchlens module" pivot (``getattr(torchlens._io.tlspec, 'os').system`` has
    ``__module__ == 'posix'`` and is therefore denied) -- mirroring the resolver's
    identically-named check.
    """

    owner = str(getattr(obj, "__module__", "") or "")
    return owner == "torchlens" or owner.startswith("torchlens.")


# Extras-gated "appliance" subpackages whose ``__init__`` imports heavy FOREIGN
# third-party dependencies (rsatoolbox / brainscore_core for ``torchlens.neuro``,
# IPython / jupyter_client for ``torchlens.notebook``). Unlike the rest of the
# first-party ``torchlens.*`` namespace -- which is safe to import + inspect
# because it runs only our own already-installed code (see
# ``_is_torchlens_owned`` above) -- importing one of these two runs foreign
# top-level code with NO trust opt-in. A bundle-supplied global naming one of
# them (or a submodule) must therefore be EXCLUDED from the "torchlens is our
# own code" fast path below and instead fall through to receive the exact same
# deny-by-default / trust-opt-in / load-tolerate treatment already applied to a
# genuinely foreign module at the tail of ``find_class`` -- never passed to
# ``super().find_class`` (which imports the module) to make that determination.
_TORCHLENS_APPLIANCE_MODULES: frozenset[str] = frozenset({"torchlens.neuro", "torchlens.notebook"})


def _is_torchlens_appliance_module(module: str) -> bool:
    """Return whether ``module`` is (nested under) an extras-gated appliance package."""

    return any(
        module == appliance or module.startswith(appliance + ".")
        for appliance in _TORCHLENS_APPLIANCE_MODULES
    )


# Exact (module, name) globals resolved directly. Every entry is either a pure
# data container/value type or a pure tensor-reconstruction helper -- none can
# execute attacker code merely by being resolved or called on pickle-provided
# arguments.
_SAFE_EXPLICIT_GLOBALS: frozenset[tuple[str, str]] = frozenset(
    {
        ("collections", "OrderedDict"),
        ("collections", "defaultdict"),
        ("collections", "Counter"),
        # Pure-data builtin constructors / base type. Explicitly EXCLUDES getattr
        # / eval / exec / __import__ / compile / open and every other code-exec
        # builtin. ``object`` is the base type used by default ``__reduce_ex__``.
        ("builtins", "object"),
        ("builtins", "list"),
        ("builtins", "set"),
        ("builtins", "frozenset"),
        ("builtins", "dict"),
        ("builtins", "tuple"),
        ("builtins", "bytearray"),
        ("builtins", "bytes"),
        ("builtins", "complex"),
        ("builtins", "int"),
        ("builtins", "float"),
        ("builtins", "str"),
        ("builtins", "bool"),
        ("builtins", "slice"),
        # The ``Ellipsis`` singleton (``...``). Pickle resolves it as a bare
        # ``builtins.Ellipsis`` global (no dedicated opcode, unlike ``None``);
        # resolving the name just returns the existing singleton object, so this
        # is strictly inert -- no different in kind from admitting the ``slice``
        # constructor above.
        ("builtins", "Ellipsis"),
        # torch C callable-holder classes (used as getattr targets and, rarely,
        # as bare globals). They are types; returning them is inert -- the only
        # exploit path (getattr pivot) is closed by ``_safe_getattr``.
        ("torch._C", "TensorBase"),
        ("torch._C", "_TensorBase"),
        ("torch._C", "_VariableFunctionsClass"),
        # NumPy numeric-array reconstruction helpers (raw-input ndarrays and
        # preview-backend numeric payloads). These rebuild arrays from a raw
        # numeric buffer / dtype and execute no attacker code for numeric dtypes.
        # Both the legacy (<2.0) and new (>=2.0) private module paths are covered.
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
        ("numpy.core.numeric", "_frombuffer"),
        ("numpy._core.numeric", "_frombuffer"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
    }
)

# Preview-backend array module roots whose resolved *types* / dtype instances are
# admitted (MLX / Paddle / JAX numeric array + dtype reconstruction). Only ``type``
# objects and framework dtype-like instances are admitted; callables are not.
_PREVIEW_ARRAY_MODULE_ROOTS: tuple[str, ...] = ("mlx.core",)


def _build_allowed_getattr_objects() -> frozenset[Any]:
    """Resolve the torch C callable-holder classes admitted as getattr targets.

    Returns
    -------
    frozenset[Any]
        The torch C classes off which ``getattr`` may legitimately fetch a
        tensor-method reference. Missing names (torch version drift) are skipped.
    """

    objects: list[Any] = []
    for name in ("TensorBase", "_TensorBase", "_VariableFunctionsClass"):
        holder = getattr(torch._C, name, None)
        if isinstance(holder, type):
            objects.append(holder)
    return frozenset(objects)


_ALLOWED_GETATTR_OBJECTS: frozenset[Any] = _build_allowed_getattr_objects()


def _safe_getattr(obj: Any, name: Any) -> Any:
    """Restricted ``getattr`` admitted in place of ``builtins.getattr``.

    Legitimate bundle metadata reconstructs torch C tensor-method references via
    ``getattr(torch._C.TensorBase, "sum")`` and similar. Admitting raw
    ``builtins.getattr`` would enable the classic pickle pivot
    ``getattr(<any function>, "__globals__")`` -> ``eval`` / ``exec``. This
    wrapper structurally blocks that pivot by restricting the target object to
    the enumerated torch C callable-holder classes; any other target -- including
    an already-resolved torchlens recipe function -- fails closed.

    Restricting the TARGET is not enough (round-5): the admitted
    ``_VariableFunctionsClass`` holder also exposes side-effecting file-I/O
    builtins -- notably ``torch.from_file`` -- so ``getattr(_VariableFunctionsClass,
    "from_file")`` would return a callable that CREATES/TRUNCATES or READS an
    arbitrary file when used as the callable of a following pickle ``REDUCE``. The
    resolved callable is therefore additionally gated through the shared
    callable-safety policy (``is_pure_forward_callable``): a pure tensor-method /
    factory reference (``sum`` / ``from_numpy`` / ``frombuffer``) is returned, while
    ``from_file`` and any other side-effecting builtin fail closed. The policy is
    imported lazily to keep this early-imported security front door free of
    import-order coupling.

    Parameters
    ----------
    obj:
        Attribute-access target supplied by the pickle stream.
    name:
        Attribute name supplied by the pickle stream.

    Returns
    -------
    Any
        ``getattr(obj, name)`` for an admitted torch C target and a non-side-effecting
        result.

    Raises
    ------
    pickle.UnpicklingError
        If ``obj`` is not an admitted torch C callable-holder class, ``name`` is not
        a string, or the resolved attribute is a side-effecting callable.
    """

    if not isinstance(name, str) or obj not in _ALLOWED_GETATTR_OBJECTS:
        raise pickle.UnpicklingError(
            f"Blocked getattr during bundle metadata unpickle: getattr on {obj!r} is not permitted."
        )
    resolved = getattr(obj, name)
    if callable(resolved):
        from ..utils._callable_safety import is_pure_forward_callable

        if not is_pure_forward_callable(resolved):
            raise pickle.UnpicklingError(
                "Blocked side-effecting torch callable during bundle metadata "
                f"unpickle: getattr(..., {name!r}) is not a pure forward/tensor op."
            )
    return resolved


def _safe_load_from_bytes(data: bytes) -> Any:
    """Restricted replacement for ``torch.storage._load_from_bytes``.

    The stock helper reconstructs an embedded tensor storage by calling
    ``torch.load`` with the *unsafe* legacy default (an unrestricted nested
    unpickler == RCE). This wrapper forces ``weights_only=True`` so the nested
    load routes through torch's own restricted unpickler; a malicious embedded
    payload therefore fails closed instead of executing.

    Parameters
    ----------
    data:
        Serialized tensor-storage bytes from the pickle stream.

    Returns
    -------
    Any
        The reconstructed storage/tensor from a weights-only nested load.

    Raises
    ------
    pickle.UnpicklingError
        If the nested weights-only load fails (including because it refused a
        disallowed global).
    """

    try:
        return torch.load(io.BytesIO(data), weights_only=True)
    except Exception as exc:  # noqa: BLE001 -- fail closed, never re-exec unsafely
        raise pickle.UnpicklingError(
            "Blocked unsafe embedded-tensor load during bundle metadata unpickle."
        ) from exc


class SafeBundleUnpickler(pickle.Unpickler):
    """Default-deny unpickler for untrusted TorchLens bundle ``metadata.pkl``.

    Applies an optional rename map (for classes moved by locked renames) and then
    gates every resolved global through the module allowlist. Anything outside the
    allowlist raises :class:`pickle.UnpicklingError` without importing or calling
    attacker-chosen code.
    """

    def __init__(
        self,
        file: BinaryIO,
        *,
        rename_map: Mapping[tuple[str, str], tuple[str, str]] | None = None,
        trust_custom_callables: bool = False,
        allowed_custom_callable_modules: Collection[str] | None = None,
    ) -> None:
        """Initialize the restricted unpickler.

        Parameters
        ----------
        file:
            Binary file object positioned at the start of a pickle stream.
        rename_map:
            Optional ``(module, name) -> (module, name)`` remapping applied before
            allowlist resolution, preserving load of legitimately-renamed classes.
        trust_custom_callables:
            Whether to admit a foreign (non-torch/torchlens/numpy) callable by
            importing and resolving it. Mirrors the intervention-resolver contract:
            ``False`` (default) denies unless the callable is already registered in
            the live facet registry (a positive allowlist against trusted live
            state that never triggers a fresh import).
        allowed_custom_callable_modules:
            Optional narrow allowlist of foreign module names whose callables may be
            imported and resolved even when ``trust_custom_callables`` is ``False``.
        """

        super().__init__(file)
        self._rename_map: Mapping[tuple[str, str], tuple[str, str]] = rename_map or {}
        self._trust_custom_callables = trust_custom_callables
        self._allowed_custom_callable_modules = (
            frozenset(allowed_custom_callable_modules)
            if allowed_custom_callable_modules is not None
            else None
        )

    def _custom_module_trusted(self, module: str) -> bool:
        """Return whether a foreign module may be imported+resolved via trust flags.

        Parameters
        ----------
        module:
            Foreign pickled module path.

        Returns
        -------
        bool
            ``True`` if the module is explicitly allowlisted, or broad trust is on.
        """

        if self._allowed_custom_callable_modules is not None:
            return module in self._allowed_custom_callable_modules
        return self._trust_custom_callables

    def find_class(self, module: str, name: str) -> Any:  # noqa: C901 -- explicit policy
        """Resolve a pickled global only if it is on the metadata allowlist.

        Parameters
        ----------
        module:
            Pickled module path.
        name:
            Pickled global name.

        Returns
        -------
        Any
            The resolved class / value / safe wrapper.

        Raises
        ------
        pickle.UnpicklingError
            If the (possibly renamed) global is not admitted by the allowlist.
        """

        module, name = self._rename_map.get((module, name), (module, name))

        # Safe wrappers for the two legitimately-needed but individually-dangerous
        # callables.
        if (module, name) == ("builtins", "getattr"):
            return _safe_getattr
        if (module, name) == ("torch.storage", "_load_from_bytes"):
            return _safe_load_from_bytes

        # Exact-pair allowlist of inert data/value types and tensor reconstructors.
        if (module, name) in _SAFE_EXPLICIT_GLOBALS:
            return super().find_class(module, name)

        # Pure torch tensor reconstruction helpers. ``_rebuild_*`` functions only
        # reassemble tensors from already-resolved (allowlisted) components; any
        # class they receive as an argument was itself gated through find_class.
        if module == "torch._utils" and name.startswith("_rebuild"):
            return super().find_class(module, name)

        # torch package: admit value instances (dtype / layout / memory_format /
        # ``torch.Size`` etc.) and a resolved DATA ``type`` (tensor / Module /
        # storage classes, e.g. ``torch.nn.modules.linear.Identity``). Resolving a
        # global merely references it, and constructing a torch DATA type executes
        # no attacker code -- but constructing an I/O-capable torch type
        # (``PyTorchFileReader`` / ``PyTorchFileWriter`` / ``FileCheck`` /
        # ``PackageImporter`` / ``torch.serialization.*`` / ``torch.jit.*``) opens/
        # reads/writes an attacker-named filesystem path at unpickle time. Those
        # I/O / serialization / packaging / jit types are DENIED via
        # ``_torch_type_denied`` (module-prefix + structural name guard), aligning
        # with torch's own ``weights_only`` allowlist which excludes them. Non-type
        # callables (e.g. ``torch.serialization.load``) are denied; the two
        # individually-dangerous ones are wrapped above.
        if module == "torch" or module.startswith("torch."):
            obj = super().find_class(module, name)
            if isinstance(obj, (torch.dtype, torch.layout, torch.memory_format)):
                return obj
            if isinstance(obj, type):
                if _torch_type_denied(module, name, obj):
                    raise pickle.UnpicklingError(
                        "Blocked torch I/O / serialization type during bundle "
                        f"metadata unpickle: {module}.{name} (construction performs "
                        "filesystem access; not an inert data type)."
                    )
                return obj
            raise pickle.UnpicklingError(
                f"Blocked disallowed torch global during bundle metadata unpickle: {module}.{name}."
            )

        # Preview-backend numeric array/dtype reconstruction (MLX / Paddle / JAX):
        # admit resolved ``type`` objects and framework dtype-like instances only.
        if any(
            module == root or module.startswith(root + ".") for root in _PREVIEW_ARRAY_MODULE_ROOTS
        ):
            obj = super().find_class(module, name)
            if isinstance(obj, type):
                return obj
            resolved_module = getattr(type(obj), "__module__", "") or ""
            if any(
                resolved_module == root or resolved_module.startswith(root + ".")
                for root in _PREVIEW_ARRAY_MODULE_ROOTS
            ):
                return obj
            raise pickle.UnpicklingError(
                "Blocked disallowed preview-backend global during bundle metadata "
                f"unpickle: {module}.{name}."
            )

        # TorchLens-owned metadata: admit classes always, and a NARROW, vetted set
        # of first-party CALLABLES. Importing a torchlens submodule runs only
        # trusted first-party code, so it is safe to import + inspect the resolved
        # object's true owner. Trust is decided by the resolved ``__module__``,
        # NEVER by the pickled path: a key like ``torchlens._io.tlspec`` /
        # ``os.system`` walks attributes off a torchlens module to reach
        # ``os.system`` (real module ``posix``), which is NOT torchlens-owned and is
        # therefore denied.
        #
        # A resolved TYPE is admitted -- it is reconstructed via normal unpickling,
        # never INVOKED with attacker args. A resolved CALLABLE is admitted only if
        # it is a vetted-INERT first-party callable (public facet recipe / transform
        # / helper that cannot import/exec/open/spawn): a pickle ``REDUCE`` INVOKES
        # the admitted callable with attacker args, and the pre-r21 blanket
        # ``_is_torchlens_owned`` admission trusted EVERY torchlens-owned callable --
        # including the private import gadget ``torchlens.utils._module_is_installed``
        # (``importlib.import_module`` of attacker input == confirmed load-time RCE).
        # The gate is DETERMINISTIC (public-name + side-effect-name guard), never
        # keyed off the live facet registry (that ordering-dependent snapshot was a
        # prior load-failure bug).
        #
        # EXCLUDES the extras-gated appliance packages (``torchlens.neuro`` /
        # ``torchlens.notebook``): importing THOSE runs foreign third-party code,
        # so a global naming one of them must never reach ``super().find_class``
        # here -- it falls through to the foreign-global handling below instead.
        if (
            module == "torchlens" or module.startswith("torchlens.")
        ) and not _is_torchlens_appliance_module(module):
            obj = super().find_class(module, name)
            if isinstance(obj, type):
                return obj
            # Imported lazily to keep this early security front door free of
            # import-order coupling (mirrors ``_safe_getattr`` above).
            from ..utils._callable_safety import is_inert_first_party_callable

            if callable(obj) and _is_torchlens_owned(obj) and is_inert_first_party_callable(obj):
                return obj
            raise pickle.UnpicklingError(
                "Blocked disallowed torchlens global during bundle metadata "
                f"unpickle: {module}.{name}."
            )

        # Foreign global -- e.g. a user/test-registered ``@tl.facets.register``
        # recipe embedded by identity in a FacetRegistrySnapshot. Admission here is
        # DETERMINISTIC: it depends ONLY on the pickled ``(module, name)`` and the
        # caller's trust flags -- never on process state (whether the module is
        # already imported, or whether the recipe is in the live registry). Keying
        # off live state was the ordering-dependent load-failure bug.
        #
        # 1. Hard-deny known dangerous modules (os / sys / subprocess / pickle /
        #    builtins / importlib / ...) unconditionally -- they are never a
        #    legitimate metadata global and importing / resolving them is the RCE.
        if _module_denied(module):
            raise pickle.UnpicklingError(
                f"Blocked dangerous global during bundle metadata unpickle: {module}.{name}."
            )
        # 2. Explicit trust opt-in: import + resolve the real foreign callable,
        #    mirroring ``resolve_function_registry_key`` under trust. Importing the
        #    module executes its top-level code, so this is gated behind the flags.
        if self._custom_module_trusted(module):
            return super().find_class(module, name)
        # 3. Default (untrusted): LOAD-TOLERATE as an inert deferred reference. The
        #    foreign module is NEVER imported here (import executes code); the recipe
        #    reference is preserved so the trace loads structurally, and any attempt
        #    to CALL it (facet computation, or a ``__reduce__`` gadget) fails closed.
        return _DeferredForeignCallable(module, name)
