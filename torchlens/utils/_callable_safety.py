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

GENERAL INVARIANT (the denylist-completeness class -- r5/r6 ``from_file`` /
``resize_``; r26 ``apply_`` / ``map_``; r39 ``map2_`` / ``register_hook``). A callable
reachable in the allowed namespace is NOT a pure forward op -- and must NOT resolve from
an untrusted bundle -- if it does ANY of the following, regardless of whether it was
individually enumerated:

* INVOKES an arbitrary Python callable (elementwise runners ``apply_`` / ``map_`` /
  ``map2_`` / any future ``map3_``; transforms like ``vmap``; the hook-registration
  family ``register_hook`` / ``register_post_accumulate_grad_hook`` / any future
  ``register_*``);
* REALLOCATES / rebinds tensor storage (``resize_`` / ``resize_as_`` /
  ``_resize_output_`` / ``_copy_from_and_resize`` -- uninitialized-memory disclosure or
  storage rebind), or
* MUTATES process-global torch / interpreter state that OUTLIVES ``Trace.run`` (the
  ``set_*`` seed/dtype/device/flag setters; the device-backend registration
  ``_register_device_module``).

Each class is closed by BOTH an audited exact-name set AND a STRUCTURAL pattern guard
(``(map|apply)\\d*_`` runners; a leading ``register`` after underscore-strip; a
``resize`` substring; the ``set_`` / ``_set_`` prefix) so a FUTURE sibling gadget is
denied by shape even when it is never added to a list. Every pattern was verified by
exhaustive enumeration of the fixed roots (torch / torch.Tensor / torch.nn.functional /
operator) to catch NO pure forward op; the sole benign structural false-deny is the
inert functionalization introspection query ``_functionalize_was_inductor_storage_resized``
(a boolean read, never a captured forward op).
"""

from __future__ import annotations

import sys
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
    }
)

# The ``operator`` / ``_operator`` root is handled by a dedicated POSITIVE
# allowlist (below), NOT the module gate: the module also exposes generic
# gadget / side-effecting primitives (``operator.call`` / ``attrgetter`` /
# ``methodcaller`` / ``itemgetter`` / ``setitem`` / ``delitem`` / the ``i*``
# in-place mutators) that are plainly not forward ops. It is therefore
# deliberately absent from ``_ALLOWED_FORWARD_OP_MODULES``.
_OPERATOR_MODULES: frozenset[str] = frozenset({"operator", "_operator"})

# Pure, side-effect-free ``operator`` callables that legitimately appear in a
# captured forward graph: arithmetic, comparison, bitwise, and index/sequence
# operators. This is a POSITIVE allowlist (default-DENY the rest of
# ``operator`` / ``_operator``). Unlike torch's dynamic ``_VariableFunctions``
# namespace -- thousands of drifting builtins that forced a denylist -- the pure
# operator set is small and stable, so an allowlist closes the generic-gadget
# class BY CONSTRUCTION: ``operator.call`` / ``attrgetter`` / ``methodcaller`` /
# ``itemgetter`` / ``setitem`` / ``delitem`` / ``setattr`` / ``delattr`` and the
# in-place ``iadd`` / ``imul`` / ... mutators can never be re-admitted by a
# future widening. The ``and_`` / ``or_`` names carry the module's trailing
# underscore; ``getitem`` (not ``setitem``) and ``concat`` (not ``iconcat``) are
# the read-only sequence ops.
_ALLOWED_OPERATOR_NAMES: frozenset[str] = frozenset(
    {
        "add",
        "sub",
        "mul",
        "truediv",
        "floordiv",
        "mod",
        "pow",
        "neg",
        "pos",
        "abs",
        "matmul",
        "and_",
        "or_",
        "xor",
        "invert",
        "lshift",
        "rshift",
        "lt",
        "le",
        "eq",
        "ne",
        "gt",
        "ge",
        "getitem",
        "index",
        "concat",
        "contains",
        "not_",
        "is_",
        "is_not",
        "length_hint",
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
        # The real import machinery behind ``importlib`` (r28, E-r28-1): under trust
        # these expose ``_frozen_importlib:__import__``, ``_call_with_frames_removed``
        # (a universal call gadget), and ``exec_module`` reachable via dotted walk.
        # Prefix matching then also covers their submodules.
        "_frozen_importlib",
        "_frozen_importlib_external",
        "runpy",
        "code",
        "codeop",
        "ctypes",
        # Exec / spawn / install (r26): stdlib entry points that EXECUTE
        # arbitrary code strings/callables (debuggers, tracers, profilers,
        # timeit), spawn processes or browsers (pydoc/webbrowser/antigravity/
        # platform/asyncio), or install packages (pip/setuptools/venv). Denied
        # EVEN under trust_custom_callables -- trust never authorizes these.
        "pdb",
        "bdb",
        "timeit",
        "trace",
        "cProfile",
        "profile",
        "pydoc",
        "webbrowser",
        "antigravity",
        "platform",
        "asyncio",
        "pip",
        "setuptools",
        "venv",
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


# STRUCTURAL close of the denylist-completeness class (r31). The explicit
# ``_DENIED_MODULES`` denylist above is inherently incomplete: successive audits kept
# finding *more* resolvable stdlib/builtin gadgets (r26 exec/spawn; r28
# ``_frozen_importlib``; r30 ``_imp`` / ``zipimport`` / ``io`` / ``linecache`` /
# ``fileinput`` / ``py_compile`` / ``compileall`` / ``gc`` / ``mmap`` / ``fcntl`` ...).
# Chasing modules one at a time never terminates. Instead we close the CLASS with a
# POSITIVE structural rule: on the TRUSTED foreign resolution path, DENY any resolved
# callable whose REAL top-level module is a Python STANDARD-LIBRARY or BUILTIN module.
# The detector is the authoritative ``sys.stdlib_module_names`` (frozenset, py3.10+)
# UNION ``sys.builtin_module_names``. Trust means "run this user recipe", NEVER "import
# a stdlib module": a stdlib/builtin owner is never a legitimate user recipe, so it is
# denied EVEN under ``trust_custom_callables`` or an explicit module allowlist. The
# explicit ``_DENIED_MODULES`` denylist above is KEPT as belt-and-suspenders.
#
# CARVE-OUTS (must STILL resolve): the pure-forward ``operator`` root -- ``operator``
# is stdlib and ``operator.neg`` et al. resolve to the C accelerator ``_operator``
# (builtin) -- is admitted here so the legitimate ``operator:neg`` custom ref survives.
# The torch namespaces (torch / torch.Tensor / torch.nn.functional) are NOT stdlib, so
# they are naturally allowed and need no carve-out. First-party ``torchlens.*`` (incl.
# appliance ``torchlens.neuro`` / ``torchlens.notebook`` under explicit trust) is not
# stdlib either, so it is naturally allowed. A user's ``allowed_custom_callable_modules``
# entry that names a NON-stdlib user package still resolves; but a user CANNOT re-allow
# a stdlib module (an allowlist entry naming e.g. ``io`` stays DENIED -- that is the
# whole point of "denied even under trust").
_ALLOWED_STDLIB_ROOTS: frozenset[str] = frozenset({"operator", "_operator"})


def _compute_stdlib_and_builtin_top_level() -> frozenset[str]:
    """Return the top-level Python stdlib + builtin module-name detector set.

    Uses the authoritative ``sys.stdlib_module_names`` (py3.10+) UNION
    ``sys.builtin_module_names``. On pre-3.10 interpreters (no
    ``stdlib_module_names``) it falls back to ``sys.builtin_module_names`` UNION the
    top-level names enumerated from the standard-library directory; the explicit
    ``_DENIED_MODULES`` denylist remains the belt-and-suspenders floor on that path.
    """

    names: set[str] = set(sys.builtin_module_names)
    stdlib = getattr(sys, "stdlib_module_names", None)
    if stdlib is not None:
        names |= set(stdlib)
        return frozenset(names)
    # pre-3.10 fallback: enumerate top-level module/package names from the stdlib dir
    # (``os.__file__``'s directory is the pure stdlib root, NOT site-packages).
    try:
        import os

        std_dir = os.path.dirname(os.__file__ or "")
        if std_dir:
            for entry in os.listdir(std_dir):
                if entry.endswith(".py"):
                    names.add(entry[:-3])
                elif "." not in entry and not entry.startswith("_"):
                    names.add(entry)
    except OSError:  # pragma: no cover - defensive; stdlib dir is always readable here.
        pass
    return frozenset(names)


_STDLIB_AND_BUILTIN_TOP_LEVEL: frozenset[str] = _compute_stdlib_and_builtin_top_level()


def is_denied_stdlib_or_builtin_module(module: str) -> bool:
    """Return whether a REAL module identity is a denied stdlib / builtin module.

    Keyed on the TOP-LEVEL package of ``module`` (so submodules such as
    ``importlib.util`` or ``os.path`` are covered). The pure-forward ``operator`` /
    ``_operator`` carve-out is admitted (returns ``False``); everything else whose
    top-level name is in the stdlib/builtin detector set is denied. First-party
    (``torchlens.*``) and third-party (``torch`` / ``numpy`` / user packages) modules
    are not in the detector set and are therefore allowed by this gate.
    """

    if not module:
        return False
    top_level = module.split(".", 1)[0]
    if top_level in _ALLOWED_STDLIB_ROOTS:
        return False
    return top_level in _STDLIB_AND_BUILTIN_TOP_LEVEL


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
        # Arbitrary-code-execution vectors reachable in the allowlisted torch
        # surface (r26): ``torch.compile`` wraps/compiles an arbitrary callable
        # (and mutates process-global dynamo state), and the elementwise
        # Python-callable runners ``Tensor.apply_`` / ``Tensor.map_`` each
        # INVOKE an attacker-supplied callable per element. Terminal names, per
        # this set's convention; none collides with a pure forward-op name.
        "compile",
        "apply_",
        "map_",
        # r39 (secE-r38-1): the SAME arbitrary-callable-INVOKE class that r26
        # denied ``apply_`` / ``map_`` for, but MISSED. ``Tensor.map2_`` is the
        # 3-tensor elementwise Python-callable runner (identical primitive to
        # ``map_``); ``torch.vmap`` (real module ``torch.func``, reachable as
        # ``torch.vmap``) is a transform that INVOKES an attacker fn; and
        # ``Tensor.register_hook`` / ``register_post_accumulate_grad_hook``
        # register an arbitrary callback fired on backward. Also caught by the
        # structural ``(map|apply)\d*_`` / leading-``register`` guards below, but
        # pinned here as the confirmed named misses.
        "map2_",
        "vmap",
        "register_hook",
        "register_post_accumulate_grad_hook",
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

# Process-global-state MUTATORS reachable in the allowlisted torch namespace
# (round-6). These are NOT forward ops: they persistently flip interpreter /
# torch global state (RNG seed + state, default dtype/device/tensor-type, thread
# counts, autograd/anomaly/determinism/autocast/matmul-precision flags) that
# OUTLIVES ``Trace.run()``. Their real ``__module__`` is the allowlisted torch
# namespace (``torch``, ``torch.random``, ``torch.autograd.grad_mode``,
# ``torch._tensor_str``), so the module gate admitted them and a crafted bundle
# op keyed e.g. ``("torch", "set_default_dtype")`` resolved pure=True and was
# CALLED at run() with attacker literals (persistent host-state corruption).
# Enumerated by auditing every ``set_*`` / seed / rng callable reachable through
# the allowlisted roots; the structural prefix guard below covers the whole
# ``set_*`` class, so this set mainly documents the confirmed family and pins the
# two non-``set_`` names (``manual_seed`` / ``seed``). GETTERS
# (``get_rng_state`` / ``get_default_dtype`` / ``initial_seed`` /
# ``get_num_threads``) are pure reads and STAY resolvable.
_DENIED_STATE_MUTATOR_NAMES: frozenset[str] = frozenset(
    {
        "manual_seed",
        "seed",
        "set_rng_state",
        "set_default_dtype",
        "set_default_device",
        "set_default_tensor_type",
        "set_num_threads",
        "set_num_interop_threads",
        "set_grad_enabled",
        "set_deterministic_debug_mode",
        "use_deterministic_algorithms",
        "set_flush_denormal",
        "set_anomaly_enabled",
        "set_printoptions",
        "set_float32_matmul_precision",
        "set_warn_always",
        "set_vital",
        # r39 (secE-r38-1): registers an arbitrary object as a device-backend
        # module -- a persistent process-global registry mutation. Its terminal
        # name does not lead with ``set_`` (it is ``_register_device_module``), so
        # it is pinned here AND caught by the leading-``register`` structural guard.
        "_register_device_module",
    }
)

# Storage-unsafe in-place ops (round-6): they REBIND or REALLOCATE a tensor's
# underlying storage rather than compute an elementwise result. ``resize_`` in
# particular exposes UNINITIALIZED heap memory as a trace-output tensor
# (info-leak / memory disclosure); ``set_`` / ``set_source_*`` repoint a tensor
# at attacker-chosen storage. Denied by EXACT terminal name -- these are NOT
# ordinary in-place ELEMENTWISE ops. C-level tensor methods report
# ``__module__ is None`` and would otherwise be admitted as tensor descriptors,
# so the name guard (which runs first) is what closes them.
_STORAGE_UNSAFE_NAMES: frozenset[str] = frozenset(
    {
        "set_",
        "set_source_Tensor",
        "set_source_Storage",
        "resize_",
        "resize_as_",
        "resize_as_sparse_",
        "sparse_resize_",
        "sparse_resize_and_clear_",
        # r39 (secE-r38-1): private reallocators and the non-underscore deprecated
        # ``resize`` / ``resize_as`` that escaped the exact set above. All are
        # covered by the ``resize`` substring guard too; pinned here as the
        # confirmed named misses. ``_resize_output_`` reallocates an output tensor's
        # storage; ``_copy_from_and_resize`` reallocates then copies.
        "_resize_output_",
        "_copy_from_and_resize",
        "resize",
        "resize_as",
    }
)

# STRUCTURAL close of the arbitrary-callable-INVOKE class (r39, secE-r38-1). The
# ``_DENIED_CALLABLE_NAMES`` enumeration kept missing siblings of the same
# elementwise Python-callable RUNNER / callback-REGISTRATION class (r26 denied
# ``apply_`` / ``map_``; ``map2_`` and the ``register_*hook`` family escaped). These
# POSITIVE structural rules deny a FUTURE sibling by SHAPE even if never enumerated.
# Verified by exhaustive enumeration of the fixed roots (2234 callables): NO pure
# forward op matches either pattern.
#   * ``(map|apply)\d*_`` -- the elementwise Python-callable runners ``map_`` /
#     ``map2_`` / ``apply_`` (+ any future ``map3_`` / ``apply2_``). Anchored to the
#     START (no leading underscore) so the aten sparse op
#     ``_sparse_semi_structured_apply`` -- NOT a Python-callable runner -- is NOT hit.
#   * a leading ``register`` (after stripping leading underscores) -- every
#     ``register_*`` / ``_register_*`` callable reachable in the allowed surface
#     REGISTERS a callback (``register_hook`` / ``register_post_accumulate_grad_hook``)
#     or an arbitrary object as global device-backend state
#     (``_register_device_module``); NONE is a pure forward op.
_CALLABLE_INVOKER_NAME_STEMS: tuple[str, ...] = ("map", "apply")


def _is_callable_invoker_name(name: str) -> bool:
    """Return whether ``name`` is an arbitrary-Python-callable INVOKER by SHAPE.

    Closes the callable-runner / callback-registration class structurally (r39):
    the ``(map|apply)\\d*_`` elementwise runners and the ``register_*`` /
    ``_register_*`` registration family. Verified against the fixed roots to match NO
    pure forward op (the aten ``_sparse_semi_structured_apply`` is excluded by the
    START anchor).
    """

    if name.lstrip("_").startswith("register"):
        return True
    for stem in _CALLABLE_INVOKER_NAME_STEMS:
        if name.startswith(stem):
            middle = name[len(stem) :]
            if middle.endswith("_"):
                digits = middle[:-1]
                if digits == "" or digits.isdigit():
                    return True
    return False


# Structural guard for the GLOBAL-STATE-SETTER prefix class (round-6). Verified
# by EXHAUSTIVE enumeration across every allowlisted root + tensor-op namespace:
# every ``set_*`` / ``_set_*`` callable reachable in the allowed surface is a
# process-global-state setter (or the storage-unsafe ``Tensor.set_``) -- NONE is
# a pure forward / elementwise op. THAT is what lets this be a leading-``set_``
# prefix guard while ordinary in-place ELEMENTWISE ops -- ``add_`` / ``mul_`` /
# ``sub_`` / ``div_`` / ``clamp_`` / ``relu_`` / ``sigmoid_`` / ``copy_`` /
# ``zero_`` / ``fill_`` / ``normal_`` / ``uniform_`` and friends, which TRAIL an
# underscore but never LEAD with ``set_`` -- stay resolvable. It is deliberately
# NOT a blanket trailing-underscore deny (that would wrongly kill those forward
# ops). ``_set_`` catches the ``torch._C._set_*`` private setter family;
# ``use_deterministic`` catches the sole non-``set_`` global-determinism setter.
# A future sibling setter (autocast / precision / backend flag) is thus denied by
# pattern even if not enumerated above.
_DENIED_STATE_SETTER_PREFIXES: tuple[str, ...] = (
    "set_",
    "_set_",
    "use_deterministic",
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
    callables that live inside an allowlisted module yet are not pure forward ops
    -- which a module-only policy cannot distinguish from the pure tensor ops
    sharing that module. It covers three classes:

    * file-I/O / serialization / import gadgets (round-5): ``torch.from_file`` and
      its audited siblings, plus the structural file/serial/import name guard;
    * process-global-state MUTATORS (round-6): ``set_default_dtype`` /
      ``manual_seed`` / ``set_num_threads`` and the whole ``set_*`` / ``_set_*``
      setter class, caught by exact name AND leading-``set_`` prefix;
    * storage-unsafe / REALLOCATING ops (round-6, r39): ``set_`` / ``resize_`` /
      ``set_source_*`` / ``_resize_output_`` / ``_copy_from_and_resize`` (info-leak /
      storage rebind), caught by exact name AND a ``resize`` substring guard;
    * arbitrary-callable INVOKERS (r26, r39): the elementwise runners ``apply_`` /
      ``map_`` / ``map2_``, the transform ``vmap``, and the callback / device-module
      registration family ``register_*``, caught by exact name AND the
      ``(map|apply)\\d*_`` / leading-``register`` structural guard.

    The ``set_``-prefix guard is safe against legitimate in-place ELEMENTWISE
    ops (``add_`` / ``mul_`` / ``clamp_`` ...): those TRAIL an underscore but
    never LEAD with ``set_`` (verified by exhaustive enumeration). Likewise the
    ``resize`` / ``(map|apply)\\d*_`` / ``register`` guards were verified to hit no
    pure forward op (the pure ``Tensor.is_set_to`` read, e.g., contains ``set_`` only
    mid-name and is preserved by the leading-only ``set_`` prefix).
    """

    name = _terminal_callable_name(func)
    if name in _DENIED_CALLABLE_NAMES or name in _DENIED_EXACT_NAMES:
        return True
    if name in _DENIED_STATE_MUTATOR_NAMES or name in _STORAGE_UNSAFE_NAMES:
        return True
    if any(name.startswith(prefix) for prefix in _DENIED_STATE_SETTER_PREFIXES):
        return True
    # r39: structural close of the arbitrary-callable-INVOKE and storage-REALLOC
    # classes so a future sibling (``map3_`` / a new ``register_*hook`` / a new
    # ``*_resize_*`` reallocator) is denied by shape even if never enumerated.
    if _is_callable_invoker_name(name):
        return True
    lowered = name.lower()
    if "resize" in lowered:
        return True
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
    terminal NAME is not a side-effecting callable (file-I/O / serialization /
    import gadget, process-global-state mutator, or storage-unsafe in-place op)
    and (b) its module is on the positive allowlist and off the side-effecting
    denylist. The ``operator`` / ``_operator`` root is gated separately by a
    POSITIVE NAME allowlist (``_ALLOWED_OPERATOR_NAMES``), so generic gadget /
    mutation primitives (``operator.call`` / ``attrgetter`` / ``methodcaller`` /
    ``itemgetter`` / ``setitem`` / ``delitem`` / ``iadd`` / ...) are default-denied
    while the pure arithmetic / comparison / bitwise / index operators still
    resolve. Module-less C tensor method descriptors are admitted when bound to a
    Tensor class. Anything else -- notably ``torch.load`` / ``torch.save`` /
    ``torch.from_file``, the state mutators ``set_default_dtype`` / ``manual_seed``
    / ``set_num_threads``, the storage-unsafe ``resize_`` / ``set_``, and any
    ``os`` / ``pickle`` / ``subprocess`` callable -- is refused. The name guard
    runs FIRST so a side-effecting builtin or method whose real module is the
    allowlisted ``torch`` namespace (or ``None`` for a C tensor method) cannot slip
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
    if module in _OPERATOR_MODULES:
        # POSITIVE allowlist for the operator root: only the pure arithmetic /
        # comparison / bitwise / index operators are admitted; every generic
        # gadget or mutation primitive (``call`` / ``attrgetter`` / ``setitem``
        # / ``iadd`` / ...) is default-denied.
        return _terminal_callable_name(real) in _ALLOWED_OPERATOR_NAMES
    return _matches(module, _ALLOWED_FORWARD_OP_MODULES)


def real_callable_module(func: Callable[..., Any]) -> str:
    """Return the REAL (capture-unwrapped) ``__module__`` of ``func`` (``""`` if none).

    The intervention resolver / metadata unpickler FOREIGN branches decide whether a
    resolved callable is a torch-namespace or module-less C callable (which must pass
    ``is_pure_forward_callable``) versus a genuinely-foreign TRUSTED user recipe (which
    resolves as-is -- trust means "run this user recipe"). That decision MUST key on the
    callable's REAL identity, never the surface ``__module__``:

    * a C tensor-method descriptor (``resize_`` / ``set_`` / ``apply_`` / ``map_``)
      reports ``__module__ is None``, and the foreign branches historically fell back to
      the (benign) trusted module name -- so a string check treated a side-effecting
      method as a benign foreign recipe (secE-r36-1);
    * WORSE, a torch op TorchLens has capture-wrapped reports
      ``__module__ == 'torchlens.backends.torch.wrappers'`` -- a truthy, non-torch
      string -- so BOTH a ``__module__``-missing check AND a raw torch-prefix check are
      spoofed once torch is wrapped (the near-universal live state), letting a wrapped
      ``resize_`` / ``torch.load`` slip.

    Unwrapping first restores the true owner: ``""`` for a module-less C tensor method,
    ``"torch"`` / ``"torch.serialization"`` for a torch builtin, and the genuine user
    module for a real foreign recipe.
    """

    real = _unwrap_capture_wrapper(func)
    return str(getattr(real, "__module__", "") or "")


def is_denied_operator_gadget(func: Callable[..., Any]) -> bool:
    """Return whether ``func`` is an ``operator`` / ``_operator`` gadget to DENY.

    The ``operator`` / ``_operator`` root is carved OUT of the stdlib/builtin denial
    (``_ALLOWED_STDLIB_ROOTS``) so the legitimate pure operators (``operator:neg`` et
    al.) still resolve on the FOREIGN resolution paths (the intervention resolver's
    custom/foreign tail and the trusted-unpickler foreign tail). Without an additional
    POSITIVE name filter that carve-out would re-admit the generic operator GADGETS --
    ``attrgetter`` / ``methodcaller`` / ``call`` / ``getitem`` / ``setitem`` /
    ``delitem`` / the in-place ``iadd`` / ``imul`` / ... mutators -- which enable an
    RCE chain (``attrgetter('__globals__')(identity)`` -> ``__builtins__`` ->
    ``__import__`` -> ``os.system``).

    This REUSES ``is_pure_forward_callable``'s operator branch: it returns ``True``
    (DENY) only when the resolved callable's REAL (capture-unwrapped) module is
    ``operator`` / ``_operator`` AND its terminal name is NOT in the pure-forward
    ``_ALLOWED_OPERATOR_NAMES`` allowlist. Any non-operator callable returns ``False``
    (not this gate's concern -- the caller applies the torch / stdlib / denylist gates
    separately). ``operator:neg`` / ``_operator:neg`` return ``False`` (ALLOWED);
    ``operator:attrgetter`` / ``_operator:setitem`` return ``True`` (DENIED even under
    operator trust).
    """

    real = _unwrap_capture_wrapper(func)
    module = str(getattr(real, "__module__", "") or "")
    if module not in _OPERATOR_MODULES:
        return False
    return _terminal_callable_name(real) not in _ALLOWED_OPERATOR_NAMES


# Extras-gated "appliance" subpackages whose ``__init__`` imports FOREIGN
# third-party dependencies; a callable resolved from one of these ran foreign
# top-level code on import and must never be treated as inert first-party code.
# Mirrors ``torchlens._io._safe_unpickle._TORCHLENS_APPLIANCE_MODULES``.
_APPLIANCE_MODULES: frozenset[str] = frozenset({"torchlens.neuro", "torchlens.notebook"})

# Attribute stamped on a callable at ``@torchlens.facets.register`` time (on the
# recipe function AND its optional predicate). It is the DETERMINISTIC, process-
# state-INDEPENDENT marker that a callable is a genuine, vetted facet-registration
# entry point -- travels with the function object itself, so it is set whenever the
# defining module is imported, independent of the LIVE facet registry (keying off
# that mutable registry was a prior ordering-dependent load-failure bug). It lets a
# genuine but PRIVATE-named registration callable (e.g. the built-in residual
# predicate ``_is_transformer_block``) be admitted while a private import gadget
# (``torchlens.utils._module_is_installed``) -- which is never a registration entry
# point -- stays denied. Set via ``torchlens.semantic.facets.register``.
FACET_RECIPE_MARKER_ATTR = "_torchlens_facet_recipe"


# POSITIVE allowlist of the genuinely-INERT public first-party callables that
# legitimately appear as pickle-REDUCE / resolver targets in real bundles, keyed by
# the callable's REAL ``(module, qualname)`` (decided AFTER capture-unwrap, never by
# the attacker-controlled pickled path). This REPLACES the pre-r22 "any PUBLIC
# torchlens.* callable is inert" fallthrough, which was unsound: that policy admitted
# side-effecting PUBLIC callables (``fastlog.cleanup.cleanup_partial`` ->
# ``shutil.rmtree`` of an attacker-named directory on plain ``tl.load``;
# ``mark_torch_capability_missing`` / ``register_payload_codec`` / ``save_intervention``
# / ``export.*`` / ``wrap_torch`` -> global/process-state poison) because the verb
# denylist (``_DENIED_*`` name guards) never covered the filesystem / registry /
# capability-mutation verb class and a public name alone proved nothing. A denylist
# over a version-drifting first-party surface is unsound; default-deny is the fix
# (exactly as ``_ALLOWED_OPERATOR_NAMES`` inverted the operator surface).
#
# Membership derivation (empirical + vetted): the ONLY first-party callables that
# legit bundles / the intervention resolver reference BY IDENTITY are (a) facet
# recipes + their predicates -- admitted deterministically by the marker branch
# below, NOT enumerated here -- and (b) the pure built-in intervention helper
# factories plus the ``identity`` display transform, enumerated here. Each entry was
# read and vetted INERT to INVOKE with attacker args: every helper factory merely
# constructs and returns a ``HelperSpec`` dataclass (its tensor-op side effects live
# only inside an unexecuted ``factory``/``_hook`` closure) or raises a validation
# error; ``identity`` returns its argument. NONE performs filesystem / import / exec /
# spawn / network / global-state mutation. Anything torchlens-owned but not here (and
# not marker-stamped) fails closed. Erring to DENY: only these vetted members admit.
_VETTED_INERT_FIRST_PARTY: frozenset[tuple[str, str]] = frozenset(
    {
        # Pure display transform (raw-input/output transform saved by reference).
        ("torchlens.utils.display", "identity"),
        # Built-in intervention helper factories (import-ref-portable helper saves +
        # the intervention resolver's ``torchlens.intervention.helpers:<name>`` refs).
        ("torchlens.intervention.helpers", "zero_ablate"),
        ("torchlens.intervention.helpers", "mean_ablate"),
        ("torchlens.intervention.helpers", "resample_ablate"),
        ("torchlens.intervention.helpers", "steer"),
        ("torchlens.intervention.helpers", "scale"),
        ("torchlens.intervention.helpers", "clamp"),
        ("torchlens.intervention.helpers", "noise"),
        ("torchlens.intervention.helpers", "project_onto"),
        ("torchlens.intervention.helpers", "project_off"),
        ("torchlens.intervention.helpers", "swap_with"),
        ("torchlens.intervention.helpers", "splice_module"),
        ("torchlens.intervention.helpers", "bwd_hook"),
        ("torchlens.intervention.helpers", "grad_zero"),
        ("torchlens.intervention.helpers", "grad_scale"),
        ("torchlens.intervention.helpers", "grad_clip"),
        ("torchlens.intervention.helpers", "grad_noise"),
        ("torchlens.intervention.helpers", "grad_clamp"),
    }
)


def is_inert_first_party_callable(func: Callable[..., Any]) -> bool:
    """Return whether ``func`` is a first-party TorchLens callable that is INERT to
    INVOKE with attacker-controlled arguments at unpickle / resolve time.

    A pickle ``REDUCE`` (and an intervention ``custom`` import ref) INVOKES the
    admitted callable with attacker-supplied args. Trusting *every* torchlens-owned
    callable (the pre-r21 policy) was a confirmed load-time RCE: the private helper
    ``torchlens.utils._module_is_installed`` performs ``importlib.import_module`` on
    its argument. The pre-r22 narrowing kept a "PUBLIC name proves inertness"
    fallthrough, which was STILL unsound: the verb denylist omitted the filesystem /
    registry / capability-mutation verb class, so PUBLIC side-effecting callables such
    as ``torchlens.fastlog.cleanup.cleanup_partial`` (``shutil.rmtree`` of an
    attacker-named directory) and ``mark_torch_capability_missing`` (global HAS_*
    flag poison) were admitted and INVOKED on a plain ``tl.load(path)``.

    This is now a POSITIVE ALLOWLIST (default-DENY), keyed on the callable's REAL,
    capture-unwrapped identity -- never the attacker-controlled pickled path:

    * FIRST-PARTY only -- the real ``__module__`` is genuinely ``torchlens.*`` and is
      NOT an extras-gated appliance package.
    * SIDE-EFFECT-FREE by name -- the same structural file-I/O / serialization /
      import / spawn / global-state-mutation guard applied to the torch surface
      (belt-and-suspenders; the allowlist below is the load-bearing gate).
    * Then admitted ONLY if EITHER it carries the deterministic facet-registration
      marker (``FACET_RECIPE_MARKER_ATTR``, stamped on every recipe function AND
      predicate at ``@torchlens.facets.register`` time -- this covers a genuine but
      PRIVATE-named registration callable such as the built-in residual predicate
      ``_is_transformer_block``), OR its real ``(module, qualname)`` is in the frozen,
      empirically-vetted ``_VETTED_INERT_FIRST_PARTY`` set (``identity`` + the pure
      built-in intervention helper factories).

    Everything else torchlens-owned-but-not-vetted fails closed (the caller raises).
    The public-name fallthrough is GONE: ``cleanup_partial`` / ``cleanup_tmp`` /
    ``mark_torch_capability_missing`` / ``register_payload_codec`` /
    ``save_intervention`` / ``torchlens.export.*`` / ``wrap_torch`` / ``unwrap_torch`` /
    ``_module_is_installed`` are all denied -- neither marker-stamped nor allowlisted.

    The marker cannot be forged through the pickle stream: it is an attribute set on
    the resolved function object at import/registration time, NOT carried in the
    pickle. A REDUCE naming an arbitrary function resolves to THAT object, which bears
    the marker only if it was genuinely stamped by ``@facets.register``.
    """

    real = _unwrap_capture_wrapper(func)
    module = str(getattr(real, "__module__", "") or "")
    if not (module == "torchlens" or module.startswith("torchlens.")):
        return False
    if _matches(module, _APPLIANCE_MODULES):
        return False
    # Deny I/O / import / exec / spawn / state-mutation by NAME regardless of marker
    # (belt-and-suspenders; the positive allowlist below is the load-bearing gate).
    if _is_side_effecting_callable_name(real):
        return False
    # Genuine facet-registration entry point (recipe func or its predicate), even if
    # its name is private -- the marker is set by our own trusted registration code.
    if getattr(real, FACET_RECIPE_MARKER_ATTR, False) is True:
        return True
    # Otherwise admit ONLY the frozen, vetted-inert public callables by their REAL
    # (module, qualname). No public-name fallthrough: default-deny.
    qualname = str(getattr(real, "__qualname__", "") or "")
    if not qualname:
        return False
    return (module, qualname) in _VETTED_INERT_FIRST_PARTY


def unsafe_callable_reason(func: Callable[..., Any]) -> str:
    """Return a short, stable description of a refused callable's real module.

    Used only to populate a typed diagnostic; never returned to untrusted code.
    """

    real = _unwrap_capture_wrapper(func)
    module = str(getattr(real, "__module__", "") or "<none>")
    qualname = str(getattr(real, "__qualname__", getattr(real, "__name__", "?")))
    return f"{module}:{qualname}"
