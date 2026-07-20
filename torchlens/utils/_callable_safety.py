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
``resize_``; r26 ``apply_`` / ``map_``; r39 ``map2_`` / ``register_hook``; r41
``cond`` / ``while_loop`` / ``autocast_increment_nesting``). A callable reachable in the
allowed namespace is NOT a pure forward op -- and must NOT resolve from an untrusted
bundle -- if it does ANY of the following, regardless of whether it was individually
enumerated:

* INVOKES an arbitrary Python callable -- whether the callable arrives by NAME shape
  (elementwise runners ``apply_`` / ``map_`` / ``map2_`` / any future ``map3_``;
  transforms like ``vmap``; the hook-registration family ``register_hook`` /
  ``register_post_accumulate_grad_hook`` / any future ``register_*``) OR by SIGNATURE
  shape (a higher-order / callback-taking op whose ``inspect.signature`` exposes a
  ``Callable``-annotated or callable-named parameter -- ``torch.cond`` / ``while_loop``
  branch fns, ``handle_torch_function``'s ``public_api``,
  ``triplet_margin_with_distance_loss``'s ``distance_function``, ``_check_with``'s
  ``message``, ``_disable_dynamo``'s ``fn``);
* REALLOCATES / rebinds tensor storage (``resize_`` / ``resize_as_`` /
  ``_resize_output_`` / ``_copy_from_and_resize`` -- uninitialized-memory disclosure or
  storage rebind), or
* MUTATES process-global torch / interpreter state that OUTLIVES ``Trace.run`` (the
  ``set_*`` seed/dtype/device/flag setters; the device-backend registration
  ``_register_device_module``; and the NON-``set_`` global mutators the ``set_`` prefix
  missed -- the autocast nesting counters ``autocast_increment_nesting`` /
  ``autocast_decrement_nesting``, the cache flushers ``clear_autocast_cache`` /
  ``_cufft_clear_plan_cache``, and the cuFFT plan-cache setter
  ``_cufft_set_plan_cache_max_size``).

Each class is closed by BOTH an audited exact-name set AND a STRUCTURAL guard so a FUTURE
sibling gadget is denied by shape even when it is never added to a list:

* callable-INVOKE -- the ``(map|apply)\\d*_`` runner / leading-``register`` name guard
  AND a SIGNATURE guard (``_signature_invokes_callable``: any ``Callable``-annotated or
  callable-named parameter). The signature guard is what makes r39's "denied by shape
  even when never enumerated" claim TRUE for the higher-order ops (``cond`` /
  ``while_loop`` / ...) that carry no ``map`` / ``register`` name marker;
* storage-REALLOC -- the ``resize`` substring guard;
* global-MUTATE -- the ``set_`` / ``_set_`` prefix guard PLUS the non-``set_`` verb close
  (``nesting`` counter; ``clear`` + ``cache`` flush; ``set_plan_cache`` sizer).

Every pattern was verified by exhaustive enumeration of the fixed roots (torch /
torch.Tensor / torch.nn.functional / operator) to catch NO pure forward op: the
``nesting`` / ``clear``+``cache`` / ``set_plan_cache`` verbs leave the pure ``nuclear_norm``
(``clear`` without ``cache``), the pure quantization op
``_fake_quantize_per_tensor_affine_cachemask_tensor_qparams`` (``cache`` without
``clear``), the casting ops ``_autocast_to_full_precision`` /
``_autocast_to_reduced_precision``, and every ``is_*``/``get_*`` autocast + cuFFT getter
resolvable; the SIGNATURE guard leaves every pure forward op (no pure tensor op takes a
callable parameter) resolvable. The sole benign structural false-deny remains the inert
functionalization introspection query ``_functionalize_was_inductor_storage_resized``
(a boolean read, never a captured forward op).

R43 STRUCTURAL INVERSION (the denylist-of-verbs approach reached its limit). The
denylist above closes the CLASS "callable is a stdlib/builtin module" structurally, but
for the internal-builtin torch roots ``torch`` / ``torch._C`` / ``torch._tensor`` it
still relied on a growing verb denylist to subtract the non-forward builtins those roots
host alongside the pure op catalog. Successive audits kept defeating that with a sibling
the verb list never enumerated (r41 ``autocast_increment_nesting``; r42
``_enable_functionalization`` / ``_functionalize_enable_reapply_views`` /
``share_memory_`` / ``_sobol_engine_initialize_state_``). r43 inverts the decision on
those EXACT roots to DEFAULT-DENY with a positive STRUCTURAL recognized-operator
predicate (``_is_recognized_operator`` -- torch-overridable identity / aten schema /
pure factory / audited ``to_sparse_coo`` wrapper), decided against torch's OWN operator
authority (``get_overridable_functions`` / ``torch.ops.aten``) which is independent of
this gate and self-updates across torch versions. This closes the whole functionalization
family, ``share_memory_``, JIT / IR type constructors, Storage / legacy ``*Tensor``
ctors, state getters, and deprecated methods as a CLASS -- not instance by instance. The
verb / name / signature belts above are KEPT and run FIRST (as diagnostic belts and to
catch overridable-but-unsafe ops such as ``share_memory_``). DEEPER allowlisted operator
submodules keep their module-prefix admission. ``torch._sobol_engine_initialize_state_``
is a genuine aten operator and is admitted as a documented residual (its native
crash-on-malformed-args is a torch operator-robustness boundary outside the
side-effect-free admission contract).
"""

from __future__ import annotations

import inspect
import sys
import warnings
from contextlib import ExitStack, contextmanager
from functools import lru_cache
from typing import Any, Callable, Iterator

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
        # r49 hon2_1: ENUMERATION-COMPLETENESS ONLY (gate-NEUTRAL). ``torch._C._sparse`` is
        # ALREADY admitted by the ``torch._C`` prefix entry above (``_matches`` prefix-covers
        # it), so adding it changes NO admission outcome; it is listed so the private-C
        # forward-op MODULE enumeration (the single source of truth consumed by the
        # cross-thread witness belt via ``private_c_forward_op_module_names``) is complete.
        "torch._C._sparse",
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


def private_c_forward_op_module_names() -> tuple[str, ...]:
    """Return the canonical ``torch._C._*`` private-C forward-op MODULE names (r49 hon2_1).

    Single source of truth for the cross-thread witness's private-C free-function belt
    (:func:`torchlens.backends.torch.completeness_witness._private_c_forward_op_modules`): the
    ``torch._C._*`` entries of :data:`_ALLOWED_FORWARD_OP_MODULES`, so a future private-C op
    module added to that curated set is AUTO-covered by the witness. Names only -- the witness
    resolves each on the RUNNING torch and filters to ``types.ModuleType`` objects, which drops
    the class-typed, read-only / non-Python-patchable ``_VariableFunctions`` / ``_TensorBase``
    holders (an accepted residual) and any name absent on the running torch (graceful degrade).
    """

    return tuple(
        sorted(name for name in _ALLOWED_FORWARD_OP_MODULES if name.startswith("torch._C._"))
    )


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
        # r41 (secE-r40-1): NON-``set_`` process-global mutators the r6 ``set_*``
        # audit MISSED because none leads with ``set_``. All MUTATE process-global
        # torch state that OUTLIVES ``Trace.run`` and are directly reachable via the
        # run-path (zero/low-arg, take no callable). Pinned here as the confirmed
        # named misses AND closed structurally by ``_is_global_state_mutator_name``
        # below (``nesting`` / ``clear``+``cache`` / ``set_plan_cache`` verbs).
        # ``autocast_increment_nesting`` / ``autocast_decrement_nesting`` bump the
        # process-global autocast nesting counter (NOT restored by the run's
        # ambient-context snapshot); ``clear_autocast_cache`` / ``_cufft_clear_plan_cache``
        # flush process-global caches; ``_cufft_set_plan_cache_max_size`` resizes the
        # global cuFFT plan cache. The ``is_*``/``get_*`` autocast + cuFFT-plan-cache
        # getters are pure reads and STAY resolvable.
        "autocast_increment_nesting",
        "autocast_decrement_nesting",
        "clear_autocast_cache",
        "_cufft_clear_plan_cache",
        "_cufft_set_plan_cache_max_size",
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
        # r43 (secE-r42-2): ``Tensor.share_memory_`` REBINDS the tensor's storage to a
        # shared-memory-backed allocation (``data_ptr()`` changes, ``is_shared() -> True``)
        # -- an OS-level shared mapping / IPC data-exposure surface that outlives
        # ``Trace.run``, directly parallel to the denied ``set_`` (repoint) and ``resize_``
        # (reallocate). It is torch-OVERRIDABLE, so the r43 structural operator predicate
        # would ADMIT it on identity; this storage belt runs FIRST and closes it. Also
        # covered by the ``share_memory`` substring guard below; pinned here as the
        # confirmed named miss. The pure boolean reader ``is_shared`` leads ``is_`` and is
        # never a rebind -- it stays resolvable.
        "share_memory_",
        "_share_memory_",
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


def _is_global_state_mutator_name(name: str) -> bool:
    """Return whether ``name`` marks a NON-``set_`` process-global-state MUTATOR (r41).

    The r6 ``set_*`` prefix guard closes only the leading-``set_`` global-setter class.
    It MISSED a sibling family of process-global torch mutators whose names do NOT lead
    with ``set_`` (secE-r40-1): the autocast nesting counters, the autocast / cuFFT cache
    flushers, and the ``_cufft_``-prefixed plan-cache SIZER (whose ``set_`` is embedded
    after the ``_cufft_`` prefix, so ``startswith('set_')`` never fires). These VERB
    markers were verified by exhaustive enumeration of the fixed roots to have ZERO
    overlap with the pure forward surface, so a FUTURE sibling mutator is denied by shape:

    * ``nesting`` -- the ``autocast_increment_nesting`` / ``autocast_decrement_nesting``
      counter mutators (the ONLY two ``nesting`` callables reachable);
    * ``clear`` AND ``cache`` together -- the cache FLUSHERS ``clear_autocast_cache`` /
      ``_cufft_clear_plan_cache`` (and the per-tensor
      ``_clear_non_serializable_cached_data``). Requiring BOTH tokens preserves the pure
      ``nuclear_norm`` (has ``clear`` via "nu-CLEAR", no ``cache``) and the pure
      quantization op ``..._cachemask_...`` (has ``cache``, no ``clear``);
    * ``set_plan_cache`` -- the cuFFT plan-cache SIZER
      ``_cufft_set_plan_cache_max_size``. The read-only ``get_plan_cache`` GETTERS carry
      ``get_plan_cache``, so they are NOT hit and STAY resolvable.
    """

    low = name.lower()
    if "nesting" in low:
        return True
    if "clear" in low and "cache" in low:
        return True
    if "set_plan_cache" in low:
        return True
    return False


# STRUCTURAL close of the arbitrary-callable-INVOKE class by SIGNATURE (r41,
# secE-r40-1). ``_is_callable_invoker_name`` (below) closes the class by NAME shape
# (``(map|apply)\\d*_`` / leading-``register``) -- but the higher-order control-flow and
# callback-taking ops carry NO such name marker (``torch.cond`` / ``torch.while_loop`` /
# ``handle_torch_function`` / ``triplet_margin_with_distance_loss`` / ``_check_with`` /
# ``_disable_dynamo``), so r39's "denied by shape even when never enumerated" claim was
# FALSE for them. They are unified instead by SIGNATURE: each takes an arbitrary Python
# callable as a parameter. ``_signature_invokes_callable`` denies any op whose
# ``inspect.signature`` exposes a ``Callable``-annotated parameter OR a callable-NAMED
# parameter (``fn`` / ``func`` / ``callback`` / ``hook`` / ``if_true`` / ``if_false`` /
# any ``*_fn`` / ``*_func``). Verified by exhaustive enumeration of the fixed roots to
# hit NO pure forward op -- no pure tensor op takes a callable parameter. A no-signature
# C builtin (the vast pure-op surface) yields ``False`` here and falls through to the
# module gate: this detector can only ADD denials, never rescue.
_CALLABLE_PARAM_NAMES: frozenset[str] = frozenset(
    {"fn", "func", "callback", "hook", "closure", "body", "branch", "if_true", "if_false"}
)


def _annotation_takes_callable(annotation: Any) -> bool:
    """Return whether a parameter annotation denotes a ``Callable`` type."""

    if annotation is inspect.Parameter.empty:
        return False
    return "callable" in str(annotation).lower()


def _signature_invokes_callable(func: Callable[..., Any]) -> bool:
    """Return whether ``func``'s signature exposes an arbitrary-Python-callable parameter.

    Closes the higher-order / callback-taking INVOKE class structurally (r41): a torch
    op that accepts a Python callable is NOT a pure forward op and must not resolve from
    an untrusted bundle -- it is the same class as the r39-denied ``vmap`` (both invoke an
    attacker fn), just carrying no ``map`` / ``register`` name marker. Detected by BOTH a
    ``Callable``-typed annotation AND a callable-conventional parameter NAME, so an op
    with unannotated callable params (``torch.while_loop(cond_fn, body_fn, ...)``) is
    still caught. Fails SAFE: a callable with no inspectable signature (the pure C tensor
    ops) returns ``False`` and is left to the module gate; this guard only ever ADDS a
    denial.
    """

    try:
        signature = inspect.signature(func)
    except (ValueError, TypeError):
        return False
    for parameter in signature.parameters.values():
        if _annotation_takes_callable(parameter.annotation):
            return True
        low = parameter.name.lower()
        if low in _CALLABLE_PARAM_NAMES or low.endswith("_fn") or low.endswith("_func"):
            return True
    return False


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
    * process-global-state MUTATORS (round-6, r41): ``set_default_dtype`` /
      ``manual_seed`` / ``set_num_threads`` and the whole ``set_*`` / ``_set_*``
      setter class, caught by exact name AND leading-``set_`` prefix; PLUS the
      NON-``set_`` global mutators the ``set_`` prefix missed
      (``autocast_increment_nesting`` / ``autocast_decrement_nesting`` /
      ``clear_autocast_cache`` / ``_cufft_clear_plan_cache`` /
      ``_cufft_set_plan_cache_max_size``), caught by exact name AND the
      ``nesting`` / ``clear``+``cache`` / ``set_plan_cache`` verb close;
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
    # r41: structural close of the NON-``set_`` process-global mutator class
    # (``nesting`` counters / ``clear``+``cache`` flushers / ``set_plan_cache`` sizer)
    # so a future sibling is denied by shape even if never enumerated.
    if _is_global_state_mutator_name(name):
        return True
    # r39: structural close of the arbitrary-callable-INVOKE and storage-REALLOC
    # classes so a future sibling (``map3_`` / a new ``register_*hook`` / a new
    # ``*_resize_*`` reallocator) is denied by shape even if never enumerated.
    if _is_callable_invoker_name(name):
        return True
    lowered = name.lower()
    # r43 (secE-r42-2): the ``resize`` storage-realloc guard did not cover the
    # ``share_memory`` storage-REBIND primitive; broaden to both storage-rebind verbs so
    # a future ``*_share_memory*`` sibling is denied by shape even if never enumerated.
    # The only sibling carrying the token is the pure reader ``is_shared`` (leads ``is_``,
    # no ``share_memory`` token), so no pure forward op is denied.
    if "resize" in lowered or "share_memory" in lowered:
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


# --------------------------------------------------------------------------- #
# STRUCTURAL RECOGNIZED-OPERATOR predicate (r43). Closes the CLASS the r6/r39/r41
# name-and-verb denylists could only chase instance-by-instance (secE-r42-1/2/3).
# --------------------------------------------------------------------------- #
#
# The three internal-builtin roots ``torch`` / ``torch._C`` / ``torch._tensor`` are
# the whack-a-mole surface: they host the entire pure forward-op catalog AND thousands
# of non-forward internal builtins (functionalization dispatch-mode controls, JIT / IR
# type constructors, Storage + legacy ``*Tensor`` type ctors, accelerator / device
# hooks, process-global state getters, deprecated Tensor methods). The pre-r43 gate
# admitted any callable whose ``__module__`` was one of these roots by PREFIX, then
# tried to subtract the dangerous ones with an ever-growing verb denylist -- which
# successive audits kept defeating with a sibling the verb list never enumerated
# (r41 ``autocast_increment_nesting``; r42 ``_enable_functionalization`` /
# ``share_memory_`` / ``_sobol_engine_initialize_state_``). Chasing verbs never
# terminates.
#
# We invert it: on those EXACT roots, DEFAULT-DENY and admit ONLY when the callable is
# structurally RECOGNIZED as a genuine forward operator, decided against torch's own
# OPERATOR AUTHORITY (which is independent of this buggy gate and self-updates across
# torch versions):
#   * torch-OVERRIDABLE identity -- it is a member of
#     ``torch.overrides.get_overridable_functions()`` / ``get_testing_overrides()``
#     (the canonical "is a real torch operator" registry), OR
#   * it carries an ``aten`` OPERATOR SCHEMA (``torch.ops.aten.<name>.overloads()``
#     non-empty), OR
#   * it is one of a small, stable pure tensor FACTORY names (``from_numpy`` /
#     ``frombuffer`` / ``asarray`` / ``from_dlpack`` -- genuine forward constructors that
#     are neither overridable nor aten by that terminal name), OR
#   * it is a narrow, audited pure Tensor WRAPPER (``to_sparse_coo`` -- the ONE genuine
#     live forward method that is a pure Python wrapper delegating to
#     ``self.to_sparse()``, so it is neither overridable nor aten by name; every sibling
#     ``to_sparse_csr`` / ``csc`` / ``bsr`` / ``bsc`` / ``to_dense`` / ``to_mkldnn`` IS
#     overridable-or-aten and admits without a rescue).
#
# A C-level Tensor method descriptor reports ``__module__ is None`` and is admitted by
# the module-less ``_is_tensor_method_descriptor`` path BEFORE these roots are reached,
# so ``Tensor.to_sparse`` / ``Tensor.relu_`` / ``Tensor.scatter_add_`` and the whole C
# tensor-method surface keep resolving; only PYTHON methods/functions carrying a real
# ``torch`` / ``torch._C`` / ``torch._tensor`` ``__module__`` reach this predicate.
# DEEPER allowlisted torch operator submodules (``torch.nn.functional`` /
# ``torch.functional`` / ``torch._VF`` / ``torch._C._nn`` / ``torch.linalg`` / ...) are
# NOT in the exact-root set and keep their module-PREFIX admission, so no deep-surface
# forward op is over-denied.
#
# Reconciliation (torch 2.8.0, exhaustive surface enumeration): of the pre-r43
# gate-passers on these roots, the structural predicate DENIES 234 and admits ZERO new
# -- every one of the 234 is a non-forward internal builtin (functionalization family,
# JIT/IR type ctors, Storage/legacy ``*Tensor`` ctors, docstring/torch-function plumbing,
# state getters, deprecated ``eig`` / ``lstsq`` / ``solve`` / ``symeig`` / ``reinforce``),
# and the ONLY genuine forward method the raw predicate lost -- ``to_sparse_coo`` -- is
# rescued by the wrapper allowlist. The name/verb/signature belts above still run FIRST
# (as diagnostic belts): they catch the OVERRIDABLE-but-unsafe ``share_memory_`` (a
# storage rebind) before this predicate could admit it on identity.
#
# DOCUMENTED RESIDUAL: ``torch._sobol_engine_initialize_state_`` is a genuine ``aten``
# operator, so this predicate ADMITS it. Its native crash-on-malformed-args behavior is
# a torch-wide operator robustness boundary OUTSIDE the side-effect-free callable-
# admission contract -- not an in-scope side-effect finding. There is no structural
# signal that denies it while keeping ``add_`` / ``relu_`` (all aten, in-place,
# trailing-``_``); the only ways to deny it are the name-enumeration this predicate
# removes. It is admitted as an accepted, documented residual (see the immunizer, which
# pins it as ADMITTED so a future refactor cannot silently name-deny it, and
# ``docs/reference/runnable_tlspec_contract.md`` sec. 11).
_OPERATOR_GATED_ROOTS: frozenset[str] = frozenset({"torch", "torch._C", "torch._tensor"})

# Pure tensor FACTORY constructors that are genuine forward ops but are neither
# torch-overridable nor an aten schema by their terminal name. Small and stable.
_PURE_TENSOR_FACTORY_NAMES: frozenset[str] = frozenset(
    {"from_numpy", "frombuffer", "asarray", "from_dlpack"}
)

# Narrow, audited pure Tensor Python WRAPPERS that delegate to a recognized operator but
# are themselves neither overridable nor aten by terminal name. ``to_sparse_coo`` is the
# sole such live forward method (delegates to ``self.to_sparse()``); widening this set is
# how a future bare-root legit op is rescued -- NOT by loosening the operator gate.
_PURE_TENSOR_WRAPPER_NAMES: frozenset[str] = frozenset({"to_sparse_coo"})

# Safe, pure-READ tensor PROPERTY accessors (``x.T`` / ``x.mT`` / ``x.H`` / ``x.mH`` /
# ``x.real`` / ``x.imag``): getset descriptors on the C ``TensorBase`` whose access is a
# pure view / read with no side effect. The loader resolves a recorded
# ``("torch.Tensor", <name>, "method")`` property key to a SYNTHETIC Python getter
# (``torchlens._io.runnable_load._safe_tensor_property_getter``) whose ``__module__`` is
# ``"torch._tensor"``, so it lands on the operator-gated roots with a FRESH function id
# (never torch-overridable) and no aten schema by terminal name -- the r43 inversion
# denied the whole class (a regression; pre-r43 module-prefix admission allowed it).
# This is the CANONICAL copy of the safe-property allowlist; the capture-side keyer
# (``torchlens.backends.torch.ops``) and the load-side resolver import it, so the three
# surfaces cannot drift.
#
# STRUCTURAL, not hand-listed (r45): the set is COMPUTED by probing every live
# ``TensorBase`` getset descriptor against ``_pure_view`` -- a descriptor is admitted
# iff its getter returns a storage-sharing, autograd-PRESERVING, non-mutating tensor
# view. This admits ``{T, mT, H, mH, real, imag}`` and DENIES ``data`` by construction
# (``.data`` shares storage but DETACHES from autograd and is a live lvalue mutation
# channel -- the autograd-bypass alias, not a pure forward read). A FUTURE torch tensor
# property is auto-classified: admitted-if-pure-view / denied-otherwise, with no per-name
# edit. The r45 immunizer (``tests/test_r45_property_classification.py``) pins the full
# classification of every descriptor so drift goes RED. Recognizing ``.H``/``.mH`` closed
# the r44 corr1_1 / secF_1 over-deny (a real capture used them; the frozen 4-name set
# refused them).


def _iter_tensor_getset_descriptor_names() -> tuple[str, ...]:
    """Return the name of every getset descriptor on the C tensor base class.

    ``torch._C.TensorBase`` (``_TensorBase`` on legacy torch) is the C base that owns
    the property descriptors (``T``, ``mT``, ``H``, ``mH``, ``real``, ``imag``, ``data``,
    ``grad``, ``shape`` ...). Enumerating it structurally means a torch upgrade that adds
    a new tensor property is seen by the classifier automatically.
    """

    base = getattr(torch._C, "TensorBase", None) or getattr(torch._C, "_TensorBase", None)
    if base is None:  # pragma: no cover - torch always exposes the C tensor base here.
        return ()
    return tuple(
        name for name, obj in vars(base).items() if type(obj).__name__ == "getset_descriptor"
    )


@contextmanager
def _mode_free_probe_context() -> Iterator[None]:
    """Neutralize ambient torch execution modes for the import-time view probe.

    ``_pure_view`` allocates probe tensors and runs real tensor ops; the probe
    must be steered by NONE of the caller's ambient execution state, because this
    module is imported LAZILY -- the first ``tl.trace`` triggers the import, so the
    ambient torch state can be hostile:

    * a torch-FUNCTION mode (a default-device / meta / subclass mode) would make
      ``torch.randn`` build meta tensors and ``aten::equal`` raise; disabled via
      the feature-detected ``torch._C.DisableTorchFunction`` (CPU-device fallback
      on an unexpected build). Neither path mutates the caller's default device.
    * ``torch.inference_mode()`` (r55 corr_3): an inference tensor does NOT track a
      version counter, so ``probe._version`` raises
      ``RuntimeError: Inference tensors do not track version counter`` and the
      first inference-mode capture crashes before recording. Enter
      ``torch.inference_mode(False)`` + ``torch.enable_grad()`` so the probe
      always allocates a normal, version-tracked, autograd-live tensor.

    The neutralization is scoped to the probe only; the caller's ambient
    grad/inference/default-device state is untouched on exit.
    """

    with ExitStack() as stack:
        disabler = getattr(torch._C, "DisableTorchFunction", None)
        if disabler is not None:
            stack.enter_context(disabler())
        else:
            stack.enter_context(torch.device("cpu"))
        stack.enter_context(torch.inference_mode(False))
        stack.enter_context(torch.enable_grad())
        yield


def _pure_view(name: str) -> bool:
    """Return whether tensor property ``name`` is a safe pure-read view getter.

    A ``TensorBase`` getset descriptor is a safe pure-read view iff its getter, on a
    ``requires_grad`` source, returns a Tensor that (a) SHARES STORAGE with the source,
    (b) does NOT mutate the source (neither value nor version counter), and (c) PRESERVES
    autograd -- a ``requires_grad`` source yields a ``requires_grad`` view. Probed on both
    a real and a complex source (some descriptors, e.g. ``imag``, are only defined for one
    dtype); a descriptor undefined for a dtype is skipped, and admission requires at least
    one dtype to satisfy every clause.

    This lands EXACTLY on ``{T, mT, H, mH, real, imag}`` and DENIES ``data``: ``.data``
    shares storage but returns an autograd-DETACHED leaf (``requires_grad`` False, no
    ``grad_fn``) and is the canonical lvalue mutation channel (``x.data = y`` swaps storage
    under autograd/version tracking) -- exactly the surgery a faithful run-path must not
    bless. The rule is purely structural, so no per-name carve-out is needed to exclude it.
    """

    ok_any = False
    for use_complex in (False, True):
        # The probe runs at MODULE IMPORT time, which is LAZY here -- the first capture
        # imports this module, so the ambient torch state can be hostile: e.g. a forward
        # under ``torch.set_default_device("meta")`` pushes a torch-function DeviceContext
        # mode, which would make ``torch.randn`` build meta tensors and ``aten::equal``
        # raise ``NotImplementedError`` (there is no meta kernel). Neutralize any ambient
        # torch-function mode and pin CPU so the classification is deterministic and cannot
        # crash import. ``DisableTorchFunction`` does NOT alter the default device, so no
        # state leaks (verified: the caller's default device is untouched).
        with _mode_free_probe_context():
            if use_complex:
                probe = torch.randn(2, 3, device="cpu", dtype=torch.complex64, requires_grad=True)
            else:
                probe = torch.randn(2, 3, device="cpu", requires_grad=True)
            before = probe.detach().clone()
            version = probe._version
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = getattr(probe, name)
            except Exception:
                continue  # undefined / raising for this dtype -- not evidence either way.
            if not isinstance(result, torch.Tensor):
                return False  # non-tensor read (metadata / flag / hooks) -- never a view.
            if probe._version != version or not torch.equal(probe.detach(), before):
                return False  # the getter mutated the source -- not a pure read.
            try:
                shares_storage = (
                    result.untyped_storage().data_ptr() == probe.untyped_storage().data_ptr()
                )
            except Exception:
                return False  # storage-sharing unprovable -> refuse (fail closed).
            if not shares_storage:
                return False  # materialized a copy -- not a pure view.
            if probe.requires_grad and not result.requires_grad:
                return False  # autograd-DETACHING alias (``.data``) -- not a pure forward read.
            ok_any = True
    return ok_any


def _compute_pure_view_property_names() -> frozenset[str]:
    """Return the structurally-classified safe pure-view tensor property getter names."""

    return frozenset(name for name in _iter_tensor_getset_descriptor_names() if _pure_view(name))


_PURE_TENSOR_PROPERTY_NAMES: frozenset[str] = _compute_pure_view_property_names()


@lru_cache(maxsize=1)
def _torch_overridable_callable_ids() -> frozenset[int]:
    """Return the id-set of torch's canonical OVERRIDABLE / testing-override callables.

    This is torch's own authority on "is a genuine torch operator", independent of this
    module's (buggy) admission gate and self-updating across torch versions. Both the RAW
    callable id AND its capture-UNWRAPPED id are recorded, because
    ``get_overridable_functions()`` returns a MIX of wrapped and unwrapped forms once
    ``wrap_torch()`` has run (verified 2026-07: the ``relu`` entry set spans modules
    ``torch`` / ``torch.nn.functional`` / ``torchlens.backends.torch.wrappers``). Recording
    both makes the membership test correct regardless of when this ``lru_cache`` is first
    populated relative to wrapping, since ``is_pure_forward_callable`` always tests the
    UNWRAPPED identity. Built once and frozen: the overridable set is torch-version-fixed.
    """

    ids: set[int] = set()
    try:
        overridable = torch.overrides.get_overridable_functions()
    except Exception:  # pragma: no cover - defensive; torch.overrides always imports here.
        overridable = {}
    for funcs in overridable.values():
        for func in funcs:
            ids.add(id(func))
            ids.add(id(_unwrap_capture_wrapper(func)))
    try:
        testing = torch.overrides.get_testing_overrides()
    except Exception:  # pragma: no cover - defensive.
        testing = {}
    for func in testing:
        ids.add(id(func))
        ids.add(id(_unwrap_capture_wrapper(func)))
    return frozenset(ids)


def _has_aten_operator_schema(name: str) -> bool:
    """Return whether ``torch.ops.aten.<name>`` exposes a real operator schema.

    A non-empty overload set on the aten packet is torch's structural marker that
    ``name`` is a genuine dispatched operator (the surface a captured forward DAG
    resolves to). Fails SAFE: any lookup / introspection error yields ``False`` (the
    callable then falls through to deny on these roots).
    """

    if not name:
        return False
    try:
        packet = getattr(torch.ops.aten, name, None)
    except (AttributeError, RuntimeError):
        return False
    if packet is None:
        return False
    try:
        return len(packet.overloads()) > 0
    except Exception:
        return False


def _is_recognized_operator(real: Callable[..., Any], terminal_name: str) -> bool:
    """Return whether ``real`` is a structurally RECOGNIZED genuine forward operator.

    Admission on the exact operator-gated roots (``torch`` / ``torch._C`` /
    ``torch._tensor``): torch-overridable identity OR an aten operator schema OR a small
    stable pure tensor factory name OR a narrow audited pure Tensor wrapper name OR a
    safe pure-read tensor PROPERTY name (the loader's synthetic ``x.T`` / ``x.mT`` /
    ``x.H`` / ``x.mH`` / ``x.real`` / ``x.imag`` getters -- see
    ``_PURE_TENSOR_PROPERTY_NAMES``, structurally computed). Every
    other internal builtin on those roots is default-DENIED (the r43 inversion). Note the
    name/verb belts in ``_is_side_effecting_callable_name`` AND the r47 forward-dunder shape
    gate (``_is_denied_forward_dunder_name``, positive ``_ALLOWED_FORWARD_DUNDERS`` allowlist)
    run BEFORE this and catch the overridable-but-unsafe cases (e.g. ``share_memory_``, and the
    non-forward pickle-protocol ``__setstate__`` storage rebind), so identity-recognition here
    never re-admits a belt-denied or non-forward-dunder op.
    """

    if id(real) in _torch_overridable_callable_ids():
        return True
    if _has_aten_operator_schema(terminal_name):
        return True
    if terminal_name in _PURE_TENSOR_FACTORY_NAMES:
        return True
    if terminal_name in _PURE_TENSOR_WRAPPER_NAMES:
        return True
    return terminal_name in _PURE_TENSOR_PROPERTY_NAMES


# POSITIVE forward-dunder allowlist (r47, secE_1). The r43 recognized-operator gate
# admits a callable by torch-OVERRIDABLE identity / aten schema; a NON-forward dunder
# that satisfies overridable identity therefore slips (``Tensor.__setstate__`` -- the
# pickle-protocol state restorer whose legacy tuple form REBINDS the tensor's storage to
# an attacker donor with a fabricated size/stride, the SAME uninitialized/OOB heap-read
# class the storage belt denies ``set_`` / ``resize_`` / ``share_memory_`` for; and its 9
# siblings ``__reduce_ex__`` / ``__array__`` / ``__array_wrap__`` / ``__deepcopy__`` /
# ``__dlpack__`` / ``__dlpack_device__`` / ``__format__`` / ``__repr__`` / ``__reversed__``).
#
# A blanket dunder-DENY is WRONG: the run path resolves genuine FORWARD dunders. An
# exhaustive live ``torch.Tensor`` dunder sweep (torch 2.8, py3.11) found 14 Python-level
# forward dunders on the gated root ``torch._tensor`` alone -- ``__pow__`` / ``__floordiv__``
# / ``__rmatmul__`` / ``__rsub__`` / ``__rpow__`` / ``__rtruediv__`` / ``__rfloordiv__`` /
# ``__rlshift__`` / ``__rrshift__`` / ``__rmod__`` / ``__rdiv__`` / ``__ipow__`` /
# ``__len__`` / ``__contains__`` -- plus the arithmetic ``__add__`` / ``__mul__`` /
# ``__matmul__`` / ``__getitem__`` and their in-place / comparison / bitwise siblings on the
# module-less descriptor path. Denying those breaks replay (violates the LOCKED zero-
# forward-regression + validation-tripwire rules).
#
# So we ADMIT exactly the operator-protocol dunders -- arithmetic / reflected / in-place /
# comparison / bitwise / index-and-item / numeric-conversion -- and DENY every OTHER
# ``__x__`` by SHAPE, even when torch-OVERRIDABLE. Verified against the full live sweep:
# the allowlist yields ZERO forward regressions (every currently-admitted forward dunder
# is a member) and newly denies EXACTLY the 10 dangerous non-forward dunders above. The
# absent members (``__divmod__`` / ``__imatmul__`` / ``__rdivmod__`` / ``__round__`` /
# ``__trunc__`` / ``__floor__`` / ``__ceil__``) are future-compatible siblings that harm
# nothing when unbound. This is the same closure posture as the r43 ``share_memory_`` pin:
# an overridable-but-non-forward op denied by shape, not by growing an enumeration.
_ALLOWED_FORWARD_DUNDERS: frozenset[str] = frozenset(
    {
        # binary arithmetic + reflected + in-place
        "__add__",
        "__radd__",
        "__iadd__",
        "__sub__",
        "__rsub__",
        "__isub__",
        "__mul__",
        "__rmul__",
        "__imul__",
        "__matmul__",
        "__rmatmul__",
        "__imatmul__",
        "__truediv__",
        "__rtruediv__",
        "__itruediv__",
        "__div__",
        "__rdiv__",
        "__idiv__",
        "__floordiv__",
        "__rfloordiv__",
        "__ifloordiv__",
        "__mod__",
        "__rmod__",
        "__imod__",
        "__divmod__",
        "__rdivmod__",
        "__pow__",
        "__rpow__",
        "__ipow__",
        # bitwise
        "__lshift__",
        "__rlshift__",
        "__ilshift__",
        "__rshift__",
        "__rrshift__",
        "__irshift__",
        "__and__",
        "__rand__",
        "__iand__",
        "__or__",
        "__ror__",
        "__ior__",
        "__xor__",
        "__rxor__",
        "__ixor__",
        # unary arithmetic
        "__neg__",
        "__pos__",
        "__abs__",
        "__invert__",
        # comparison
        "__lt__",
        "__le__",
        "__eq__",
        "__ne__",
        "__gt__",
        "__ge__",
        # index / item / container-length / membership
        "__getitem__",
        "__setitem__",
        "__delitem__",
        "__len__",
        "__contains__",
        "__index__",
        # numeric conversion (pure value reads; no storage/callable/state side effect)
        "__int__",
        "__float__",
        "__complex__",
        "__bool__",
        "__nonzero__",
        "__long__",
        "__round__",
        "__trunc__",
        "__floor__",
        "__ceil__",
    }
)


def _is_denied_forward_dunder_name(name: str) -> bool:
    """Return whether a dunder terminal name is a NON-forward protocol method to DENY (r47).

    secE_1: a captured forward op node legitimately resolves to an OPERATOR-PROTOCOL dunder
    (arithmetic / reflected / in-place / comparison / bitwise / index-item / numeric-
    conversion -- ``_ALLOWED_FORWARD_DUNDERS``). EVERY OTHER ``__x__`` reaching the run-path
    reattach gate is a NON-forward protocol method (pickle/state ``__setstate__`` /
    ``__reduce_ex__``, copy ``__deepcopy__``, array/export ``__array__`` / ``__array_wrap__``
    / ``__dlpack__`` / ``__dlpack_device__``, stringify ``__format__`` / ``__repr__``, iter
    ``__reversed__``) a genuine forward op never names -- deny by SHAPE even when torch-
    OVERRIDABLE, closing the storage-rebind ``__setstate__`` class the name/storage belts
    miss (they key on ``set_`` prefix / ``resize`` / ``share_memory`` tokens, none of which a
    leading-``__`` dunder carries). Non-dunder names return ``False`` (untouched), so the
    ``operator`` / torch-function surface is unaffected.
    """

    return name.startswith("__") and name.endswith("__") and name not in _ALLOWED_FORWARD_DUNDERS


def is_pure_forward_callable(func: Callable[..., Any]) -> bool:
    """Return whether a resolved callable is a pure, side-effect-free forward op.

    The callable is unwrapped to its real identity, then admitted only if (a) its
    terminal NAME is not a side-effecting callable (file-I/O / serialization /
    import gadget, process-global-state mutator, or storage-unsafe in-place op --
    incl. the r43 ``share_memory_`` storage rebind), (a2) its terminal name, if a
    dunder, is one of the operator-protocol forward dunders (``_ALLOWED_FORWARD_DUNDERS``:
    arithmetic / reflected / in-place / comparison / bitwise / index-item / numeric-
    conversion) -- every OTHER ``__x__`` (the r47 secE_1 storage-REBIND ``__setstate__``
    and its pickle/copy/array/dlpack/format/repr/reversed siblings) is denied by shape
    even when torch-OVERRIDABLE, (b) its signature exposes no arbitrary-callable
    parameter, and (c) its module clears the gate.

    Module resolution (r43): the exact internal-builtin roots ``torch`` / ``torch._C``
    / ``torch._tensor`` are gated by the STRUCTURAL recognized-operator predicate
    (``_is_recognized_operator``) -- DEFAULT-DENY, admit ONLY torch-overridable
    identities, aten-schema ops, the small pure tensor factories, the audited
    ``to_sparse_coo`` wrapper, and the safe pure-read tensor property getters
    (``T`` / ``mT`` / ``real`` / ``imag``). This closes, as a CLASS, the non-forward internal
    builtins those roots host (functionalization dispatch controls, JIT / type
    constructors, Storage / legacy ``*Tensor`` ctors, process-global state getters,
    deprecated Tensor methods) that the pre-r43 module-prefix admission let through.
    DEEPER allowlisted operator submodules (``torch.nn.functional`` / ``torch._VF`` /
    ``torch._C._nn`` / ``torch.linalg`` / ...) keep their module-PREFIX admission. The
    ``operator`` / ``_operator`` root is gated separately by a POSITIVE NAME allowlist
    (``_ALLOWED_OPERATOR_NAMES``), so generic gadget / mutation primitives
    (``operator.call`` / ``attrgetter`` / ``methodcaller`` / ``itemgetter`` /
    ``setitem`` / ``delitem`` / ``iadd`` / ...) are default-denied while the pure
    arithmetic / comparison / bitwise / index operators still resolve. Module-less C
    tensor method descriptors are admitted when bound to a Tensor class (so the whole C
    tensor-op surface keeps resolving without reaching the root predicate).

    Anything else -- notably ``torch.load`` / ``torch.save`` / ``torch.from_file``,
    the functionalization controls ``_enable_functionalization`` /
    ``_functionalize_enable_reapply_views``, the state mutators ``set_default_dtype`` /
    ``manual_seed`` / ``set_num_threads``, the storage-unsafe ``resize_`` / ``set_`` /
    ``share_memory_``, JIT / IR type constructors, and any ``os`` / ``pickle`` /
    ``subprocess`` callable -- is refused. The name / signature guards run FIRST so a
    side-effecting builtin or method whose real module is an allowlisted ``torch``
    namespace (or ``None`` for a C tensor method) cannot slip the module gate.
    ``torch._sobol_engine_initialize_state_`` is admitted as a documented residual (a
    genuine aten operator; see ``_is_recognized_operator``).
    """

    real = _unwrap_capture_wrapper(func)
    if _is_side_effecting_callable_name(real):
        return False
    # r47 (secE_1): deny NON-forward protocol dunders by SHAPE (positive forward-dunder
    # allowlist). Runs BEFORE every module branch, so it covers BOTH the module-less
    # ``_is_tensor_method_descriptor`` path AND the gated-root ``_is_recognized_operator``
    # path -- closing ``Tensor.__setstate__`` (an overridable storage-REBIND the name/storage
    # belts miss because its leading-``__`` carries no ``set_`` / ``resize`` / ``share_memory``
    # token) and its pickle/copy/array/dlpack/format/repr/reversed siblings, while the genuine
    # forward operator dunders (``__add__`` / ``__mul__`` / ``__matmul__`` / ``__getitem__`` /
    # ``__pow__`` / ``__floordiv__`` / reflected ops / ``__len__`` / ``__contains__``) stay
    # admitted. Deliberately a DEDICATED helper, NOT folded into ``_is_side_effecting_callable_name``
    # (whose second caller ``is_inert_first_party_callable`` gates ``torchlens.*`` facet recipes).
    if _is_denied_forward_dunder_name(_terminal_callable_name(real)):
        return False
    # r41: deny higher-order / callback-taking ops by SIGNATURE shape (``torch.cond`` /
    # ``while_loop`` / ``handle_torch_function`` / ``triplet_margin_with_distance_loss`` /
    # ``_check_with`` / ``_disable_dynamo``) -- the same arbitrary-callable-INVOKE class as
    # the r39-denied ``vmap``, but carrying no ``map`` / ``register`` name marker. Runs
    # BEFORE the module gate so a callable-taking op whose real module is the allowlisted
    # ``torch`` namespace cannot slip. Fails safe: no-signature C ops fall through.
    if _signature_invokes_callable(real):
        return False
    module = str(getattr(real, "__module__", "") or "")
    if module == "":
        return _is_tensor_method_descriptor(real)
    if _matches(module, _DENIED_MODULES) or is_denied_stdlib_or_builtin_module(module):
        return False
    if module in _OPERATOR_MODULES:
        # POSITIVE allowlist for the operator root: only the pure arithmetic /
        # comparison / bitwise / index operators are admitted; every generic
        # gadget or mutation primitive (``call`` / ``attrgetter`` / ``setitem``
        # / ``iadd`` / ...) is default-denied.
        return _terminal_callable_name(real) in _ALLOWED_OPERATOR_NAMES
    # r43: on the EXACT internal-builtin roots (``torch`` / ``torch._C`` /
    # ``torch._tensor``) DEFAULT-DENY and admit ONLY structurally-recognized genuine
    # forward operators (overridable identity / aten schema / pure factory / audited
    # wrapper). This inverts the pre-r43 prefix admission on these roots that let every
    # non-forward internal builtin (functionalization controls, JIT/type ctors, state
    # getters, deprecated methods) through. Checked BEFORE the prefix ``_matches`` so
    # DEEPER allowlisted operator submodules (``torch.nn.functional`` / ``torch._VF`` /
    # ``torch._C._nn`` / ...) keep their module-prefix admission and no deep forward op
    # is over-denied.
    if module in _OPERATOR_GATED_ROOTS:
        return _is_recognized_operator(real, _terminal_callable_name(real))
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
