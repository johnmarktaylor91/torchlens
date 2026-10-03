"""Global state for torchlens toggle-gated decoration.

This module is the single source of truth for all mutable state that controls
whether decorated torch wrappers log or pass through.  It also stores
pre-computed lookup tables populated by ``decorate_all_once``.

WARNING — No torchlens imports at module level:
    Every other torchlens module imports from here.  If this module imported
    back, Python's import machinery would hit a circular dependency before any
    code ran.  Type-hint-only imports are safe inside ``TYPE_CHECKING`` guards
    because they are never evaluated at runtime.  The ONE sanctioned runtime
    exception is ``errors._base``: it is an import leaf (typing only, no
    torchlens imports), so pulling the taxonomy base classes from it can never
    close a cycle.

Design rationale:
    The "toggle architecture" means every torch function is wrapped once (on first
    use of ``trace`` or related API) and stays wrapped afterward.
    Wrappers check ``_logging_enabled`` (a single bool) on every call — when
    False, the wrapper is a one-branch-check no-op.  This avoids the cost of
    re-wrapping / un-wrapping on every ``trace`` call.  All shared
    state lives here so wrappers never need to import heavy torchlens modules
    just to check the toggle.

Access policy (disputed-r2 b5/R45, exempt-by-declaration):
    This module is the sanctioned shared toggle substrate. Direct READS of its
    published globals from other torchlens modules — including the bare
    ``_state._logging_enabled`` / ``_state._active_trace`` loads on the per-op
    wrapper hot path — are the documented design, not private-member
    reach-ins, and are exempt from private-access lint ratchets by this
    declaration. The exemption is pinned no-growth by
    ``tests/test_state_access_ratchet.py``: new cross-module access sites may
    not silently accumulate. NEW code should prefer the atomic
    ``active_capture()`` snapshot below over paired raw reads. Multi-field
    session-state TRANSITIONS (enable/disable, session setup/teardown) belong
    in state-owned functions and context managers here (``active_logging``,
    ``pause_logging``, ``reset_capture_runtime_context``, ...), not in
    external assignment clusters.
"""

import contextvars
import itertools
import threading
import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

# Sanctioned exception to the no-torchlens-imports rule (see module docstring):
# ``errors._base`` imports nothing from torchlens, so this cannot form a cycle.
from .errors._base import CaptureError

# TYPE_CHECKING is False at runtime, so this import only exists for static
# analysis / IDE support — it will never trigger the circular-import problem.
if TYPE_CHECKING:
    from .data_classes.trace import Trace
    from .intervention.types import InterventionSpec

# ---------------------------------------------------------------------------
# Toggle — the single bool that gates every decorated wrapper
# ---------------------------------------------------------------------------

_logging_enabled: bool = False
"""Master switch checked by every decorated torch-function wrapper.

When False (the default / steady state), wrappers execute the original function
directly with negligible overhead (one ``if`` check).  Set to True only for the
duration of a forward pass inside ``active_logging()``.
"""

# ---------------------------------------------------------------------------
# Session state — reset every forward pass
# ---------------------------------------------------------------------------

_active_owner_thread_id: int | None = None
"""Thread ident of the thread that entered ``active_logging()`` (r43 hon2_4).

TorchLens capture is single-owner-threaded by design (the aten census is a thread-local
``TorchDispatchMode``; the docs forbid concurrent captures). The torch-function wrapper,
however, is a GLOBAL decoration that fires on EVERY thread, so a NON-owner thread running
torch ops during a capture (a worker formatting ``str(tensor)``, a DataLoader thread) would
otherwise be logged into the owner's Trace -- tagging its temporaries with capture labels
(a false cross-thread ceiling) and corrupting owner-op attribution (an observed crash). The
wrapper op-logging fast-path skips any thread other than this owner. Cross-thread tensor->host
escapes are still observed by the mode-independent belt (which patches tensor methods directly,
independent of this wrapper). Set with ``_active_trace`` in ``active_logging`` and cleared on
exit.
"""

_nonowner_belt_armed: bool = False
"""Global gate for the r45 hon2_1 cross-thread captured-operand observer.

Set True (beside ``_WitnessState.belt_armed``) for exactly the runnable-capture forward
window inside ``_observe_invisible_host_escapes`` and cleared in its ``finally``. The global
torch-function wrapper's NON-owner fast path reads this single bool: when False (every plain
trace and the entire steady state) the wrapper stays a near-noop; when True a non-owner thread's
torch op is inspected for consumption of a captured operand (which ceilings replay proof to
``unverifiable``). A bare bool keeps ``_state.py`` import-free.
"""

_active_trace: "Trace | None" = None
"""The Trace accumulating data for the current forward pass.

Set at the start of ``active_logging()`` and cleared on exit.  Wrappers read
this to know *where* to record tensor operations.  Always None outside a
logging session.
"""

_active_fast_run_collector: Any | None = None
"""Explicit fast-run collector active around one native live-model forward.

The slot stays import-free so decorated torch wrappers can cheaply offer selected
functional-op collection without importing the fast-run implementation. It is non-``None``
only inside ``Trace.run(inputs=..., fast=True)`` and is restored in ``finally``.
"""

_rf_probe_depth: int = 0
"""Depth of in-flight receptive/projective-field gradient probes.

While positive, no wrapped autograd entry may mint a managed backward pass on
ANY trace and no capture-time tensor grad hook may record: probes are pure
measurements. The per-trace ``_tl_rf_probe_active`` flag alone cannot express
this — the probed trace can be a FORK, while the grad-fn registry and the
tensor hooks resolve to the BASE trace, whose flag is unset (a fork probe used
to mint a phantom ``autograd_grad`` pass with retained gradients on the
parent). Incremented/decremented by ``_probe_suppressed`` in
``receptive_field/_gradient.py``; single-threaded capture by design.
"""

_active_hook_plan: Any | None = None
"""Hook plan for the active intervention-ready capture.

Phase 4a only stores this slot. Hook normalization/execution is intentionally
deferred to Phase 4c, so the runtime value remains protocol-friendly and avoids
importing ``torchlens.intervention`` at module load.
"""

_active_intervention_spec: "InterventionSpec | None" = None
"""Intervention spec associated with the active capture, if any.

This module uses a string annotation plus a ``TYPE_CHECKING`` import so
``torchlens._state`` never imports the intervention package at runtime.
"""

_func_call_id_iter: "itertools.count[int]" = itertools.count(1)
"""Session-scoped monotonic function-call id source.

``next()`` on a C-level ``itertools.count`` is atomic under the GIL, so the
autograd engine threads that stamp ids during multi-device backward (one
engine thread per device: ``_ops_autograd.py``, ``collectives.py``) cannot
lose updates or mint duplicate ids -- the bare ``+= 1`` read-modify-write it
replaces could. Reset by rebinding a fresh counter at session start.
"""

_capture_replay_templates: bool = False
"""Whether the active capture should collect replay-template data.

Phase 4a only plumbs the flag. Phase 4b builds the actual templates.
"""

_relationship_model_id: int | None = None
"""Relationship evidence seed: ``id(model)`` at capture start."""

_relationship_model_class: str | None = None
"""Relationship evidence seed: model class qualname at capture start."""

_relationship_weight_fingerprint: str | None = None
"""Relationship evidence seed: deterministic parameter-structure fingerprint."""

_relationship_input_id: int | None = None
"""Relationship evidence seed: ``id(input_args)`` or first input tensor id."""

_relationship_input_shape_hash: str | None = None
"""Relationship evidence seed: deterministic input shape/dtype/device hash."""

_hook_reentrancy_depth: int = 0
"""Primitive hook execution depth mirrored by ``intervention.runtime``.

``_state`` owns this primitive instead of importing the runtime guard object,
which keeps this module free of runtime intervention imports.
"""

_log_registry: "weakref.WeakSet[Trace]" = weakref.WeakSet()
"""Process-wide weak registry of currently live ``Trace`` objects."""

_active_record_spans: "contextvars.ContextVar[tuple[dict[str, Any], ...]]" = contextvars.ContextVar(
    "_active_record_spans", default=()
)
"""Observer spans currently active around or inside a logging session.

Context-local (r5 b2-sol R54): as a plain process-global list, a two-thread
probe showed one thread's captures/taps receiving the OTHER thread's active
span annotations. A ``ContextVar`` holding an immutable tuple isolates spans
per thread/context; each thread starts from the empty default.
"""

_naming_counters: dict[str, int] = {}
"""Process-global counters used by unnamed ``trace`` captures.

Guarded by ``_naming_lock`` (r7 b8-sol R54): ``_auto_name`` runs during
capture setup BEFORE admission, so ``active_logging()``'s re-entrancy guard
does not serialize it -- two racing pre-admission threads could interleave the
read-modify-write and mint the SAME name for both captures (the refused loser
had already consumed the bump). The lock makes the get+increment atomic; it is
never held around user code or any other lock.
"""

_naming_lock = threading.Lock()
"""Serializes ``_naming_counters`` read-modify-write (see that docstring)."""

_HF_CLASS_SUFFIXES: tuple[str, ...] = (
    "ForCausalLM",
    "ForSequenceClassification",
    "ForMaskedLM",
    "ForQuestionAnswering",
    "ForTokenClassification",
    "ForImageClassification",
    "PreTrainedModel",
    "Model",
)
"""Common HuggingFace class suffixes stripped from generated log names."""


def register_log(log: "Trace") -> None:
    """Register a model log in the process-wide weak registry.

    The state-owned registry transition (R45 raw-access ratchet): callers
    spell it ``_state.register_log(...)`` rather than reaching into
    ``_log_registry`` directly.

    Parameters
    ----------
    log:
        Model log object to track weakly.

    Returns
    -------
    None
        The weak registry is updated in place.
    """

    _log_registry.add(log)


def unregister_log(log: "Trace") -> None:
    """Remove an unexposed transactional Trace from the live registry.

    The state-owned registry transition paired with ``register_log``.

    Parameters
    ----------
    log:
        Transactional model log that must no longer be discoverable.

    Returns
    -------
    None
        The weak registry is updated in place.
    """

    _log_registry.discard(log)


def list_logs() -> tuple["Trace", ...]:
    """Return a snapshot of currently live ``Trace`` objects.

    Returns
    -------
    tuple[Trace, ...]
        Immutable snapshot of logs still alive in this process.
    """

    snapshot = list(_log_registry)
    return tuple(log for log in snapshot if log is not None and hasattr(log, "layer_list"))


def _strip_hf_suffix(class_name: str) -> str:
    """Strip common HuggingFace suffixes from a model class name.

    Parameters
    ----------
    class_name:
        Model class name.

    Returns
    -------
    str
        Shortened class name when a known suffix matched.
    """

    for suffix in _HF_CLASS_SUFFIXES:
        if class_name.endswith(suffix) and len(class_name) > len(suffix):
            return class_name[: -len(suffix)]
    return class_name


def _auto_name(model: Any) -> str:
    """Return the next automatic name for a model instance.

    Parameters
    ----------
    model:
        PyTorch module-like object whose class name seeds the generated name.

    Returns
    -------
    str
        Lowercase short class name plus a monotonic counter.
    """

    class_name = type(model).__name__
    short = _strip_hf_suffix(class_name).lower()
    with _naming_lock:
        n = _naming_counters.get(short, 0) + 1
        _naming_counters[short] = n
    return f"{short}_{n}"


def reset_naming_counter(class_name: str | None = None) -> None:
    """Reset automatic log-name counters.

    Parameters
    ----------
    class_name:
        Lowercase short class name to reset, or ``None`` to reset all counters.

    Returns
    -------
    None
        The naming counter dictionary is mutated in place.
    """

    with _naming_lock:
        if class_name is None:
            _naming_counters.clear()
        else:
            _naming_counters.pop(class_name, None)


def reset_capture_runtime_context() -> None:
    """Reset per-capture intervention runtime context fields.

    Returns
    -------
    None
        The module-level runtime context is reset in place.
    """

    global _active_hook_plan, _active_intervention_spec, _func_call_id_iter
    global _capture_replay_templates
    global _relationship_model_id, _relationship_model_class
    global _relationship_weight_fingerprint, _relationship_input_id
    global _relationship_input_shape_hash

    _active_hook_plan = None
    _active_intervention_spec = None
    _func_call_id_iter = itertools.count(1)
    _capture_replay_templates = False
    _relationship_model_id = None
    _relationship_model_class = None
    _relationship_weight_fingerprint = None
    _relationship_input_id = None
    _relationship_input_shape_hash = None


def configure_capture_runtime_context(
    *,
    hook_plan: Any | None = None,
    intervention_spec: "InterventionSpec | None" = None,
    capture_replay_templates: bool = False,
    model_object_id: int | None = None,
    model_class_qualname: str | None = None,
    weight_fingerprint: str | None = None,
    input_object_id: int | None = None,
    input_signature_hash: str | None = None,
) -> None:
    """Set per-capture intervention runtime context fields.

    Parameters
    ----------
    hook_plan:
        Active hook plan placeholder. Execution is deferred to Phase 4c.
    intervention_spec:
        Active intervention spec placeholder. Mutators land in a later phase.
    capture_replay_templates:
        Whether replay-template capture should be enabled for this run.
    model_object_id:
        ``id(model)`` captured at the public API boundary.
    model_class_qualname:
        Model class qualname captured at the public API boundary.
    weight_fingerprint:
        Deterministic model-parameter fingerprint.
    input_object_id:
        Input object identity captured at the public API boundary.
    input_signature_hash:
        Deterministic input shape/dtype/device fingerprint.

    Returns
    -------
    None
        The module-level runtime context is updated in place.
    """

    global _active_hook_plan, _active_intervention_spec, _capture_replay_templates
    global _relationship_model_id, _relationship_model_class
    global _relationship_weight_fingerprint, _relationship_input_id
    global _relationship_input_shape_hash

    _active_hook_plan = hook_plan
    _active_intervention_spec = intervention_spec
    _capture_replay_templates = capture_replay_templates
    _relationship_model_id = model_object_id
    _relationship_model_class = model_class_qualname
    _relationship_weight_fingerprint = weight_fingerprint
    _relationship_input_id = input_object_id
    _relationship_input_shape_hash = input_signature_hash


def next_func_call_id() -> int:
    """Allocate the next session-scoped function-call id.

    Returns
    -------
    int
        Monotonic id for one decorated torch function invocation. Atomic
        (C-level ``next`` under the GIL), so concurrent autograd engine
        threads never observe a lost update or a duplicate id.
    """

    return next(_func_call_id_iter)


# ---------------------------------------------------------------------------
# Decoration state — tracks whether torch functions are currently wrapped
# ---------------------------------------------------------------------------

_is_decorated: bool = False
"""True when torch functions are currently wrapped with torchlens interceptors.

Set to True at the end of ``decorate_all_once()`` / ``wrap_torch()``, and
to False at the end of ``unwrap_torch()``.  Checked by ``_ensure_decorated()``
to decide whether (re-)decoration is needed before a logging session.
"""

_decorated_identity: Callable[..., Any] | None = None
"""Decorated version of the ``identity`` no-op, used at module boundaries.

When ``nn.Identity`` is encountered or a module's output tensor is the same
object as its input, ``_decorated_identity(t)`` forces a new log entry so the
graph correctly shows the module boundary.  Stored here instead of on
``torch.identity`` to avoid monkey-patching an attribute that doesn't exist
in PyTorch's type stubs.
"""

# ---------------------------------------------------------------------------
# Pre-computed lookup tables (populated once by decorate_all_once, immutable after)
# ---------------------------------------------------------------------------
# These dicts are written exactly once during ``decorate_all_once()`` and are
# treated as read-only afterward.  They exist here (not in decoration/) so that
# wrapper code can look up argument names and original functions without
# importing the decoration subpackage.

_arg_names: dict[str, tuple[str, ...]] = {}
"""func_name -> tuple of argument names, pre-computed via ``inspect.signature``
for every torch function at decoration time.  Used by the wrapper to build
keyword-argument metadata for logged operations.
"""

_orig_to_decorated: dict[int, Callable[..., Any]] = {}
"""id(original_func) -> decorated wrapper.  Used by the rescue net
(``RescueTorchFunctionMode``) and the mechanical belt to redirect stale
pre-wrap references to their exact wrappers.  Keyed by id() for O(1) lookup.
"""

_decorated_to_orig: dict[int, Callable[..., Any]] = {}
"""id(decorated_func) -> original_func.  The reverse of ``_orig_to_decorated``.
Keyed by id() for fast lookup when a wrapper needs the unwrapped callable.
"""

# Also keep a version keyed by the decorated func object itself (not id),
# for use in model_funcs where we need ``func in decorated_func_mapper``
# (i.e. the ``in`` operator needs the actual object, not its id).
_decorated_func_mapper: dict[Callable[..., Any], Callable[..., Any]] = {}
"""Bidirectional map: decorated -> original AND original -> decorated.

Keyed by actual callable objects (not ids) so that ``func in _decorated_func_mapper``
works.  Used in model_funcs to determine whether a callable is already wrapped.
"""

# ---------------------------------------------------------------------------
# Introspection cache
# ---------------------------------------------------------------------------

_dir_cache: dict[type, list[str]] = {}
"""Per-type cache of filtered ``dir()`` results for ``extend_search_stack_from_item``."""


_wrap_epoch: int = 0
"""Monotonic wrapper lifecycle counter; bumps on every ``wrap_torch()`` install."""

_escape_detector_mode: str = "off"
"""Callable escape detector mode: ``"off"`` or diagnostic ``"shadow"``."""

_completeness_witness_mode: str = "off"
"""Dispatcher completeness witness mode: ``"off"`` or diagnostic ``"shadow"``.

OFF BY DEFAULT (AUD-HONESTY H4): a plain ``tl.trace()`` never arms the aten-dispatch
census, so ``Trace.capture_verified`` stays ``None`` (NOT RECORDED -- never a clean
bill) and host escapes only the census can see (``from_numpy``/``as_tensor``/
``from_dlpack`` round-trips, numpy-backed ``autograd.Function`` forwards, in-forward
storage writes) leave no ceiling. The ``tl.validate`` paths flip it to ``"shadow"``
for their own run, and ``wrap_torch(completeness_witness=True)`` arms it for every
capture; only an armed census can produce ``capture_verified=True``. Flipping the
default on is a maintainer fork (cost + observer-effect trade), not a lane decision.
"""

_runnable_ledger_armed: bool = False
# Private wave-0 ATen recorder edge-token arm. It is capture-scoped and never portable.
_aten_recording_armed: bool = False
"""Whether the r35 event-lifecycle ledger requires wrapper ownership tokens.

Armed only around a runnable-eligible (``intervention_ready``) capture forward so
the dispatch census can attribute raised / host-returning aten events to their
exact wrapper owner even when both diagnostic shadow modes are off.
"""
_prepared_models: weakref.WeakSet[Any] = weakref.WeakSet()
"""Models that have already been through ``_prepare_model_once()``.

Using a WeakSet ensures that if the user discards a model, it can be
garbage-collected without this set holding a strong reference.  Membership
here means the model's forward and submodule hooks are already installed.
"""

# ---------------------------------------------------------------------------
# Module role-swap tracking (root vs non-root, per prepared tree)
# ---------------------------------------------------------------------------
# ``_prepare_model_once`` assigns role-DEPENDENT permanent metadata: a module's
# dotted address (root-relative) and, for non-root modules, a toggle-gated
# ``forward`` decoration. The root's ``forward`` is deliberately left UNDECORATED
# because ``trace`` invokes and frames it separately. That assignment silently
# assumed a module's root/non-root role never changes across its lifetime.
#
# It can: the SAME module can be traced as a non-root submodule in one trace and
# as its own top-level root in a later trace (or vice versa). When that happens,
# the metadata cached for the old role is stale for the new one -- a decorated
# forward run as a root crashes ``push_frame`` (the root is never registered in
# the per-session module-call dicts), and a descendant re-rooted under a new
# model leaves its ancestor's cached addresses pointing at the wrong root.
#
# These two structures let ``_prepare_model_once`` keep its O(1) cache fast path
# for the overwhelmingly common case (a model traced repeatedly in a fixed role,
# or independent models traced in any interleaving) while forcing a correct
# re-preparation of exactly the trees whose role assignment went stale.

_prepared_root_by_module: "weakref.WeakKeyDictionary[Any, weakref.ref[Any]]" = (
    weakref.WeakKeyDictionary()
)
"""module -> weakref to the ROOT model it was last prepared under.

Weak on both sides: the key (module) is weakly held by the WeakKeyDictionary,
and the value is a ``weakref.ref`` to the root, so neither keeps the other alive.
Read/written only through the helpers below during ``_prepare_model_once``.
"""

_stale_prepared_roots: "weakref.WeakSet[Any]" = weakref.WeakSet()
"""Prepared root models whose cached role metadata is known to be stale.

A root lands here when some module in its subtree is later re-prepared under a
DIFFERENT root (role swap). ``_prepare_model_once`` treats a stale root as
un-prepared for the fast-path check and re-establishes its tree's addresses and
forward decorations before clearing the staleness flag.
"""


def record_module_root_prep(root: Any, module: Any) -> None:
    """Stamp ``module`` as prepared under ``root``; flag a displaced prior root.

    If ``module`` was previously prepared under a different (still-live) root,
    that prior root's cached tree now has a re-rooted descendant and is marked
    stale so its next preparation re-establishes correct role metadata.

    Parameters
    ----------
    root:
        The root model whose ``_prepare_model_once`` traversal is running.
    module:
        A module (the root itself or any descendant) being (re)prepared.

    Returns
    -------
    None
        ``_prepared_root_by_module`` and ``_stale_prepared_roots`` are updated in
        place.
    """
    previous = _prepared_root_by_module.get(module)
    if previous is not None:
        previous_root = previous()
        if previous_root is not None and previous_root is not root:
            _stale_prepared_roots.add(previous_root)
    _prepared_root_by_module[module] = weakref.ref(root)


def root_prep_is_stale(root: Any) -> bool:
    """Return whether ``root``'s cached role metadata is known to be stale.

    Parameters
    ----------
    root:
        Candidate root model.

    Returns
    -------
    bool
        ``True`` when a descendant of ``root`` was re-rooted under another model
        since ``root`` was last prepared.
    """
    return root in _stale_prepared_roots


def clear_root_prep_stale(root: Any) -> None:
    """Clear the staleness flag for ``root`` after re-preparing its tree.

    Parameters
    ----------
    root:
        Root model that has just been (re)prepared.

    Returns
    -------
    None
        ``_stale_prepared_roots`` is updated in place.
    """
    _stale_prepared_roots.discard(root)


def release_model_prep(root: Any, modules: tuple[Any, ...]) -> None:
    """Evict a released module tree from persistent preparation bookkeeping.

    Parameters
    ----------
    root:
        Root model passed to the public release operation.
    modules:
        Current full module tree rooted at ``root``.

    Returns
    -------
    None
        Preparation and role-swap registries are updated in place.

    Notes
    -----
    A current descendant may have last been prepared beneath another root after
    a role swap. Those displaced roots are evicted too: releasing the shared
    descendant removes its forward wrapper, so their next capture must rebuild
    the full role-dependent preparation state.
    """
    released_modules = set(modules)
    affected_roots = {root}
    entries_to_remove: list[Any] = []
    for module, prepared_root_ref in list(_prepared_root_by_module.items()):
        prepared_root = prepared_root_ref()
        if module in released_modules or prepared_root is root:
            entries_to_remove.append(module)
            if prepared_root is not None:
                affected_roots.add(prepared_root)

    for module in entries_to_remove:
        _prepared_root_by_module.pop(module, None)
    for affected_root in affected_roots:
        _prepared_models.discard(affected_root)
        _stale_prepared_roots.discard(affected_root)


# ---------------------------------------------------------------------------
# Usage stats — opt-in per-function call counting for coverage analysis
# ---------------------------------------------------------------------------

_collect_usage_stats: bool = False
"""When True, every decorated wrapper increments call counts in
``_function_call_counts`` during logged forward ops.  Used by the
test suite to verify ArgSpec lookup table coverage."""

_functorch_warning_emitted: bool = False
"""True if a warning about skipped functorch/vmap ops has been emitted for
the current logging session.  Reset to False at the start of every
``active_logging()`` context so each forward pass gets at most one warning."""

_dynamo_warning_emitted: bool = False
"""True if a warning about ops skipped inside a Dynamo-traced region has been
emitted for the current logging session.  Reset to False at the start of every
``active_logging()`` context so each forward pass gets at most one warning."""

_function_call_counts: dict[str, int] = {}
"""func_name -> total calls across all logged forward ops."""

_function_call_models: dict[str, set[str]] = {}
"""func_name -> set of model names that called this function."""

_current_model_name: str = ""
"""Set by test fixtures to identify which model is being logged."""

# ---------------------------------------------------------------------------
# Dynamic ArgSpec cache — Tier 3 of the O(1) extraction strategy
# ---------------------------------------------------------------------------

_dynamic_arg_specs: dict[str, object] = {}
"""Normalized func_name -> ArgSpec, populated by BFS fallback on first
call to an uncovered function.  Subsequent calls reuse the cached spec."""

# ---------------------------------------------------------------------------
# Tagged tensor tracking — for fast cleanup
# ---------------------------------------------------------------------------

_tagged_buffer_ids: set[int] = set()
"""ids of tensors tagged with _tl.address during prepare_buffer_tensors.
Used by _undecorate_model_tensors for O(n) cleanup instead of re-scanning
all module attributes."""

# ---------------------------------------------------------------------------
# Context managers
# ---------------------------------------------------------------------------


class ReentrantTraceError(CaptureError, RuntimeError):
    """Raised when a TorchLens trace is started while another trace is active.

    Part of the typed taxonomy (catchable as ``tl.errors.CaptureError``) while
    keeping ``RuntimeError`` in the MRO so historical ``except RuntimeError``
    handlers keep working. Structured context on ``fields``: ``code`` is always
    ``"reentrant_trace"``, ``remedy`` names the fix, and ``active_model``
    carries the label of the capture already running (``None`` when unknown).
    """


_capture_reserved_by: int | None = None
"""Thread ident holding the pre-admission capture RESERVATION, or ``None``.

The admission lock makes ``active_logging`` publication atomic, but a capture's
GLOBAL side effects start earlier: model preparation swaps the per-capture label
session and ``tl.record`` installs the fastlog ``RecordingState`` BEFORE the
forward reaches admission. A concurrent capture that is ultimately REFUSED
therefore used to degrade the admitted winner's data quality (runtime-probed:
orphaned label stamps, ``capture_verified=False``). ``capture_reservation()``
moves the typed refusal in front of those side effects: the reservation is
claimed under ``_capture_admission_lock`` before any capture-global mutation,
``active_logging`` admits only the reserving thread (or an unreserved caller),
and the loser's ``ReentrantTraceError`` fires before it can touch shared state.
Written only under the admission lock; never read on the wrapper hot path.
"""

_capture_admission_lock = threading.Lock()
"""Serializes capture ADMISSION and teardown bookkeeping (never the forward).

``active_logging`` reads the "is a capture already running?" predicate and then
publishes ``_active_trace`` / ``_active_owner_thread_id`` / ``_logging_enabled``.
Those are separate bytecodes: without a lock two threads entering together can
both pass the check, and the loser overwrites the winner's owner id -- after
which the winner's ops are dropped by the owner-thread fast path and its Trace
is silently short. Holding this lock across check-then-publish makes admission
atomic, so exactly one of N racing captures is admitted and the rest get the
documented ``ReentrantTraceError``. It is held for a handful of assignments
once per capture (never for the forward pass, never around user code), so it
costs nothing measurable and cannot deadlock: no other lock is acquired under
it, and it is never re-entered (a nested capture is refused before publishing).

The wrapper hot path deliberately does NOT take this lock -- it reads the
published globals unsynchronized, exactly as before. The lock closes the
admission race, not the (documented, unsupported) concurrent-capture case.
"""


def _reentrant_refusal() -> ReentrantTraceError:
    """Build the typed concurrent-capture refusal (call under the admission lock).

    Returns
    -------
    ReentrantTraceError
        Refusal naming the active model when one is identifiable.
    """

    active_model = getattr(_active_trace, "model_label", None)
    if active_model is None:
        active_model = getattr(_active_trace, "model_class_name", None)
    active_model_text = f" for active model {active_model!r}" if active_model else ""
    return ReentrantTraceError(
        "torchlens.trace / active_logging is not re-entrant: "
        f"another forward pass{active_model_text} is already being logged. Nested logging "
        "would silently corrupt the outer Trace. Remedy: finish the outer "
        "capture before starting another one (e.g. return from the custom "
        "activation_transform or hook that called tl.trace).",
        code="reentrant_trace",
        remedy="finish the outer capture before starting another one",
        active_model=active_model,
    )


def _capture_conflict_is_live() -> bool:
    """Return whether a capture (or its pre-admission window) conflicts (lock held).

    A live toggle, a published trace, primitive hook depth, or a reservation
    held by ANOTHER thread all refuse; this thread's own reservation is the
    sanctioned path into ``active_logging`` and does not conflict.
    """

    if _logging_enabled or _active_trace is not None or _hook_reentrancy_depth > 0:
        return True
    return _capture_reserved_by is not None and _capture_reserved_by != threading.get_ident()


_capture_reservation_token: object | None = None
"""Opaque continuation token minted with the live reservation claim.

Same-thread re-entry into ``capture_reservation`` is sanctioned for exactly
one caller: the capture orchestration invoked BY the reserving recorder pass,
which receives this token from the recorder and presents it back. A nested
PUBLIC capture entered from user code running inside the reserved window
(input-walk container protocols, model-prep hooks, tensor-subclass
``__torch_function__`` during input setup) holds no token, so the thread-ident
check alone must never admit it (R55: both captures used to COMPLETE).
Written only under the admission lock.
"""


@contextmanager
def capture_reservation(resume: object | None = None) -> Iterator[object]:
    """Reserve the capture slot BEFORE any capture-global side effect runs.

    Entered at the top of a public capture (``tl.trace`` orchestration,
    ``tl.record``'s recorder pass) so a concurrent capture is refused typed
    BEFORE it can sweep the admitted capture's label session or overwrite the
    fastlog ``RecordingState`` (the refused-loser data-quality corruption).

    Yields the reservation's continuation token. Same-thread re-entry is a
    passthrough ONLY when ``resume`` presents the live token: the recorder
    reserves around ``active_recording_state``, hands the yielded token to the
    inner orchestration, and that orchestration re-enters here before
    ``active_logging`` without releasing the outer claim. A same-thread entry
    WITHOUT the token is a nested public capture started by user code inside
    the reserved window and refuses typed (R55) -- the bare thread-ident
    passthrough used to let both captures run to completion. A genuinely
    nested capture (inside a live forward) refuses on the same predicate as
    ``active_logging``.
    """

    global _capture_reserved_by, _capture_reservation_token
    ident = threading.get_ident()
    with _capture_admission_lock:
        if _logging_enabled or _active_trace is not None or _hook_reentrancy_depth > 0:
            raise _reentrant_refusal()
        if _capture_reserved_by is None:
            _capture_reserved_by = ident
            _capture_reservation_token = object()
            token = _capture_reservation_token
            owns_reservation = True
        elif _capture_reserved_by == ident:
            if resume is None or resume is not _capture_reservation_token:
                raise _reentrant_refusal()
            token = _capture_reservation_token
            owns_reservation = False
        else:
            raise _reentrant_refusal()
    try:
        yield token
    finally:
        if owns_reservation:
            with _capture_admission_lock:
                _capture_reserved_by = None
                _capture_reservation_token = None


@contextmanager
def publish_active_trace(trace: "Trace") -> Iterator[None]:
    """Admission-locked ``_active_trace`` publication for a non-forward window.

    The sanctioned spelling for every window that must make a trace globally
    visible WITHOUT the logging toggle: preview-backend captures (tf), derived
    gradient replays (paddle), and backward projection. Raw save/restore swaps
    of ``_state._active_trace`` bypassed admission entirely -- a tf capture
    concurrent with a torch capture silently rebound the torch wrapper's
    target trace, and the ``finally`` restore could republish a since-finished
    trace, wedging every later capture's admission check. This helper refuses
    typed under the admission lock (same predicate as ``active_logging``),
    sets ``_active_owner_thread_id`` so the r43 non-owner ``pause_logging``
    protection covers the window, and clears to ``None`` on exit (a refused
    entry proves there was no previous trace to restore).
    """

    global _active_trace, _active_owner_thread_id
    with _capture_admission_lock:
        if _capture_conflict_is_live():
            raise _reentrant_refusal()
        _active_trace = trace
        _active_owner_thread_id = threading.get_ident()
    try:
        yield
    finally:
        with _capture_admission_lock:
            _active_trace = None
            _active_owner_thread_id = None


class _BackwardCapturePublication:
    """Admission-locked publish/restore handle for a torch backward window.

    ``_run_backward_with_capture`` historically raw-swapped ``_active_trace``
    (plus the hook plan and intervention spec) with an unlocked save/restore
    — the single site left unconverted when tf/paddle moved to
    ``publish_active_trace``. Interleaved with a concurrent capture, the
    unlocked read could snapshot that capture's live trace as "previous" and
    the ``finally`` could republish it after the capture had finished,
    leaving ``_active_trace`` permanently non-``None`` — every later
    capture's admission check then refuses (a process-global wedge).

    ``publish_active_trace`` itself cannot be reused verbatim: backward
    windows legitimately NEST on one thread (the multi-trace backward
    bracket, an inner ``backward()`` inside a traced forward), so this handle
    keeps the exact same-thread save/restore semantics of the raw swap while
    refusing a window owned by ANOTHER thread typed under the admission lock.
    Cross-thread "previous" snapshots are therefore impossible, which is what
    kills the wedge.

    Not a context manager: both backward unwind arms must restore the
    globals FIRST and then run fallible cleanup, so the owner calls
    ``restore()`` explicitly. ``restore()`` is idempotent and safe against a
    double-restore from stacked unwind arms.
    """

    __slots__ = ("_prev_trace", "_prev_owner", "_prev_plan", "_prev_spec", "_restored")

    def __init__(
        self,
        prev_trace: "Trace | None",
        prev_owner: int | None,
        prev_plan: Any,
        prev_spec: "InterventionSpec | None",
    ) -> None:
        """Snapshot the previous owner globals (caller holds the admission lock)."""

        self._prev_trace = prev_trace
        self._prev_owner = prev_owner
        self._prev_plan = prev_plan
        self._prev_spec = prev_spec
        self._restored = False

    def restore(self) -> None:
        """Restore the snapshotted owner globals under the admission lock."""

        global _active_trace, _active_owner_thread_id
        global _active_hook_plan, _active_intervention_spec
        if self._restored:
            return
        with _capture_admission_lock:
            if self._restored:
                return
            _active_trace = self._prev_trace
            _active_owner_thread_id = self._prev_owner
            _active_hook_plan = self._prev_plan
            _active_intervention_spec = self._prev_spec
            self._restored = True


def publish_backward_capture(
    trace: "Trace",
    *,
    hook_plan: Any,
    intervention_spec: "InterventionSpec | None",
) -> _BackwardCapturePublication:
    """Publish a backward capture window; refuse a foreign live window typed.

    Same-thread nesting (an already-published capture or backward window
    owned by THIS thread) is the sanctioned multi-trace/nested-backward path
    and keeps save/restore semantics; a window owned by another thread — or a
    capture reservation held by another thread — raises the same typed
    ``ReentrantTraceError`` admission uses, instead of silently corrupting
    the other thread's capture.

    Returns
    -------
    _BackwardCapturePublication
        Handle whose ``restore()`` puts the previous owner globals back.
    """

    global _active_trace, _active_owner_thread_id
    global _active_hook_plan, _active_intervention_spec
    ident = threading.get_ident()
    with _capture_admission_lock:
        window_live = _active_trace is not None or _logging_enabled
        if window_live and _active_owner_thread_id != ident:
            raise _reentrant_refusal()
        if _capture_reserved_by is not None and _capture_reserved_by != ident:
            raise _reentrant_refusal()
        publication = _BackwardCapturePublication(
            _active_trace,
            _active_owner_thread_id,
            _active_hook_plan,
            _active_intervention_spec,
        )
        _active_trace = trace
        _active_owner_thread_id = ident
        _active_hook_plan = hook_plan
        _active_intervention_spec = intervention_spec
    return publication


@contextmanager
def active_logging(trace: "Trace") -> Iterator[None]:
    """Activate logging for the duration of a forward pass.

    Sets ``_logging_enabled = True`` and ``_active_trace = trace``.
    On exit (including exceptions), resets both.

    Ordering invariant:
        - On entry: set ``_active_trace`` *before* the toggle, so wrappers
          never see ``_logging_enabled=True`` with a stale/None trace.
        - On exit: clear the toggle *before* the trace, for the same reason.

    This context manager is NOT nestable.  Only one forward pass may be logged
    at a time (single-threaded design).  Entering a second ``active_logging``
    while another is already active raises ``ReentrantTraceError`` — silently
    corrupting the outer log (overwriting ``_active_trace`` and then
    clearing it on inner exit) is worse than failing loudly.
    """
    global _logging_enabled, _active_trace, _functorch_warning_emitted, _func_call_id_iter
    global _dynamo_warning_emitted
    global _active_owner_thread_id
    # Admission is atomic: the refusal check and the publication of the three
    # owner globals happen under one lock, so two threads entering together
    # cannot both be admitted (see ``_capture_admission_lock``).
    with _capture_admission_lock:
        if _capture_conflict_is_live():
            raise _reentrant_refusal()
        # Model log must be visible before the toggle flips — wrappers will
        # immediately read _active_trace once _logging_enabled is True.
        _active_trace = trace
        _active_owner_thread_id = threading.get_ident()
        _functorch_warning_emitted = False
        _dynamo_warning_emitted = False
        _func_call_id_iter = itertools.count(1)
        _logging_enabled = True
    try:
        yield
    finally:
        with _capture_admission_lock:
            # Toggle off first so no wrapper sees enabled=True with trace=None
            _logging_enabled = False
            _active_trace = None
            _active_owner_thread_id = None


class _PauseLogging:
    """One-shot context manager that pauses the logging toggle.

    A plain ``__slots__`` class instead of a ``@contextmanager`` generator:
    ``pause_logging()`` is entered tens of thousands of times per trace, and the
    generator machinery (``_GeneratorContextManager.__init__`` + ``next``/throw
    dispatch in ``__enter__``/``__exit__``) was several percent of capture wall
    time. Semantics are identical: the toggle state is saved at ``__enter__``
    (not at construction), restored unconditionally on exit — including on
    exception, matching the generator's ``finally`` — and exceptions are never
    suppressed. Nesting works because every ``pause_logging()`` call returns a
    fresh instance with its own saved state.

    A pause entered from a NON-OWNER thread while a capture is live is a no-op:
    the toggle belongs to the owner's forward pass, and clearing it from another
    thread blinds that capture (ops silently missing, no error). This is the
    general form of the r43 fix that ``materialize_deferred_for_call`` applied at
    one call site; every one of the ~40 ``pause_logging()`` sites is covered here,
    including the ones reachable with NO concurrent capture at all -- a thread
    merely analyzing an older Trace (``tl.save``, validation, an ``.out``
    transform) while another thread captures.
    """

    __slots__ = ("_prev", "_owns_toggle")

    def __enter__(self) -> None:
        global _logging_enabled
        owner = _active_owner_thread_id
        if owner is not None and owner != threading.get_ident():
            # Live capture owned by a different thread: do not touch the global.
            self._owns_toggle = False
            self._prev = False
            return
        self._owns_toggle = True
        self._prev = _logging_enabled  # save current state (True or False)
        _logging_enabled = False

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        global _logging_enabled
        if not self._owns_toggle:
            # Symmetric no-op: a stale restore from a non-owner thread could
            # re-enable logging after the owner's capture already finished.
            return
        # b2:A2 remnant (r5 fable R54): re-check ownership at RESTORE time.
        # This thread can have read owner=None an instant before another
        # thread's locked capture publication; restoring the stale pre-pause
        # value here would then silently blind the remainder of that
        # capture's forward. If a different thread owns the toggle now, the
        # publication already set the value it needs -- leave it alone.
        owner = _active_owner_thread_id
        if owner is not None and owner != threading.get_ident():
            return
        _logging_enabled = self._prev  # restore — enables nesting without corruption


def pause_logging() -> _PauseLogging:
    """Temporarily disable logging so internal torch ops don't get recorded.

    Nestable via save/restore: if already paused, restoring ``prev`` (False)
    is a harmless no-op.  If logging was active, it resumes on exit.

    This intentionally does NOT clear ``_active_trace``. The active log
    remains visible while the toggle is paused so ``active_logging()`` can still
    reject nested captures inside paused internal work.

    Typical callers:
        - ``safe_copy``: copies tensors without logging the copy op
        - ``activation_transform``: applies user post-processing without logging
    """
    return _PauseLogging()


@contextmanager
def aten_recording(enabled: bool = True) -> Iterator[None]:
    """Arm the primitive-op wrapper edge for one nested capture window.

    Parameters
    ----------
    enabled:
        Whether this window requests primitive-op ownership tokens. A false
        nested request preserves an already-armed outer window.

    Yields
    ------
    None
        The caller runs with the requested ATen edge state installed.
    """

    global _aten_recording_armed
    previous = _aten_recording_armed
    _aten_recording_armed = previous or enabled
    try:
        yield
    finally:
        _aten_recording_armed = previous


def diagnostic_observer_armed() -> bool:
    """Return whether any wrapper-edge diagnostic observer is armed.

    Returns
    -------
    bool
        ``True`` when wrappers must mint an exact ownership token.
    """

    return (
        _escape_detector_mode == "shadow"
        or _completeness_witness_mode == "shadow"
        or _runnable_ledger_armed
        or _aten_recording_armed
    )


def active_capture() -> "tuple[Trace | None, bool]":
    """Return one coherent ``(active trace, logging enabled)`` snapshot.

    The SANCTIONED NEW-CODE spelling for reading the capture toggle pair
    (module access policy above): reads ``_logging_enabled`` BEFORE
    ``_active_trace``, so under the ``active_logging`` ordering invariant
    (trace published before the toggle flips; toggle cleared before the trace)
    an enabled snapshot always carries the live trace, never a stale or
    ``None`` one. Existing raw reads are exempt by declaration and are not
    migrated; hot wrapper paths may keep single-field raw loads.

    Returns
    -------
    tuple[Trace | None, bool]
        Active trace (or ``None``) and whether logging is currently enabled.
    """

    enabled = _logging_enabled
    return _active_trace, enabled


def wrap_epoch_ledgers() -> "tuple[dict[int, Callable[..., Any]], dict[int, Callable[..., Any]]]":
    """Return the ``(decorated -> orig, orig -> decorated)`` unwrap ledgers.

    The SANCTIONED NEW-CODE spelling for ledger-fenced wrapper/original
    resolution outside the hot wrapper paths (R45 raw-access ratchet):
    callers receive the live id-keyed ledgers for READ-ONLY lookup.
    ``_decorated_to_orig`` is the append-only unwrap ledger — never mutate
    or clear it through this accessor (or at all; see the module policy).

    Returns
    -------
    tuple[dict[int, Callable[..., Any]], dict[int, Callable[..., Any]]]
        ``_decorated_to_orig`` and ``_orig_to_decorated``, in that order.
    """

    return _decorated_to_orig, _orig_to_decorated
