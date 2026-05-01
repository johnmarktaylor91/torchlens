"""Global state for torchlens toggle-gated decoration.

This module is the single source of truth for all mutable state that controls
whether decorated torch wrappers log or pass through.  It also stores
pre-computed lookup tables populated by ``decorate_all_once``.

WARNING — No torchlens imports at module level:
    Every other torchlens module imports from here.  If this module imported
    back, Python's import machinery would hit a circular dependency before any
    code ran.  Type-hint-only imports are safe inside ``TYPE_CHECKING`` guards
    because they are never evaluated at runtime.

Design rationale:
    The "toggle architecture" means every torch function is wrapped once (on first
    use of ``log_forward_pass`` or related API) and stays wrapped afterward.
    Wrappers check ``_logging_enabled`` (a single bool) on every call — when
    False, the wrapper is a one-branch-check no-op.  This avoids the cost of
    re-wrapping / un-wrapping on every ``log_forward_pass`` call.  All shared
    state lives here so wrappers never need to import heavy torchlens modules
    just to check the toggle.
"""

import weakref
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, Optional, Set

# TYPE_CHECKING is False at runtime, so this import only exists for static
# analysis / IDE support — it will never trigger the circular-import problem.
if TYPE_CHECKING:
    from .data_classes.model_log import ModelLog
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

_active_model_log: Optional["ModelLog"] = None
"""The ModelLog accumulating data for the current forward pass.

Set at the start of ``active_logging()`` and cleared on exit.  Wrappers read
this to know *where* to record tensor operations.  Always None outside a
logging session.
"""

_active_hook_plan: Any | None = None
"""Hook plan for the active intervention-ready capture.

Phase 4a only stores this slot. Hook normalization/execution is intentionally
deferred to Phase 4c, so the runtime value remains protocol-friendly and avoids
importing ``torchlens.intervention`` at module load.
"""

_pending_live_fire_records: dict[str, list[Any]] = {}
"""Live hook fire records keyed by capture-time raw label.

Wrappers create these records before ``LayerPassLog`` construction so hooks can
run before activation saving. ``capture.output_tensors`` consumes them after the
real log entry exists.
"""

_active_intervention_spec: "InterventionSpec | None" = None
"""Intervention spec associated with the active capture, if any.

This module uses a string annotation plus a ``TYPE_CHECKING`` import so
``torchlens._state`` never imports the intervention package at runtime.
"""

_func_call_id_counter: int = 0
"""Session-scoped monotonic function-call id counter."""

_capture_replay_templates: bool = False
"""Whether the active capture should collect replay-template data.

Phase 4a only plumbs the flag. Phase 4b builds the actual templates.
"""

_relationship_source_model_id: int | None = None
"""Relationship evidence seed: ``id(model)`` at capture start."""

_relationship_source_model_class: str | None = None
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

_log_registry: "weakref.WeakSet[ModelLog]" = weakref.WeakSet()
"""Process-wide weak registry of currently live ``ModelLog`` objects."""

_naming_counters: dict[str, int] = {}
"""Process-global counters used by unnamed ``log_forward_pass`` captures.

The counter is intentionally not thread-safe. Public capture is serialized by
``active_logging()``'s re-entrancy guard, which is the same concurrency boundary
used by the rest of TorchLens logging state.
"""

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


def _register_log(log: "ModelLog") -> None:
    """Register a model log in the process-wide weak registry.

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


def list_logs() -> tuple["ModelLog", ...]:
    """Return a snapshot of currently live ``ModelLog`` objects.

    Returns
    -------
    tuple[ModelLog, ...]
        Immutable snapshot of logs still alive in this process.
    """

    snapshot = list(_log_registry)
    return tuple(log for log in snapshot if log is not None)


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

    global _active_hook_plan, _active_intervention_spec, _func_call_id_counter
    global _pending_live_fire_records
    global _capture_replay_templates
    global _relationship_source_model_id, _relationship_source_model_class
    global _relationship_weight_fingerprint, _relationship_input_id
    global _relationship_input_shape_hash

    _active_hook_plan = None
    _pending_live_fire_records = {}
    _active_intervention_spec = None
    _func_call_id_counter = 0
    _capture_replay_templates = False
    _relationship_source_model_id = None
    _relationship_source_model_class = None
    _relationship_weight_fingerprint = None
    _relationship_input_id = None
    _relationship_input_shape_hash = None


def configure_capture_runtime_context(
    *,
    hook_plan: Any | None = None,
    intervention_spec: "InterventionSpec | None" = None,
    capture_replay_templates: bool = False,
    source_model_id: int | None = None,
    source_model_class: str | None = None,
    weight_fingerprint: str | None = None,
    input_id: int | None = None,
    input_shape_hash: str | None = None,
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
    source_model_id:
        ``id(model)`` captured at the public API boundary.
    source_model_class:
        Model class qualname captured at the public API boundary.
    weight_fingerprint:
        Deterministic model-parameter fingerprint.
    input_id:
        Input object identity captured at the public API boundary.
    input_shape_hash:
        Deterministic input shape/dtype/device fingerprint.

    Returns
    -------
    None
        The module-level runtime context is updated in place.
    """

    global _active_hook_plan, _active_intervention_spec, _capture_replay_templates
    global _relationship_source_model_id, _relationship_source_model_class
    global _relationship_weight_fingerprint, _relationship_input_id
    global _relationship_input_shape_hash

    _active_hook_plan = hook_plan
    _active_intervention_spec = intervention_spec
    _capture_replay_templates = capture_replay_templates
    _relationship_source_model_id = source_model_id
    _relationship_source_model_class = source_model_class
    _relationship_weight_fingerprint = weight_fingerprint
    _relationship_input_id = input_id
    _relationship_input_shape_hash = input_shape_hash


def next_func_call_id() -> int:
    """Allocate the next session-scoped function-call id.

    Returns
    -------
    int
        Monotonic id for one decorated torch function invocation.
    """

    global _func_call_id_counter

    _func_call_id_counter += 1
    return _func_call_id_counter


# ---------------------------------------------------------------------------
# Decoration state — tracks whether torch functions are currently wrapped
# ---------------------------------------------------------------------------

_is_decorated: bool = False
"""True when torch functions are currently wrapped with torchlens interceptors.

Set to True at the end of ``decorate_all_once()`` / ``wrap_torch()``, and
to False at the end of ``unwrap_torch()``.  Checked by ``_ensure_decorated()``
to decide whether (re-)decoration is needed before a logging session.
"""

_decorated_identity: Optional[Callable] = None
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

_func_argnames: Dict[str, tuple] = {}
"""func_name -> tuple of argument names, pre-computed via ``inspect.signature``
for every torch function at decoration time.  Used by the wrapper to build
keyword-argument metadata for logged operations.
"""

_orig_to_decorated: Dict[int, Callable] = {}
"""id(original_func) -> decorated wrapper.  Used by ``patch_detached_references``
to replace bare references (e.g. ``from torch import cos``) in sys.modules with
the decorated version.  Keyed by id() for O(1) lookup.
"""

_decorated_to_orig: Dict[int, Callable] = {}
"""id(decorated_func) -> original_func.  The reverse of ``_orig_to_decorated``.
Keyed by id() for fast lookup when a wrapper needs the unwrapped callable.
"""

# Also keep a version keyed by the decorated func object itself (not id),
# for use in model_funcs where we need ``func in decorated_func_mapper``
# (i.e. the ``in`` operator needs the actual object, not its id).
_decorated_func_mapper: Dict[Callable, Callable] = {}
"""Bidirectional map: decorated -> original AND original -> decorated.

Keyed by actual callable objects (not ids) so that ``func in _decorated_func_mapper``
works.  Used in model_funcs to determine whether a callable is already wrapped.
"""

# ---------------------------------------------------------------------------
# Crawl cache (grows monotonically, never cleared)
# ---------------------------------------------------------------------------
# ``patch_detached_references`` walks sys.modules to find bare references to
# original torch functions (e.g. ``from torch import cos``) and replaces them
# with decorated versions.  These caches avoid re-scanning already-visited
# modules on subsequent calls.

_crawled_module_keys: Set[str] = set()
"""sys.modules keys already scanned by ``patch_detached_references``.

Only new keys (modules imported after the last crawl) are scanned on each call,
making repeated crawls cheap.
"""

_dir_cache: Dict[type, list] = {}
"""Per-type cache of filtered ``dir()`` results for ``extend_search_stack_from_item``.

Avoids repeated introspection of the same type's attributes during the
recursive sys.modules crawl.
"""
_prepared_models: "weakref.WeakSet" = weakref.WeakSet()
"""Models that have already been through ``_prepare_model_once()``.

Using a WeakSet ensures that if the user discards a model, it can be
garbage-collected without this set holding a strong reference.  Membership
here means the model's forward and submodule hooks are already installed.
"""

# ---------------------------------------------------------------------------
# Usage stats — opt-in per-function call counting for coverage analysis
# ---------------------------------------------------------------------------

_collect_usage_stats: bool = False
"""When True, every decorated wrapper increments call counts in
``_function_call_counts`` during logged forward passes.  Used by the
test suite to verify ArgSpec lookup table coverage."""

_functorch_warning_emitted: bool = False
"""True if a warning about skipped functorch/vmap ops has been emitted for
the current logging session.  Reset to False at the start of every
``active_logging()`` context so each forward pass gets at most one warning."""

_function_call_counts: Dict[str, int] = {}
"""func_name -> total calls across all logged forward passes."""

_function_call_models: Dict[str, Set[str]] = {}
"""func_name -> set of model names that called this function."""

_current_model_name: str = ""
"""Set by test fixtures to identify which model is being logged."""

# ---------------------------------------------------------------------------
# Dynamic ArgSpec cache — Tier 3 of the O(1) extraction strategy
# ---------------------------------------------------------------------------

_dynamic_arg_specs: Dict[str, object] = {}
"""Normalized func_name -> ArgSpec, populated by BFS fallback on first
call to an uncovered function.  Subsequent calls reuse the cached spec."""

# ---------------------------------------------------------------------------
# Tagged tensor tracking — for fast cleanup
# ---------------------------------------------------------------------------

_tagged_buffer_ids: Set[int] = set()
"""ids of tensors tagged with tl_buffer_address during prepare_buffer_tensors.
Used by _undecorate_model_tensors for O(n) cleanup instead of re-scanning
all module attributes."""

# ---------------------------------------------------------------------------
# Context managers
# ---------------------------------------------------------------------------


@contextmanager
def active_logging(model_log: "ModelLog") -> Iterator[None]:
    """Activate logging for the duration of a forward pass.

    Sets ``_logging_enabled = True`` and ``_active_model_log = model_log``.
    On exit (including exceptions), resets both.

    Ordering invariant:
        - On entry: set ``_active_model_log`` *before* the toggle, so wrappers
          never see ``_logging_enabled=True`` with a stale/None model_log.
        - On exit: clear the toggle *before* the model_log, for the same reason.

    This context manager is NOT nestable.  Only one forward pass may be logged
    at a time (single-threaded design).  Entering a second ``active_logging``
    while another is already active raises ``RuntimeError`` — silently
    corrupting the outer log (overwriting ``_active_model_log`` and then
    clearing it on inner exit) is worse than failing loudly.
    """
    global _logging_enabled, _active_model_log, _functorch_warning_emitted, _func_call_id_counter
    if _logging_enabled or _active_model_log is not None or _hook_reentrancy_depth > 0:
        raise RuntimeError(
            "torchlens.log_forward_pass / active_logging is not re-entrant: "
            "another forward pass is already being logged. Nested logging "
            "would silently corrupt the outer ModelLog. If you need to log a "
            "model's forward pass from inside another log_forward_pass call "
            "(e.g., a custom activation_postfunc), finish the outer capture "
            "before starting another one."
        )
    # Model log must be visible before the toggle flips — wrappers will
    # immediately read _active_model_log once _logging_enabled is True.
    _active_model_log = model_log
    _functorch_warning_emitted = False
    _func_call_id_counter = 0
    _logging_enabled = True
    try:
        yield
    finally:
        # Toggle off first so no wrapper sees enabled=True with model_log=None
        _logging_enabled = False
        _active_model_log = None


@contextmanager
def pause_logging() -> Iterator[None]:
    """Temporarily disable logging so internal torch ops don't get recorded.

    Nestable via save/restore: if already paused, restoring ``prev`` (False)
    is a harmless no-op.  If logging was active, it resumes on exit.

    This intentionally does NOT clear ``_active_model_log``. The active log
    remains visible while the toggle is paused so ``active_logging()`` can still
    reject nested captures inside paused internal work.

    Typical callers:
        - ``safe_copy``: copies tensors without logging the copy op
        - ``activation_postfunc``: applies user post-processing without logging
        - ``get_tensor_memory_amount``: calls ``nelement()`` / ``element_size()``
          which are themselves decorated tensor methods — without pausing,
          they'd trigger infinite recursive logging.
    """
    global _logging_enabled
    prev = _logging_enabled  # save current state (True or False)
    _logging_enabled = False
    try:
        yield
    finally:
        _logging_enabled = prev  # restore — enables nesting without corruption
