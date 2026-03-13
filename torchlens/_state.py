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
from typing import TYPE_CHECKING, Callable, Dict, Optional, Set

# TYPE_CHECKING is False at runtime, so this import only exists for static
# analysis / IDE support — it will never trigger the circular-import problem.
if TYPE_CHECKING:
    from .data_classes.model_log import ModelLog

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
def active_logging(model_log: "ModelLog"):
    """Activate logging for the duration of a forward pass.

    Sets ``_logging_enabled = True`` and ``_active_model_log = model_log``.
    On exit (including exceptions), resets both.

    Ordering invariant:
        - On entry: set ``_active_model_log`` *before* the toggle, so wrappers
          never see ``_logging_enabled=True`` with a stale/None model_log.
        - On exit: clear the toggle *before* the model_log, for the same reason.

    This context manager is NOT nestable.  Only one forward pass may be logged
    at a time (single-threaded design).
    """
    global _logging_enabled, _active_model_log
    # Model log must be visible before the toggle flips — wrappers will
    # immediately read _active_model_log once _logging_enabled is True.
    _active_model_log = model_log
    _logging_enabled = True
    try:
        yield
    finally:
        # Toggle off first so no wrapper sees enabled=True with model_log=None
        _logging_enabled = False
        _active_model_log = None


@contextmanager
def pause_logging():
    """Temporarily disable logging so internal torch ops don't get recorded.

    Nestable via save/restore: if already paused, restoring ``prev`` (False)
    is a harmless no-op.  If logging was active, it resumes on exit.

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
