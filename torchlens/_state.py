"""Global state for torchlens toggle-gated decoration.

WARNING: This module must NOT import from any other torchlens module at
module level.  Use TYPE_CHECKING guards for type hints only.  This prevents
circular imports since every other module may import from here.
"""

import weakref
from contextlib import contextmanager
from typing import TYPE_CHECKING, Callable, Dict, Optional, Set

if TYPE_CHECKING:
    from .data_classes.model_log import ModelLog

# ---------------------------------------------------------------------------
# Toggle
# ---------------------------------------------------------------------------

_logging_enabled: bool = False
"""When False, all decorated wrappers pass through immediately."""

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

_active_model_log: Optional["ModelLog"] = None
"""Points to the current logging session's ModelLog.  None when not logging."""

# ---------------------------------------------------------------------------
# Pre-computed at import time (populated once by decorate_all_once, immutable after)
# ---------------------------------------------------------------------------

_func_argnames: Dict[str, tuple] = {}
"""func_name -> tuple of argument names, pre-computed for all torch functions."""

_orig_to_decorated: Dict[int, Callable] = {}
"""id(original_func) -> decorated_func, for sys.modules crawl."""

_decorated_to_orig: Dict[int, Callable] = {}
"""id(decorated_func) -> original_func, permanent decorated_func_mapper."""

# Also keep a version keyed by the decorated func object itself (not id),
# for use in model_funcs where we need ``func in decorated_func_mapper``.
_decorated_func_mapper: Dict[Callable, Callable] = {}
"""Maps decorated -> original AND original -> decorated (both directions)."""

# ---------------------------------------------------------------------------
# Crawl cache (grows monotonically)
# ---------------------------------------------------------------------------

_crawled_module_keys: Set[str] = set()
"""sys.modules keys already scanned; only scan new ones on each call."""

_dir_cache: Dict[type, list] = {}
"""Per-type cache of filtered dir() results for extend_search_stack_from_item."""

_prepared_models: "weakref.WeakSet" = weakref.WeakSet()
"""Models whose forwards are already wrapped; WeakSet prevents GC issues."""

# ---------------------------------------------------------------------------
# Context managers
# ---------------------------------------------------------------------------


@contextmanager
def active_logging(model_log: "ModelLog"):
    """Activate logging for the duration of a forward pass.

    Sets ``_logging_enabled = True`` and ``_active_model_log = model_log``.
    On exit (including exceptions), resets both atomically — toggle first.
    """
    global _logging_enabled, _active_model_log
    _active_model_log = model_log
    _logging_enabled = True
    try:
        yield
    finally:
        # Toggle off first — single STORE_ATTR bytecode, atomic w.r.t. signals
        _logging_enabled = False
        _active_model_log = None


@contextmanager
def pause_logging():
    """Temporarily disable logging.  Nestable (save/restore pattern).

    Used inside decorated wrappers when we need to call torch ops without
    triggering logging (e.g. safe_copy, activation_postfunc).
    """
    global _logging_enabled
    prev = _logging_enabled
    _logging_enabled = False
    try:
        yield
    finally:
        _logging_enabled = prev
