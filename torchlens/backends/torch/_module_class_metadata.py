"""Class-level metadata of a module class, cached per class for one capture."""

from __future__ import annotations

import inspect
import weakref
from typing import Any

from torch import nn

__all__ = (
    "_NN_MODULE_ATTRS",
    "_PYTORCH_INTERNAL",
    "_get_class_metadata",
    "_module_class_metadata_cache",
    "_source_start_line",
)

#: Module class -> its metadata; cleared at the start and end of every capture
#: so generated or function-local module classes are never pinned between runs.
_module_class_metadata_cache: dict[type, dict[str, Any]] = {}

# Process-stable memo for source start lines. ``inspect.findsource`` re-tokenizes
# (functions) or fully AST-parses (classes, CPython < 3.13) the defining file on
# every call, a fixed multi-ms tax repaid each capture because the per-session
# cache above is cleared. The start line is a fact of the definition site, so it
# is keyed on the class object itself or on a function's code object: a
# reloaded/redefined class or a monkey-patched method is a *new* object and can
# never be served a stale entry. Weak keys keep dynamically created classes and
# code collectable; failures are never cached, so exception behavior is unchanged.
_source_line_cache: weakref.WeakKeyDictionary[Any, int] = weakref.WeakKeyDictionary()


def _source_start_line(obj: Any) -> int:
    """Memoized ``inspect.getsourcelines(obj)[1]`` for classes and functions."""
    key = obj if isinstance(obj, type) else getattr(obj, "__code__", None)
    if key is not None:
        try:
            line = _source_line_cache.get(key)
        except TypeError:
            key = None
        else:
            if line is not None:
                return line
    line = inspect.getsourcelines(obj)[1]
    if key is not None:
        _source_line_cache[key] = line
    return line


# Pre-computed set of nn.Module attribute names (from MRO). Used to filter out
# inherited custom_methods/attrs when scanning for user-defined extras. Computed once
# at import time — nn.Module's interface is stable within a process.
_NN_MODULE_ATTRS = set(dir(nn.Module))

# PyTorch internal instance attributes to skip when scanning for user-defined
# extras. Module-level constant to avoid recreating per-module.
_PYTORCH_INTERNAL = frozenset(
    {
        "_parameters",
        "_buffers",
        "_modules",
        "_backward_hooks",
        "_backward_pre_hooks",
        "_forward_hooks",
        "_forward_pre_hooks",
        "_state_dict_hooks",
        "_load_state_dict_pre_hooks",
        "_load_state_dict_post_hooks",
        "_non_persistent_buffers_set",
        "training",
        "T_destination",
        "dump_patches",
        "call_super_init",
    }
)


def _get_class_metadata(module_class: type, save_source_context: bool = False) -> dict[str, Any]:
    """Return class-level metadata for a module class, cached across instances.

    When ``save_code_context`` is False (default), skips expensive
    ``inspect.getsourcelines`` and ``inspect.signature`` calls. Only class
    name and docstrings (already in memory) are captured.

    When True, also fetches source file/line and signatures. Cached per class
    type to avoid redundant filesystem reads.
    """
    cached = _module_class_metadata_cache.get(module_class)
    if cached is not None:
        return cached

    meta: dict[str, Any] = {}
    meta["class_name"] = module_class.__name__
    meta["class_qualname"] = f"{module_class.__module__}.{module_class.__qualname__}"
    meta["cls"] = module_class
    meta["class_docstring"] = module_class.__doc__

    # Cache user-defined custom_methods from class __dict__ (same for all instances of this class).
    user_custom_methods = []
    for attr_name in module_class.__dict__:
        if attr_name.startswith("_") or attr_name.startswith("tl_"):
            continue
        if attr_name in _PYTORCH_INTERNAL or attr_name in _NN_MODULE_ATTRS:
            continue
        val = module_class.__dict__[attr_name]
        if callable(val):
            user_custom_methods.append(attr_name)
    meta["user_custom_methods"] = user_custom_methods

    if save_source_context:
        try:
            meta["class_source_file"] = inspect.getfile(module_class)
        except (TypeError, OSError):
            meta["class_source_file"] = None
        try:
            meta["class_source_line"] = _source_start_line(module_class)
        except (TypeError, OSError):
            meta["class_source_line"] = None

        init_method = getattr(module_class, "__init__", None)
        try:
            if init_method is not None and init_method is not nn.Module.__init__:
                meta["init_source_file"] = inspect.getsourcefile(init_method) or inspect.getfile(
                    init_method
                )
                meta["init_source_line"] = _source_start_line(init_method)
            else:
                meta["init_source_file"] = None
                meta["init_source_line"] = None
        except (TypeError, OSError):
            meta["init_source_file"] = None
            meta["init_source_line"] = None
        try:
            meta["init_signature"] = (
                str(inspect.signature(init_method)) if init_method is not None else None
            )
        except (ValueError, TypeError):
            meta["init_signature"] = None
        meta["init_docstring"] = getattr(init_method, "__doc__", None)

        forward_method = getattr(module_class, "forward", None)
        try:
            if forward_method is not None:
                meta["forward_source_file"] = inspect.getsourcefile(
                    forward_method
                ) or inspect.getfile(forward_method)
                meta["forward_source_line"] = _source_start_line(forward_method)
            else:
                meta["forward_source_file"] = None
                meta["forward_source_line"] = None
        except (TypeError, OSError):
            meta["forward_source_file"] = None
            meta["forward_source_line"] = None
        try:
            meta["forward_signature"] = (
                str(inspect.signature(forward_method)) if forward_method is not None else None
            )
        except (ValueError, TypeError):
            meta["forward_signature"] = None
        meta["forward_docstring"] = getattr(forward_method, "__doc__", None)
    else:
        meta["class_source_file"] = None
        meta["class_source_line"] = None
        meta["init_source_file"] = None
        meta["init_source_line"] = None
        meta["forward_source_file"] = None
        meta["forward_source_line"] = None
        meta["init_signature"] = None
        meta["init_docstring"] = None
        meta["forward_signature"] = None
        meta["forward_docstring"] = None

    _module_class_metadata_cache[module_class] = meta
    return meta
