"""Root-module preparation: the root's identity and the bound-method owner.

The wrapper (:class:`~torchlens.backends.torch.bound_root.TLBoundMethodRoot`)
presents the owner's tree, so the once-per-model traversal gives every
descendant its plain address; the owner itself stands at the root position of
the walk and is recorded like a submodule, at the root address, so its calls
inside the bound method read ``self:1`` .. ``self:N``.
"""

from __future__ import annotations

import inspect
import weakref
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch.nn as nn

from . import _state
from ._tl import set_module_meta
from .bound_root import ROOT_ADDRESS, TLBoundMethodRoot

if TYPE_CHECKING:
    from ..._data_classes.trace import Trace


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
def prepare_bound_owner_once(
    model: nn.Module, restore_undecorated_forward: Callable[[nn.Module], None]
) -> None:
    """Address the owner of a bound-method root at the root address ``self``."""

    if not isinstance(model, TLBoundMethodRoot):
        return
    owner = model.tl_owner
    _state.record_module_root_prep(model, owner)
    restore_undecorated_forward(owner)
    set_module_meta(owner, address=ROOT_ADDRESS, module_type=str(type(owner).__name__))


def register_session_module_calls(trace: Trace, module: nn.Module) -> None:
    """Open the session-scoped call counters for one instrumented module."""

    mod_id = id(module)
    trace._module_capture_ws.mod_call_index[mod_id] = 0
    trace._module_capture_ws.mod_call_labels[mod_id] = []
    trace._module_capture_ws.mod_entered[mod_id] = []
    trace._module_capture_ws.mod_exited[mod_id] = []


def register_bound_owner_calls(trace: Trace, model: nn.Module) -> None:
    """Count the owner's own calls like a submodule's on a bound-method root."""

    if isinstance(model, TLBoundMethodRoot):
        register_session_module_calls(trace, model.tl_owner)


def record_model_identity(trace: Trace, model: nn.Module) -> None:
    """Record the root's class, signatures, docstrings and source locations on ``trace``."""

    # F41 bound-method roots: the synthetic TL-authored root's identity reads
    # the OWNER (type(owner).__name__, the owner's class/source metadata) and
    # the bound METHOD as the entry callable -- never the wrapper class name
    # or the string "method".
    if isinstance(model, TLBoundMethodRoot):
        identity_cls = type(model.tl_owner)
        entry_callable = getattr(identity_cls, model.tl_method_name, None)
    else:
        identity_cls = type(model)
        entry_callable = getattr(identity_cls, "forward", None)
    trace._source_class_name = str(identity_cls.__name__)
    trace._source_class_docstring = identity_cls.__doc__
    init_method = getattr(identity_cls, "__init__", None)
    forward_method = entry_callable
    try:
        trace._source_init_signature = str(inspect.signature(init_method)) if init_method else None
    except (TypeError, ValueError):
        trace._source_init_signature = None
    trace._source_init_docstring = getattr(init_method, "__doc__", None)
    try:
        trace._source_forward_signature = (
            str(inspect.signature(forward_method)) if forward_method else None
        )
    except (TypeError, ValueError):
        trace._source_forward_signature = None
    trace._source_forward_docstring = getattr(forward_method, "__doc__", None)
    try:
        trace._source_class_file = inspect.getfile(identity_cls)
        trace._source_class_line = _source_start_line(identity_cls)
        trace._source_init_file = inspect.getfile(identity_cls.__init__)
        trace._source_init_line = _source_start_line(identity_cls.__init__)
    except (OSError, TypeError):
        trace._source_class_file = None
        trace._source_class_line = None
        trace._source_init_file = None
        trace._source_init_line = None
    try:
        forward_func = model.forward if entry_callable is None else entry_callable
        trace._source_forward_file = inspect.getsourcefile(forward_func) or inspect.getfile(
            forward_func
        )
        trace._source_forward_line = _source_start_line(forward_func)
    except (OSError, TypeError):
        trace._source_forward_file = None
        trace._source_forward_line = None
