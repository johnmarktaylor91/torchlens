"""Candidate children of a NumPy frame for the frame-local RNG snapshot."""

from __future__ import annotations

import _thread as _c_thread_module
import collections as _collections_module
import contextvars as _contextvars_module
import functools as _functools_module
import gc as _gc_module
import weakref as _weakref_module
from types import (
    BuiltinFunctionType,
    FunctionType,
    GetSetDescriptorType,
    MappingProxyType,
    MethodType,
    MethodWrapperType,
    ModuleType,
)
from typing import Any

import numpy as np
import torch

from . import inventory_types as _types

try:  # ``resource`` is POSIX-only; feature-detected for the clock family.
    import resource as _resource_module
except ImportError:  # pragma: no cover - non-POSIX platforms
    _resource_module = None  # type: ignore[assignment]

from . import inventory_walk
from .inventory_types import (
    _CUSTOM_HOLDER_SKIP_MODULES,
    _LRU_CACHE_WRAPPER_TYPE,
    _PARTIAL_SLOT_DESCRIPTORS,
    _STDLIB_CLASS_LEAF_MODULES,
    _WEAKREF_PROXY_TYPES,
)


def _numpy_frame_candidate_children(value: Any) -> tuple[Any, ...]:
    """Return one inert holder edge below a profiled-frame value.

    Exact built-in container types are read through their concrete C
    implementations, so a hostile subclass cannot execute an overridden iterator.
    Plain-object attributes are read only through a concrete getset descriptor for
    the instance ``__dict__``; a property or other hostile descriptor named
    ``__dict__`` makes the value a leaf. Callable receivers remain gated by
    :func:`_numpy_rng_receiver`, which reads ``__self__`` only after an exact
    built-in-bound-callable or ``MethodType`` check.

    Parameters
    ----------
    value:
        Candidate frame local or referenced global.

    Returns
    -------
    tuple[Any, ...]
        Immediate inert children that may themselves be concrete NumPy RNG
        receivers.
    """

    value_type = type(value)
    if value_type is dict:
        return (*dict.keys(value), *dict.values(value))
    if value_type in (list, tuple, set, frozenset):
        return tuple(value)
    # r39: match weakref PROXIES on EXACT type BEFORE any isinstance branch below
    # -- ``isinstance`` consults the proxy's forwarded ``__class__``, so a proxy
    # whose referent is a deque/local/partial would otherwise be dispatched into a
    # base-C read that raises (or, worse, forwards). NO inert deref exists (see
    # :data:`_WEAKREF_PROXY_TYPES`); leaf here -- the deep frame walk fail-closes
    # this shape to INCOMPLETE.
    if type(value) in _WEAKREF_PROXY_TYPES:
        return ()
    # r38 stdlib-instance holder edges, mirroring the deep-inventory parity
    # branches: every read below is base-C (deref / member descriptor /
    # ``tp_traverse`` / base ``__iter__``), so no user code can fire.
    if isinstance(value, _weakref_module.ref):
        try:
            referent = _weakref_module.ref.__call__(value)
        except Exception:
            return ()
        return () if referent is None else (referent,)
    if isinstance(value, _c_thread_module._local):
        children: list[Any] = []
        for referent in _gc_module.get_referents(value):
            if type(referent) is not dict:
                continue
            for per_thread in dict.values(referent):
                if type(per_thread) is dict:
                    children.extend(dict.values(per_thread))
        return tuple(children)
    if isinstance(value, _functools_module.partial):
        interior: list[Any] = []
        for descriptor in _PARTIAL_SLOT_DESCRIPTORS:
            try:
                interior.append(descriptor.__get__(value, value_type))
            except Exception:
                continue
        return tuple(interior)
    if isinstance(value, _collections_module.deque):
        return tuple(_collections_module.deque.__iter__(value))
    # r39 C-holder parity branches (round-39 executed false-VERIFIEDs V9a-V9f):
    # a generator reached ONLY through a C-implemented holder with a C accessor
    # path never enters a Python frame. Every read below is base-C
    # (``tp_traverse`` / base getset / C ``Context`` lookup); no user code fires.
    if _LRU_CACHE_WRAPPER_TYPE is not None and isinstance(value, _LRU_CACHE_WRAPPER_TYPE):
        # A WARM wrapper's ``tp_traverse`` exposes its cache dict; flatten one
        # dict level so this single-edge belt sees the cached values directly.
        cached_values: list[Any] = []
        for referent in _gc_module.get_referents(value):
            if type(referent) is dict:
                cached_values.extend(dict.values(referent))
        return tuple(cached_values)
    if type(value) is MappingProxyType:
        # ``tp_traverse`` yields the BACKING mapping without invoking its (possibly
        # user-defined) ``keys``/``values``; flatten an exact-dict backing one level.
        proxied: list[Any] = []
        for referent in _gc_module.get_referents(value):
            if type(referent) is dict:
                proxied.extend(dict.keys(referent))
                proxied.extend(dict.values(referent))
            else:
                proxied.append(referent)
        return tuple(proxied)
    if type(value) is _contextvars_module.ContextVar:
        # The VALUE lives in the per-thread ``Context``, off the reference graph;
        # the base C ``get`` (set value or declared default, owner thread) is the
        # only inert edge. Pre-existing OTHER threads' contexts stay the documented
        # foreign-thread residual.
        try:
            return (_contextvars_module.ContextVar.get(value),)
        except LookupError:
            return ()
    if isinstance(value, type):
        # Class-attribute surface across the OWN MRO and the METACLASS MRO --
        # ``CLS.gen`` on a base class or user metaclass resolves entirely in C.
        return inventory_walk._class_attr_surface(value)
    if isinstance(value, np.ndarray):
        # An OBJECT-dtype ndarray holds ordinary Python references (numpy's own
        # parallel-streams idiom: arrays of spawned Generators) and ``ARR[0]`` is
        # a profile-silent C subscript. Element-iterate ONLY object dtypes through
        # the base getsets; non-object dtypes stay hard leaves. This belt has no
        # budget of its own, so an over-cap object array is left to the budgeted
        # (fail-closed) deep frame walk.
        try:
            dtype = np.ndarray.dtype.__get__(value)
            if dtype.hasobject and int(np.ndarray.size.__get__(value)) <= (
                _types._DEEP_INVENTORY_NODE_CAP
            ):
                return tuple(np.ndarray.flat.__get__(value))
        except Exception:
            return ()
        return ()
    if isinstance(
        value,
        (
            ModuleType,
            FunctionType,
            BuiltinFunctionType,
            MethodType,
            MethodWrapperType,
            np.generic,
            torch.Tensor,
            torch.nn.Module,
        ),
    ):
        return ()
    try:
        module = type.__dict__["__module__"].__get__(value_type)
    except Exception:
        return ()
    if not isinstance(module, str):
        return ()
    module_root = module.split(".", 1)[0]
    if module_root in _CUSTOM_HOLDER_SKIP_MODULES or module_root in _STDLIB_CLASS_LEAF_MODULES:
        return ()
    instance_dict_getter = None
    try:
        mro = type.__dict__["__mro__"].__get__(value_type)
    except Exception:
        return ()
    for klass in mro:
        try:
            class_dict = type.__dict__["__dict__"].__get__(klass)
        except Exception:
            return ()
        descriptor = class_dict.get("__dict__")
        if descriptor is None:
            continue
        if not isinstance(descriptor, GetSetDescriptorType):
            return ()
        instance_dict_getter = descriptor
        break
    if instance_dict_getter is None:
        return ()
    try:
        instance_dict = instance_dict_getter.__get__(value, value_type)
    except (AttributeError, TypeError, ValueError):
        return ()
    if not isinstance(instance_dict, dict):
        return ()
    return tuple(instance_dict.values())
