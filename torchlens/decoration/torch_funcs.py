"""Permanent torch function wrapping: decorates all torch ops at import time with toggle-gated wrappers.

This module implements the core interception mechanism for TorchLens. At ``import torchlens``
time, every function listed in ``ORIG_TORCH_FUNCS`` is replaced with a thin wrapper that
checks a single boolean (``_state._logging_enabled``) on each call:

  - **Logging off** (default): one branch check, near-zero overhead, original function called.
  - **Logging on**: all tensor outputs are captured into the active ``ModelLog``.

Key design decisions:

1. **Permanent decoration** avoids the fragility of repeatedly patching/unpatching torch
   internals. The toggle makes this safe for production use.

2. **Shared originals reuse wrappers**: If ``torch.cos`` and ``torch._VF.cos`` point to the
   same C builtin, only one wrapper is created and both namespaces point to it. This keeps
   ``_orig_to_decorated`` / JIT builtin table mappings 1:1.

3. **sys.modules crawl** (``patch_detached_references``): Code like ``from torch import cos``
   captures a reference to the *original* function before decoration. The crawl replaces
   these stale references in module dicts, class dicts, and function defaults.

4. **JIT compatibility**: Decorated wrappers are registered in ``torch.jit._builtins._builtin_table``
   so ``torch.jit.script`` recognizes them as known ATen ops.

5. **DeviceContext bypass**: Python wrappers bypass PyTorch's C-level ``TorchFunctionMode``
   dispatch, so ``torch.device('meta')`` context managers won't inject ``device`` kwargs
   automatically. We detect active ``DeviceContext`` and inject the kwarg ourselves.
"""

import inspect
import sys
import time
import types
import warnings
from functools import wraps
from typing import Callable, Dict, List, Optional, TYPE_CHECKING, Tuple

import torch

from .. import _state
from ..constants import ORIG_TORCH_FUNCS
from ..data_classes.internal_types import FuncExecutionContext
from ..utils.introspection import get_vars_of_type_from_obj, nested_getattr
from ..utils.display import identity
from ..utils.rng import log_current_autocast_state, log_current_rng_states
from ..utils.hashing import make_random_barcode
from ..utils.tensor_utils import print_override, safe_copy
from ..capture.output_tensors import log_function_output_tensors
from ..capture.source_tensors import log_source_tensor

if TYPE_CHECKING:
    from ..data_classes.model_log import ModelLog

# Functions that should never be logged — they are metadata queries, not
# computational operations, and logging them would cause infinite recursion
# (e.g. size() is called internally by logging code).
funcs_not_to_log = ["numpy", "__array__", "size", "dim"]

# Print functions get special handling: intercepted to add TorchLens label
# info to the repr without creating a new logged operation.
print_funcs = ["__repr__", "__str__", "_str"]

# Names of torch factory functions that accept a ``device`` kwarg.
# When a ``torch.device`` context manager is active (TorchFunctionMode),
# normal C dispatch injects the device automatically — but our Python
# wrappers bypass that dispatch, so we must inject it ourselves.
_DEVICE_CONSTRUCTOR_NAMES: set = set()

# Lazy imports cached at first use.
_torch_function_mode_len = None
_DeviceContext = None


def _get_active_device() -> Optional[str]:
    """Return the device from the innermost active ``DeviceContext``, or ``None``.

    Walks the ``TorchFunctionMode`` stack in reverse (innermost first) to find
    the most recently pushed ``DeviceContext``. This is the same device that
    PyTorch's C dispatch would inject — but since our Python wrappers bypass
    C dispatch, we must query it manually.
    """
    global _DeviceContext
    if _DeviceContext is None:
        from torch.utils._device import DeviceContext

        _DeviceContext = DeviceContext
    try:
        from torch.overrides import _get_current_function_mode_stack

        for mode in reversed(_get_current_function_mode_stack()):
            if isinstance(mode, _DeviceContext):
                return mode.device  # type: ignore[return-value]
    except (ImportError, AttributeError):
        pass
    return None


def _maybe_inject_device_kwarg(func_name: str, kwargs: dict) -> dict:
    """Inject ``device`` kwarg for factory functions when a ``DeviceContext`` is active.

    Python wrappers bypass PyTorch's C-level ``TorchFunctionMode`` dispatch, so
    ``torch.device('meta')`` context (used by e.g. HuggingFace ``from_pretrained``)
    won't inject the device kwarg automatically. We replicate that injection here.

    Only applies to known factory functions (``torch.zeros``, ``torch.ones``, etc.)
    whose names were collected into ``_DEVICE_CONSTRUCTOR_NAMES`` at decoration time.
    """
    # Early exit: not a factory function, or caller already specified device
    if not (_DEVICE_CONSTRUCTOR_NAMES and func_name in _DEVICE_CONSTRUCTOR_NAMES):
        return kwargs
    if "device" in kwargs:
        return kwargs
    try:
        from torch.overrides import _len_torch_function_stack

        if _len_torch_function_stack() > 0:
            device = _get_active_device()
            if device is not None:
                return {**kwargs, "device": device}
    except (ImportError, AttributeError):
        pass
    return kwargs


def torch_func_decorator(func: Callable, func_name: str):
    """Wrap a single torch function with toggle-gated logging.

    When ``_state._logging_enabled`` is ``False``, the wrapper is a near-noop
    (one bool check, then call original).  When ``True``, it:

    1. Registers any buffer tensors seen for the first time.
    2. Snapshots args (if ``save_function_args``), timing, RNG, and autocast state.
    3. Calls the original function.
    4. Detects **nested calls** via a barcode mechanism (see below).
    5. Handles **in-place ops** by copying the output and propagating the label back.
    6. Logs all output tensors into the active ``ModelLog``.

    **Barcode nesting detection**: Before calling the original function, a random
    barcode is written to ``model_log.current_function_call_barcode``.  If the
    original function internally calls *other* wrapped torch functions, those
    inner calls will overwrite the barcode.  After the call returns, if the
    barcode still matches, this is a "bottom-level" function (leaf in the call
    tree).  Bottom-level functions get richer metadata capture.

    **In-place op handling**: When a function returns the same object as its
    first argument (``id(out) == id(args[0])``), the output is ``safe_copy``-ed
    to create a distinct tensor for logging.  For true in-place ops (trailing
    ``_`` or ``__i*`` dunder), the new label is propagated back to the original
    tensor so subsequent operations see it.  Non-mutating self-returns (e.g.
    ``contiguous()`` on an already-contiguous tensor) are copied but NOT
    propagated back — they are silently dropped by barcode identity detection
    downstream.

    Args:
        func: The original (unwrapped) torch function.
        func_name: The attribute name of the function (e.g. ``"cos"``, ``"__add__"``).

    Returns:
        The wrapped function.
    """

    @wraps(func)
    def wrapped_func(*args, **kwargs):
        # ---- Fast path ----
        # When logging is off, pass through with minimal overhead.
        # DeviceContext injection is still needed even when not logging,
        # because the user's model may rely on torch.device('meta') context.
        if not _state._logging_enabled or _state._active_model_log is None:
            kwargs = _maybe_inject_device_kwarg(func_name, kwargs)
            return func(*args, **kwargs)

        model_log = _state._active_model_log

        # Usage stats: count every decorated function call during logging.
        if _state._collect_usage_stats:
            _state._function_call_counts[func_name] = (
                _state._function_call_counts.get(func_name, 0) + 1
            )
            _state._function_call_models.setdefault(func_name, set()).add(
                _state._current_model_name
            )

        # Reset barcode; skip metadata-only functions that would cause recursion.
        model_log.current_function_call_barcode = 0
        if func_name in funcs_not_to_log:
            return func(*args, **kwargs)

        all_args = args if not kwargs else (*args, *kwargs.values())
        arg_tensorlike = get_vars_of_type_from_obj(all_args, torch.Tensor)

        # Register buffer tensors on first encounter. Buffers are tagged with
        # tl_buffer_address during model prep but don't get a tl_tensor_label_raw
        # until the first function actually uses them.
        for t in arg_tensorlike:
            if hasattr(t, "tl_buffer_address") and not hasattr(t, "tl_tensor_label_raw"):
                log_source_tensor(model_log, t, "buffer", getattr(t, "tl_buffer_address"))

        # Intercept print functions to show TorchLens label info in repr.
        if (func_name in print_funcs) and (len(arg_tensorlike) > 0):
            out = print_override(args[0], func_name)
            return out

        # Snapshot args before the call in case in-place ops mutate them.
        if model_log.save_function_args:
            arg_copies = tuple([safe_copy(arg) for arg in args])
            kwarg_copies = {k: safe_copy(v) for k, v in kwargs.items()}
        else:
            arg_copies = args
            kwarg_copies = kwargs

        # ---- Execute the original function ----
        # Write a unique barcode BEFORE the call. If any inner wrapped functions
        # execute during this call, they will overwrite it. After the call,
        # matching barcode => this is the bottom-level (leaf) function.
        func_call_barcode = make_random_barcode()
        model_log.current_function_call_barcode = func_call_barcode
        start_time = time.time()
        rng_states = log_current_rng_states()
        autocast_state = log_current_autocast_state()
        out_orig = func(*args, **kwargs)
        exec_ctx = FuncExecutionContext(
            time_elapsed=time.time() - start_time,
            rng_states=rng_states,
            autocast_state=autocast_state,
        )
        is_bottom_level_func = model_log.current_function_call_barcode == func_call_barcode

        # __setitem__, zero_, __delitem__ modify in-place and return None;
        # treat the first arg (the modified tensor) as the output.
        if func_name in ["__setitem__", "zero_", "__delitem__"]:
            out_orig = args[0]

        # ---- In-place detection and safe copy ----
        same_object_returned = len(args) > 0 and id(out_orig) == id(args[0])
        # True in-place ops (add_, mul_, etc.) modify the tensor and return self.
        # No-op functions (to(same_dtype), contiguous() on contiguous tensor)
        # also return self but don't modify anything.
        # Both cases need safe_copy so logging doesn't overwrite the original's
        # label, but only true in-place ops should propagate the new label back.
        was_inplace = same_object_returned and (
            func_name.endswith("_") or func_name.startswith("__i")
        )
        if same_object_returned:
            # Create a distinct tensor object for logging — otherwise attaching
            # tl_tensor_label_raw to the output would clobber the input's label.
            out_orig = safe_copy(out_orig)

        # Log all output tensors (excluding Parameters, which are source tensors).
        output_tensors = get_vars_of_type_from_obj(
            out_orig,
            which_type=torch.Tensor,
            subclass_exceptions=[torch.nn.Parameter],
        )

        if len(output_tensors) > 0:
            log_function_output_tensors(
                model_log,
                func,
                func_name,
                args,
                kwargs,
                arg_copies,
                kwarg_copies,
                out_orig,
                exec_ctx,
                is_bottom_level_func,
            )

            # For true in-place ops, propagate the newly assigned label back
            # to the original tensor so subsequent operations see it.
            if was_inplace:
                if hasattr(out_orig, "tl_tensor_label_raw"):
                    args[0].tl_tensor_label_raw = out_orig.tl_tensor_label_raw

        return out_orig

    # ---- __wrapped__ removal for JIT compatibility ----
    # @wraps sets __wrapped__ on the wrapper. For C builtins (no __code__),
    # inspect.unwrap() follows __wrapped__ and fails because builtins have
    # no inspectable source. torch.jit.script (e.g. via timm) calls
    # inspect.unwrap internally, so we must remove __wrapped__ to prevent
    # the failure chain: jit.script -> inspect.unwrap -> inspect.getsource -> crash.
    if not hasattr(func, "__code__"):
        try:
            del wrapped_func.__wrapped__
        except AttributeError:
            pass

    return wrapped_func


# ---------------------------------------------------------------------------
# get_func_argnames — now writes to _state._func_argnames instead of self
# ---------------------------------------------------------------------------


def get_func_argnames(orig_func: Callable, func_name: str):
    """Extract argument names for a function and store in ``_state._func_argnames``.

    Tries ``inspect.signature`` first (works for Python functions). Falls back
    to docstring parsing for C builtins whose signature isn't introspectable.

    Stores under the underscore-stripped name (e.g. ``"add"`` for both ``add``
    and ``add_``) so callers can look up via ``func_name.strip("_")`` consistently (#82).

    Skipped for property-like attributes (``real``, ``imag``, ``T``, etc.) that
    aren't callable in the normal sense.
    """
    if func_name in ["real", "imag", "T", "mT", "data", "H"]:
        return

    storage_key = func_name.strip("_")

    try:
        params = inspect.signature(orig_func).parameters
        argnames = []
        for name, param in params.items():
            if name in ("cls", "self"):
                continue
            # #123: Use Parameter.kind instead of naive asterisk stripping
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                argnames.append(f"*{name}")
            elif param.kind == inspect.Parameter.VAR_KEYWORD:
                argnames.append(f"**{name}")
            else:
                argnames.append(name)
        _state._func_argnames[storage_key] = tuple(argnames)
        return
    except ValueError:
        pass

    # Fallback: parse argument names from the docstring's first line.
    # C builtins typically have docstrings like "add(input, other, *, alpha=1)".
    docstring = orig_func.__doc__
    if (type(docstring) is not str) or (len(docstring) == 0):
        return

    paren_start, paren_end = docstring.find("("), docstring.find(")")
    argstring = docstring[paren_start + 1 : paren_end]
    arg_list = argstring.split(",")
    arg_list = [arg.strip(" ") for arg in arg_list]
    argnames = []
    for arg in arg_list:
        argname = arg.split("=")[0]
        if argname in ["*", "/", "//", ""]:
            continue
        argname = argname.replace("*", "")
        argnames.append(argname)
    argnames = tuple([arg for arg in argnames if arg not in ["self", "cls"]])  # type: ignore[assignment]
    _state._func_argnames[storage_key] = argnames  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# One-time decoration at import time
# ---------------------------------------------------------------------------


def decorate_all_once():
    """Decorate all torch functions permanently at import time.

    Iterates over every ``(namespace, func_name)`` pair in ``ORIG_TORCH_FUNCS``
    and replaces each function with a ``torch_func_decorator`` wrapper. Also:

    - Pre-computes ``_state._func_argnames`` for metadata capture.
    - Populates ``_state._orig_to_decorated`` / ``_state._decorated_to_orig``
      bidirectional mappings (keyed by ``id()``).
    - Registers wrappers in ``torch.jit._builtins._builtin_table`` so JIT
      compilation recognizes wrapped functions as known ATen ops.
    - Collects ``_DEVICE_CONSTRUCTOR_NAMES`` for DeviceContext bypass.
    - Installs ``torch.identity`` (a no-op that creates a new tensor entry
      for module boundary tracking).

    **Shared-original deduplication**: Multiple torch namespaces can alias the
    same C builtin (e.g. ``torch.cos`` and ``torch._VF.cos``). When the same
    ``id(orig_func)`` is encountered again, we reuse the existing wrapper
    rather than creating a second one. This ensures the JIT builtin table and
    ``_orig_to_decorated`` stay consistent (one original -> one wrapper).

    Idempotent: returns immediately if already decorated.
    """
    if _state._orig_to_decorated:
        return  # already decorated

    # Pre-compute type objects for efficient isinstance-like checks below.
    function_class = type(lambda: 0)  # <class 'function'>
    builtin_class = type(torch.mean)  # <class 'builtin_function_or_method'>
    method_class = type(torch.Tensor.__add__)  # <class 'method_descriptor'>
    wrapper_class = type(torch.Tensor.__getitem__)  # <class 'method-wrapper'>
    getset_class = type(torch.Tensor.real)  # <class 'getset_descriptor'> (properties)

    for namespace_name, func_name in ORIG_TORCH_FUNCS:
        namespace_key = namespace_name.replace("torch.", "")
        local_func_namespace = nested_getattr(torch, namespace_key)
        if not hasattr(local_func_namespace, func_name):
            continue
        orig_func = getattr(local_func_namespace, func_name)

        # Guard against double-decoration (ORIG_TORCH_FUNCS may list duplicates).
        if getattr(orig_func, "tl_is_decorated_function", False):
            continue

        # Pre-compute argnames for metadata capture during logging.
        if func_name.strip("_") not in _state._func_argnames:
            get_func_argnames(orig_func, func_name)

        if type(orig_func) in [function_class, builtin_class, method_class, wrapper_class]:
            # --- Shared-original deduplication ---
            # If this exact C builtin was already wrapped under a different namespace
            # (e.g. torch.cos and torch._VF.cos share the same id()), reuse the
            # existing wrapper. Creating a second wrapper would break the 1:1 mapping
            # in _orig_to_decorated and the JIT builtin table.
            if id(orig_func) in _state._orig_to_decorated:
                existing = _state._orig_to_decorated[id(orig_func)]
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        setattr(local_func_namespace, func_name, existing)
                except (AttributeError, TypeError):
                    pass
                continue

            new_func = torch_func_decorator(orig_func, func_name)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    setattr(local_func_namespace, func_name, new_func)
                new_func.tl_is_decorated_function = True
                # Bidirectional id-keyed mappings for fast lookup.
                _state._orig_to_decorated[id(orig_func)] = new_func
                _state._decorated_to_orig[id(new_func)] = orig_func
                # Object-keyed mappings for cases where we have the object, not its id.
                _state._decorated_func_mapper[new_func] = orig_func
                _state._decorated_func_mapper[orig_func] = new_func
            except (AttributeError, TypeError):
                pass

        elif type(orig_func) is getset_class:
            # getset_descriptors (e.g. Tensor.real, Tensor.imag) are C-level
            # properties. We wrap getter/setter/deleter individually and
            # reassemble as a Python property.
            getter_orig, setter_orig, deleter_orig = (
                orig_func.__get__,
                orig_func.__set__,
                orig_func.__delete__,
            )
            getter_dec = torch_func_decorator(getter_orig, func_name)
            setter_dec = torch_func_decorator(setter_orig, func_name)
            deleter_dec = torch_func_decorator(deleter_orig, func_name)
            getter_dec.tl_is_decorated_function = True
            setter_dec.tl_is_decorated_function = True
            deleter_dec.tl_is_decorated_function = True
            new_property = property(getter_dec, setter_dec, deleter_dec, doc=func_name)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    setattr(local_func_namespace, func_name, new_property)
                # #31: Only add mapper entries if setattr succeeded — otherwise
                # we'd have dangling entries pointing to an uninstalled property.
                _state._orig_to_decorated[id(orig_func)] = new_property
                _state._decorated_to_orig[id(new_property)] = orig_func
                _state._decorated_func_mapper[new_property] = orig_func
                _state._decorated_func_mapper[orig_func] = new_property
            except (AttributeError, TypeError):
                pass

    # ---- JIT builtin table registration ----
    # torch.jit._builtins._builtin_table maps id(func) -> ATen op name.
    # We must register our wrappers so JIT recognizes them as the same ops.
    # Without this, torch.jit.script fails on any code using wrapped functions.
    try:
        import torch.jit._builtins as _jit_builtins

        for orig_id, decorated_func in _state._orig_to_decorated.items():
            builtin_name = _jit_builtins._builtin_table.get(orig_id)
            if builtin_name is not None:
                _jit_builtins._builtin_table[id(decorated_func)] = builtin_name
                # For properties, also register getter/setter/deleter individually
                # since JIT may call them directly.
                if isinstance(decorated_func, property):
                    for accessor in (decorated_func.fget, decorated_func.fset, decorated_func.fdel):
                        if accessor is not None:
                            _jit_builtins._builtin_table[id(accessor)] = builtin_name
    except (ImportError, AttributeError):
        pass  # JIT internals may change across PyTorch versions

    # ---- DeviceContext bypass setup ----
    # Collect names of factory functions (zeros, ones, empty, etc.) that accept
    # a device kwarg. The lru_cache must be cleared first so _device_constructors()
    # re-evaluates with our wrapped functions (otherwise it returns stale refs).
    try:
        from torch.utils._device import _device_constructors

        _device_constructors.cache_clear()
        for ctor in _device_constructors():
            name = getattr(ctor, "__name__", None)
            if name:
                _DEVICE_CONSTRUCTOR_NAMES.add(name)
    except (ImportError, AttributeError):
        pass

    # Install torch.identity — a decorated no-op that creates a new tensor entry.
    # Used at module boundaries: when nn.Identity is encountered or when an output
    # tensor is the same object as an input tensor, torch.identity() forces a new
    # log entry so the graph correctly shows the module boundary.
    new_identity = torch_func_decorator(identity, "identity")
    torch.identity = new_identity


# ---------------------------------------------------------------------------
# sys.modules deep crawl
# ---------------------------------------------------------------------------


def patch_detached_references():
    """Crawl ``sys.modules`` and replace stale references to original torch
    functions with their decorated counterparts.

    **Why this is needed**: Code like ``from torch import cos`` captures a
    reference to the *original* ``torch.cos`` before decoration. After
    ``decorate_all_once()`` replaces ``torch.cos``, the importing module
    still holds the old reference. This crawl fixes those stale references.

    **Four crawl levels**:

    1. **Module-level attributes** — ``import torch; my_cos = torch.cos`` style.
       Checks each attribute in the module's ``__dict__`` against
       ``_orig_to_decorated`` by ``id()``.

    2. **Class-level attributes** — Classes defined in other modules that store
       torch function references as class attributes or methods. Crawls
       ``vars(cls)`` for each class found in the module.

    3. **Function defaults** — Functions that use torch functions as default
       argument values (e.g. ``def f(act=torch.relu)``). Patches both
       ``__defaults__`` and ``__kwdefaults__``.

    4. **Model instance attributes** — Handled separately by
       ``patch_model_instance()`` at ``log_forward_pass`` time, since model
       instances may not exist yet when this function runs.

    Incremental: uses ``_state._crawled_module_keys`` to skip already-scanned
    modules. Called on every ``log_forward_pass`` to catch lazily-imported modules.
    """
    new_keys = set(sys.modules.keys()) - _state._crawled_module_keys
    if not new_keys:
        return

    mapping = _state._orig_to_decorated
    if not mapping:
        return

    for mod_key in new_keys:
        _state._crawled_module_keys.add(mod_key)
        mod = sys.modules.get(mod_key)
        if mod is None:
            continue
        # Skip torchlens internals — we don't want to patch our own references.
        if hasattr(mod, "__name__") and getattr(mod, "__name__", "").startswith("torchlens"):
            continue

        try:
            mod_dict = vars(mod)
        except TypeError:
            continue

        for attr_name, attr_val in list(mod_dict.items()):
            # Level 1: module-level references (e.g. ``from torch import cos``)
            if id(attr_val) in mapping:
                try:
                    mod_dict[attr_name] = mapping[id(attr_val)]
                except (TypeError, KeyError):
                    pass
                continue

            # Level 2: class-level attributes that hold torch function references
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    is_type = isinstance(attr_val, type)
            except Exception:
                is_type = False
            if is_type:
                try:
                    cls_dict = vars(attr_val)
                except TypeError:
                    continue
                for cls_attr_name, cls_attr_val in list(cls_dict.items()):
                    if id(cls_attr_val) in mapping:
                        try:
                            setattr(attr_val, cls_attr_name, mapping[id(cls_attr_val)])
                        except (AttributeError, TypeError):
                            pass

            # Level 3: function default arguments (e.g. ``def f(act=torch.relu)``)
            try:
                is_callable = callable(attr_val) and not is_type
            except Exception:
                is_callable = False
            if is_callable:
                _patch_function_defaults(attr_val, mapping)


def _patch_function_defaults(func, mapping) -> None:
    """Patch ``__defaults__`` and ``__kwdefaults__`` of a function if they contain
    original torch function references.

    This handles the case where a function uses a torch function as a default
    argument value, e.g. ``def f(activation=torch.relu)``. The default still
    points to the pre-decoration original; we replace it with the wrapper.
    """
    try:
        defaults = getattr(func, "__defaults__", None)
    except Exception:
        return
    if defaults is not None:
        new_defaults = []
        changed = False
        for d in defaults:
            if id(d) in mapping:
                new_defaults.append(mapping[id(d)])
                changed = True
            else:
                new_defaults.append(d)
        if changed:
            try:
                func.__defaults__ = tuple(new_defaults)
            except (AttributeError, TypeError):
                pass

    try:
        kwdefaults = getattr(func, "__kwdefaults__", None)
    except Exception:
        return
    if kwdefaults is not None and isinstance(kwdefaults, dict):
        for k, v in list(kwdefaults.items()):
            if id(v) in mapping:
                try:
                    kwdefaults[k] = mapping[id(v)]
                except (TypeError, KeyError):
                    pass


def patch_model_instance(model):
    """Level 4 crawl: patch detached torch function references on a model instance.

    Scans ``vars(model)`` and all submodules for instance attributes that are
    original torch functions and replaces them with decorated versions. This
    catches patterns like ``self.act = torch.relu`` in ``__init__``, where the
    reference was captured before decoration.

    Skips dunder attributes to avoid accidentally replacing internal PyTorch
    machinery (e.g. ``__class__``).
    """
    mapping = _state._orig_to_decorated
    if not mapping:
        return

    for module in model.modules():
        try:
            mod_dict = vars(module)
        except TypeError:
            continue
        for attr_name, attr_val in list(mod_dict.items()):
            if attr_name.startswith("__"):
                continue
            if id(attr_val) in mapping:
                try:
                    mod_dict[attr_name] = mapping[id(attr_val)]
                except (TypeError, KeyError):
                    pass
