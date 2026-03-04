"""Permanent torch function wrapping: decorates all torch ops at import time with toggle-gated wrappers."""

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

funcs_not_to_log = ["numpy", "__array__", "size", "dim"]
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
    """Return the device from the innermost active DeviceContext, or None."""
    global _DeviceContext
    if _DeviceContext is None:
        from torch.utils._device import DeviceContext

        _DeviceContext = DeviceContext
    try:
        from torch.overrides import _get_current_function_mode_stack

        for mode in reversed(_get_current_function_mode_stack()):
            if isinstance(mode, _DeviceContext):
                return mode.device
    except (ImportError, AttributeError):
        pass
    return None


def _maybe_inject_device_kwarg(func_name: str, kwargs: dict) -> dict:
    """Inject ``device`` kwarg for factory functions when a DeviceContext is active.

    Python wrappers bypass PyTorch's C-level TorchFunctionMode dispatch, so
    ``torch.device('meta')`` context won't inject the device kwarg automatically.
    """
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
    """Wrap a torch function with toggle-gated logging.

    When ``_state._logging_enabled`` is False, the wrapper is a near-noop
    (one bool check).  When True, it logs all tensor outputs into the
    active ModelLog.
    """

    @wraps(func)
    def wrapped_func(*args, **kwargs):
        # Fast path: if logging is off, pass through immediately.
        if not _state._logging_enabled or _state._active_model_log is None:
            kwargs = _maybe_inject_device_kwarg(func_name, kwargs)
            return func(*args, **kwargs)

        model_log = _state._active_model_log

        # Initial bookkeeping; check if it's a special function.
        model_log.current_function_call_barcode = 0
        if func_name in funcs_not_to_log:
            return func(*args, **kwargs)

        all_args = args if not kwargs else (*args, *kwargs.values())
        arg_tensorlike = get_vars_of_type_from_obj(all_args, torch.Tensor)

        # Register any buffer tensors in the arguments (only on first encounter).
        for t in arg_tensorlike:
            if hasattr(t, "tl_buffer_address") and not hasattr(t, "tl_tensor_label_raw"):
                log_source_tensor(model_log, t, "buffer", getattr(t, "tl_buffer_address"))

        if (func_name in print_funcs) and (len(arg_tensorlike) > 0):
            out = print_override(args[0], func_name)
            return out

        # Copy the args and kwargs in case they change in-place:
        if model_log.save_function_args:
            arg_copies = tuple([safe_copy(arg) for arg in args])
            kwarg_copies = {k: safe_copy(v) for k, v in kwargs.items()}
        else:
            arg_copies = args
            kwarg_copies = kwargs

        # Call the function, tracking the timing, rng states, and whether it's a nested function
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

        if func_name in ["__setitem__", "zero_", "__delitem__"]:
            out_orig = args[0]

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
            out_orig = safe_copy(out_orig)

        # Log all output tensors
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

            # Propagate label to original tensor for in-place ops so
            # subsequent operations see the updated label.
            if was_inplace:
                if hasattr(out_orig, "tl_tensor_label_raw"):
                    args[0].tl_tensor_label_raw = out_orig.tl_tensor_label_raw

        return out_orig

    # For built-in functions (no __code__), remove __wrapped__ to prevent
    # inspect.unwrap from following it and failing when torch.jit.script
    # tries to get source (e.g. timm's @torch.jit.script decorators).
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
    """Attempts to get the argument names for a function, first by checking the signature, then
    by checking the documentation. Adds these names to ``_state._func_argnames``.

    Stores under the stripped name (no leading/trailing underscores) so lookups
    via ``func_name.strip("_")`` are consistent (#82).
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
    argnames = tuple([arg for arg in argnames if arg not in ["self", "cls"]])
    _state._func_argnames[storage_key] = argnames


# ---------------------------------------------------------------------------
# One-time decoration at import time
# ---------------------------------------------------------------------------


def decorate_all_once():
    """Decorate all torch functions once at import time.

    Wraps every function in ``ORIG_TORCH_FUNCS`` with a toggle-gated
    wrapper.  Pre-computes ``_state._func_argnames`` and populates
    ``_state._orig_to_decorated`` / ``_state._decorated_to_orig``.
    Also permanently installs ``torch.identity``.

    This function is idempotent — calling it again is a no-op.
    """
    if _state._orig_to_decorated:
        return  # already decorated

    function_class = type(lambda: 0)
    builtin_class = type(torch.mean)
    method_class = type(torch.Tensor.__add__)
    wrapper_class = type(torch.Tensor.__getitem__)
    getset_class = type(torch.Tensor.real)

    for namespace_name, func_name in ORIG_TORCH_FUNCS:
        namespace_key = namespace_name.replace("torch.", "")
        local_func_namespace = nested_getattr(torch, namespace_key)
        if not hasattr(local_func_namespace, func_name):
            continue
        orig_func = getattr(local_func_namespace, func_name)

        # Skip already-decorated functions (duplicates in ORIG_TORCH_FUNCS)
        if getattr(orig_func, "tl_is_decorated_function", False):
            continue

        # Pre-compute argnames
        if func_name.strip("_") not in _state._func_argnames:
            get_func_argnames(orig_func, func_name)

        if type(orig_func) in [function_class, builtin_class, method_class, wrapper_class]:
            # If this exact original func was already wrapped under a different
            # namespace (e.g. torch.cos and torch._VF.cos share the same
            # builtin), reuse the existing wrapper to keep mappings consistent.
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
                _state._orig_to_decorated[id(orig_func)] = new_func
                _state._decorated_to_orig[id(new_func)] = orig_func
                _state._decorated_func_mapper[new_func] = orig_func
                _state._decorated_func_mapper[orig_func] = new_func
            except (AttributeError, TypeError):
                pass

        elif type(orig_func) is getset_class:
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
                # #31: Only add mapper entries if setattr succeeded
                _state._orig_to_decorated[id(orig_func)] = new_property
                _state._decorated_to_orig[id(new_property)] = orig_func
                _state._decorated_func_mapper[new_property] = orig_func
                _state._decorated_func_mapper[orig_func] = new_property
            except (AttributeError, TypeError):
                pass

    # Register wrappers with JIT builtin table so torch.jit.script
    # recognizes them as known ATen ops (prevents JIT compilation errors).
    try:
        import torch.jit._builtins as _jit_builtins

        for orig_id, decorated_func in _state._orig_to_decorated.items():
            builtin_name = _jit_builtins._builtin_table.get(orig_id)
            if builtin_name is not None:
                _jit_builtins._builtin_table[id(decorated_func)] = builtin_name
                # For properties, also register getter/setter/deleter if they exist
                if isinstance(decorated_func, property):
                    for accessor in (decorated_func.fget, decorated_func.fset, decorated_func.fdel):
                        if accessor is not None:
                            _jit_builtins._builtin_table[id(accessor)] = builtin_name
    except (ImportError, AttributeError):
        pass  # JIT internals may change across PyTorch versions

    # Build the set of device constructor function names so the fast-path
    # wrapper can inject the device kwarg when a torch.device context is active.
    try:
        from torch.utils._device import _device_constructors

        # Clear the lru_cache so it re-evaluates with wrapped functions.
        _device_constructors.cache_clear()
        for ctor in _device_constructors():
            name = getattr(ctor, "__name__", None)
            if name:
                _DEVICE_CONSTRUCTOR_NAMES.add(name)
    except (ImportError, AttributeError):
        pass

    # Permanently install torch.identity
    new_identity = torch_func_decorator(identity, "identity")
    torch.identity = new_identity


# ---------------------------------------------------------------------------
# sys.modules deep crawl
# ---------------------------------------------------------------------------


def patch_detached_references():
    """Crawl ``sys.modules`` and replace detached references to original
    torch functions with their decorated counterparts.

    Levels:
      1. Module ``__dict__`` values
      2. Class ``__dict__`` values found in modules
      3. Function ``__defaults__`` / ``__kwdefaults__``
      4. (Model instances are handled separately at log_forward_pass time)

    Uses ``_state._crawled_module_keys`` to avoid re-scanning modules.
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
        # Skip torchlens internals
        if hasattr(mod, "__name__") and getattr(mod, "__name__", "").startswith("torchlens"):
            continue

        try:
            mod_dict = vars(mod)
        except TypeError:
            continue

        for attr_name, attr_val in list(mod_dict.items()):
            # Level 1: module-level references
            if id(attr_val) in mapping:
                try:
                    mod_dict[attr_name] = mapping[id(attr_val)]
                except (TypeError, KeyError):
                    pass
                continue

            # Level 2: class dicts
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

            # Level 3: function defaults
            try:
                is_callable = callable(attr_val) and not is_type
            except Exception:
                is_callable = False
            if is_callable:
                _patch_function_defaults(attr_val, mapping)


def _patch_function_defaults(func, mapping) -> None:
    """Patch __defaults__ and __kwdefaults__ of a function if they contain
    original torch function references."""
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

    Scans ``vars(model)`` and all submodules for attributes that are original
    torch functions and replaces them with decorated versions.
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
