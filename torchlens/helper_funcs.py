import base64
import copy
import inspect
import multiprocessing as mp
import random
import secrets
import string
import warnings
from typing import Any, Callable, Dict, List, Optional, Type

import numpy as np
import torch
from IPython import get_ipython
from torch import nn

MAX_FLOATING_POINT_TOLERANCE = 3e-6

_ATTR_SKIP_SET = frozenset({"T", "mT", "real", "imag", "H"})

_cuda_available: Optional[bool] = None


def _is_cuda_available() -> bool:
    """Return True if CUDA is available on this machine (cached after first call)."""
    global _cuda_available
    if _cuda_available is None:
        _cuda_available = torch.cuda.is_available()
    return _cuda_available


def identity(x: Any) -> Any:
    """Return the input unchanged."""
    return x


def set_random_seed(seed: int):
    """Sets the random seed for all random number generators.

    Args:
        seed: Seed to set.

    Returns:
        Nothing.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_current_rng_states() -> Dict:
    """Utility function to fetch sufficient information from all RNG states to recover the same state later.

    Returns:
        Dict with sufficient information to recover all RNG states.
    """
    rng_dict = {
        "random": random.getstate(),
        "np": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if _is_cuda_available():
        rng_dict["torch_cuda"] = torch.cuda.get_rng_state("cuda")
    return rng_dict


def set_rng_from_saved_states(rng_states: Dict):
    """Utility function to set the state of random seeds to a cached value.

    Args:
        rng_states: Dict of rng_states saved by get_random_seed_states

    Returns:
        Nothing, but correctly sets all random seed states.
    """
    random.setstate(rng_states["random"])
    np.random.set_state(rng_states["np"])
    torch.random.set_rng_state(rng_states["torch"])
    if _is_cuda_available() and "torch_cuda" in rng_states:
        torch.cuda.set_rng_state(rng_states["torch_cuda"], "cuda")


# ---------------------------------------------------------------------------
# Autocast state capture/restore
# ---------------------------------------------------------------------------

_AUTOCAST_DEVICES = ("cpu", "cuda")


def log_current_autocast_state() -> Dict:
    """Capture the current autocast enabled/dtype state for all supported devices.

    Returns:
        Dict mapping device name to {enabled: bool, dtype: torch.dtype}.
    """
    state = {}
    for device in _AUTOCAST_DEVICES:
        try:
            state[device] = {
                "enabled": torch.is_autocast_enabled(device),
                "dtype": torch.get_autocast_dtype(device),
            }
        except (RuntimeError, TypeError):
            pass
    return state


class AutocastRestore:
    """Context manager that restores saved autocast states.

    Usage::

        with AutocastRestore(saved_state):
            result = func(*args, **kwargs)
    """

    __slots__ = ("_autocast_state", "_contexts")

    def __init__(self, autocast_state: Dict):
        self._autocast_state = autocast_state
        self._contexts = []

    def __enter__(self):
        for device, state in self._autocast_state.items():
            if state["enabled"]:
                ctx = torch.amp.autocast(device, dtype=state["dtype"])
                ctx.__enter__()
                self._contexts.append(ctx)
        return self

    def __exit__(self, *exc_info):
        for ctx in reversed(self._contexts):
            ctx.__exit__(*exc_info)


def _safe_copy_arg(arg: Any) -> Any:
    """Copy a single argument safely, avoiding deepcopy on arbitrary objects.

    Clones tensors, recurses into standard containers (list, tuple, dict),
    and leaves everything else as-is.  This prevents infinite loops that
    copy.deepcopy can trigger on complex tensor wrappers (e.g. ESCNN
    GeometricTensor) while still protecting user inputs from in-place
    mutations like device moves.

    Note: custom objects containing tensors are passed by reference.  If the
    model is on a different device, _fetch_label_move_input_tensors may
    mutate the wrapper's tensor attribute in-place.  This is acceptable
    because pre-fix such inputs caused an infinite hang.
    """
    if isinstance(arg, torch.Tensor):
        return arg.clone()
    elif isinstance(arg, dict):
        return type(arg)({k: _safe_copy_arg(v) for k, v in arg.items()})
    elif isinstance(arg, (list, tuple)):
        copied = [_safe_copy_arg(item) for item in arg]
        return type(arg)(*copied) if hasattr(type(arg), "_fields") else type(arg)(copied)
    else:
        return arg


def safe_copy_args(args: list) -> list:
    """Safely copy a list of function arguments."""
    return [_safe_copy_arg(arg) for arg in args]


def safe_copy_kwargs(kwargs: dict) -> dict:
    """Safely copy a dict of keyword arguments."""
    return {key: _safe_copy_arg(val) for key, val in kwargs.items()}


def _model_expects_single_arg(model: nn.Module) -> Optional[bool]:
    """Check if the model's forward expects exactly 1 positional arg (excluding self).

    Returns True if exactly 1, False if not, None if introspection fails.
    """
    try:
        spec = inspect.getfullargspec(model.forward)
    except (TypeError, ValueError):
        return None
    named_args = [a for a in spec.args if a != "self"]
    if spec.varargs is not None:
        return False
    return len(named_args) == 1


def normalize_input_args(input_args, model: nn.Module) -> list:
    """Normalize input_args into a list suitable for `model(*input_args)`.

    Handles the ambiguity when the user passes a tuple or list: it could be
    multiple positional args, or a single arg that happens to be a
    tuple/list (issue #43).  Resolves the ambiguity by checking the model's
    forward signature.
    """
    if type(input_args) in (tuple, list):
        single = _model_expects_single_arg(model)
        if single and len(input_args) != 1:
            # Model expects 1 arg; the tuple/list IS the arg.
            input_args = [input_args]
        elif type(input_args) is tuple:
            input_args = list(input_args)
        # If already a list and not wrapping, leave as-is.
    elif input_args is not None:
        input_args = [input_args]
    if not input_args:
        input_args = []
    return input_args


def make_random_barcode(barcode_len: int = 8) -> str:
    """Generates a random alphanumeric identifier string for a layer to use as internal label (invisible from user side).

    Args:
        barcode_len: Length of the desired barcode

    Returns:
        Random alphanumeric string.
    """
    alphabet = string.ascii_letters + string.digits
    barcode = "".join(secrets.choice(alphabet) for _ in range(barcode_len))
    return barcode


def make_short_barcode_from_input(things_to_hash: List[Any], barcode_len: int = 16) -> str:
    """Utility function that takes a list of anything and returns a short hash of it.

    Args:
        things_to_hash: List of things to hash; they must all be convertible to a string.
        barcode_len:

    Returns:
        Short hash of the input.
    """
    barcode = "\x00".join([str(x) for x in things_to_hash])
    barcode = str(hash(barcode))
    barcode = barcode.encode("utf-8")
    barcode = base64.urlsafe_b64encode(barcode)
    barcode = barcode.decode("utf-8")
    barcode = barcode[0:barcode_len]
    return barcode


def _get_func_call_stack(num_context_lines: int = 7) -> List:
    """Build a list of FuncCallLocation objects for the current call stack.

    Filters out torchlens internals and ``_call_impl`` frames, keeping only
    user-visible frames starting from the ``log_forward_pass`` call site
    through the model's ``forward`` method and any deeper user calls.

    Uses ``sys._getframe()`` instead of ``inspect.stack()`` to avoid
    expensive per-frame source file I/O.  Source context is loaded lazily
    by ``FuncCallLocation`` on first access via ``linecache``.

    Args:
        num_context_lines: Number of source lines to show on each side of
            the call line.  The total context window is
            ``2 * num_context_lines + 1``.

    Returns:
        List[FuncCallLocation] ordered shallow-to-deep.
    """
    import sys

    from .data_classes import FuncCallLocation

    _TORCHLENS_SUFFIXES = (
        "model_log.py",
        "torchlens/helper_funcs.py",
        "torchlens/user_funcs.py",
        "torchlens/trace_model.py",
        "torchlens/logging_funcs.py",
        "torchlens/decorate_torch.py",
        "torchlens/model_funcs.py",
    )

    # Phase 1: Collect lightweight frame data — only co_filename, co_name, f_lineno.
    # Do NOT do f_locals/f_globals dict lookups yet (expensive, ~50/call).
    raw_frames = []
    frame = sys._getframe(0)
    while frame is not None:
        raw_frames.append(
            (
                frame.f_code.co_filename,
                frame.f_code.co_name,
                frame.f_lineno,
                frame,  # keep reference for phase 2 func_obj lookup
            )
        )
        frame = frame.f_back

    # Walk bottom-up (deepest caller last → first in output) and collect
    # non-internal frames.  Start tracking once we hit a ``forward`` frame,
    # but also include the frame *before* the first ``forward`` (the user's
    # script that called ``log_forward_pass``).
    tracking = False
    pre_forward_frame_idx = None
    filtered_indices = []

    for idx in range(len(raw_frames) - 1, -1, -1):
        filename, func_name, lineno, frame_ref = raw_frames[idx]

        # Skip torchlens internals and PyTorch _call_impl
        if any(filename.endswith(s) for s in _TORCHLENS_SUFFIXES):
            continue
        if "_call_impl" in func_name:
            continue

        if func_name == "forward" and not tracking:
            tracking = True
            # Look for the user-script frame that called log_forward_pass
            for j in range(idx + 1, len(raw_frames)):
                j_filename, j_func_name, _, _ = raw_frames[j]
                if (
                    not any(j_filename.endswith(s) for s in _TORCHLENS_SUFFIXES)
                    and "_call_impl" not in j_func_name
                ):
                    pre_forward_frame_idx = j
                    break

        if tracking:
            filtered_indices.append(idx)

    # Prepend the log_forward_pass call-site frame if found and not already included
    if pre_forward_frame_idx is not None and pre_forward_frame_idx not in filtered_indices:
        filtered_indices.append(pre_forward_frame_idx)

    # Phase 2: Build FuncCallLocation objects only for surviving frames (~5-10).
    # Do the expensive f_locals/f_globals dict lookup only here.
    result = []
    for idx in filtered_indices:
        filename, func_name, lineno, frame_ref = raw_frames[idx]
        func_obj = frame_ref.f_locals.get(func_name) or frame_ref.f_globals.get(func_name)
        loc = FuncCallLocation(
            file=filename,
            line_number=lineno,
            func_name=func_name,
            num_context_lines_requested=num_context_lines,
            _frame_func_obj=func_obj,
        )
        result.append(loc)

    return result


def is_iterable(obj: Any) -> bool:
    """Checks if an object is iterable.

    Args:
        obj: Object to check.

    Returns:
        True if object is iterable, False otherwise.
    """
    try:
        iter(obj)
        return True
    except TypeError:
        return False


def ensure_iterable(obj: Any) -> Any:
    """Utility function to facilitate dealing with outputs:
    - If not a list, tuple, or dict, make it a list of length 1
    - If a dict, make it a list of the values
    - If a list or tuple, keep it.

    Args:
        obj: Output of the function

    Returns:
        Iterable output
    """
    if any([issubclass(type(obj), cls) for cls in [list, tuple, set]]):
        return obj
    elif issubclass(type(obj), dict):
        return list(obj.values())
    else:
        return [obj]


def index_nested(x: Any, indices: List[int]) -> Any:
    """Utility function to index into a nested list or tuple.

    Args:
        x: Nested list or tuple.
        indices: List of indices to use.

    Returns:
        Indexed object.
    """
    indices = ensure_iterable(indices)
    for i in indices:
        x = x[i]
    return x


def remove_entry_from_list(list_: List, entry: Any):
    """Removes all instances of an entry from a list if present, in-place.

    Args:
        list_: the list
        entry: the entry to remove
    """
    while entry in list_:
        list_.remove(entry)


def assign_to_sequence_or_dict(obj_: Any, ind: int, new_value: Any) -> Any:
    """Utility function to assign an entry of a list, tuple, or dict to a new value.

    Args:
        obj_: Sequence or dict to change.
        ind: Index to change.
        new_value: The new value.

    Returns:
        Sequence or dict with the new value swapped out.
    """
    if type(obj_) == tuple:
        list_ = list(obj_)
        list_[ind] = new_value
        return tuple(list_)

    obj_[ind] = new_value
    return obj_


def int_list_to_compact_str(int_list: List[int]) -> str:
    """Given a list of integers, returns a compact string representation of the list, where
    contiguous stretches of the integers are represented as ranges (e.g., [1 2 3 4] becomes "1-4"),
    and all such ranges are separated by commas.

    Args:
        int_list: List of integers.

    Returns:
        Compact string representation of the list.
    """
    int_list = sorted(int_list)
    if len(int_list) == 0:
        return ""
    if len(int_list) == 1:
        return str(int_list[0])
    ranges = []
    start = int_list[0]
    end = int_list[0]
    for i in range(1, len(int_list)):
        if int_list[i] == end + 1:
            end = int_list[i]
        else:
            if start == end:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{end}")
            start = int_list[i]
            end = int_list[i]
    if start == end:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{end}")
    return ",".join(ranges)


def get_vars_of_type_from_obj(
    obj: Any,
    which_type: Type,
    subclass_exceptions: Optional[List] = None,
    search_depth: int = 3,
    return_addresses=False,
    allow_repeats=False,
) -> List:
    """Recursively finds all tensors in an object, excluding specified subclasses (e.g., parameters)
    up to the given search depth.

    Args:
        obj: Object to search.
        which_type: Type of variable to pull out
        subclass_exceptions: subclasses that you don't want to pull out.
        search_depth: How many layers deep to search before giving up.
        return_addresses: if True, then returns list of tuples (object, address), where the
            address is how you'd index to get the object
        allow_repeats: whether to allow repeats of the same tensor

    Returns:
        List of objects of desired type found in the input object.
    """
    if subclass_exceptions is None:
        subclass_exceptions = []
    this_stack = [(obj, "", [])]
    found_items = []
    found_addresses = []
    found_addresses_full = []
    found_ids = []
    for _ in range(search_depth):
        this_stack = search_stack_for_vars_of_type(
            this_stack,
            which_type,
            found_items,
            found_addresses,
            found_addresses_full,
            found_ids,
            subclass_exceptions,
            allow_repeats,
        )

    if return_addresses:
        return list(zip(found_items, found_addresses, found_addresses_full))
    else:
        return found_items


def search_stack_for_vars_of_type(
    current_stack: List,
    which_type: Type,
    found_items: List,
    found_addresses: List,
    found_addresses_full: List,
    found_ids: List,
    subclass_exceptions: List,
    allow_repeats: bool,
):
    """Helper function that searches current stack for vars of a given type, and
    returns the next stack to search.

    Args:
        current_stack: The current stack.
        which_type: Type of variable to pull out
        found_items: List of items of the target type found so far
        found_addresses: Addresses of the items found so far
        found_addresses_full: explicit instructions for indexing the obj
        found_ids: List of ids of found items (used for deduplication)
        subclass_exceptions: Subclasses of the target type not to collect.
        allow_repeats: whether to allow repeat items

    Returns:
        The next stack.
    """
    next_stack = []
    if len(current_stack) == 0:
        return current_stack
    while len(current_stack) > 0:
        item, address, address_full = current_stack.pop(0)
        item_class = type(item)
        if any([issubclass(item_class, subclass) for subclass in subclass_exceptions]) or (
            (id(item) in found_ids) and not allow_repeats
        ):
            continue
        if issubclass(item_class, which_type):
            found_items.append(item)
            found_addresses.append(address)
            found_addresses_full.append(address_full)
            found_ids.append(id(item))
            continue
        if item_class in [str, int, float, bool, np.ndarray, torch.Tensor]:
            continue
        extend_search_stack_from_item(item, address, address_full, next_stack)
    return next_stack


def extend_search_stack_from_item(item: Any, address: str, address_full, next_stack: List):
    """Utility function to iterate through a single item to populate the next stack to search for.

    Args:
        item: The item
        address: Human-readable dot-separated path string (e.g. "0.weight").
        address_full: List of (type, key) tuples for programmatic indexing into the nested structure.
        next_stack: Stack to add to
    """
    from . import _state

    if type(item) in [list, tuple, set]:
        if address == "":
            next_stack.extend(
                [(x, f"{i}", address_full + [("ind", i)]) for i, x in enumerate(item)]
            )
        else:
            next_stack.extend(
                [(x, f"{address}.{i}", address_full + [("ind", i)]) for i, x in enumerate(item)]
            )

    if issubclass(type(item), dict):
        if address == "":
            next_stack.extend(
                [(val, key, address_full + [("ind", key)]) for key, val in item.items()]
            )
        else:
            next_stack.extend(
                [
                    (val, f"{address}.{key}", address_full + [("ind", key)])
                    for key, val in item.items()
                ]
            )

    # Cache dir() results per type — dir() walks the full MRO and is expensive.
    # Same types (e.g. every nn.Conv2d) have identical dir() output.
    obj_type = type(item)
    if obj_type not in _state._dir_cache:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _state._dir_cache[obj_type] = [
                a
                for a in dir(item)
                if not a.startswith("__") and a not in _ATTR_SKIP_SET and "grad" not in a
            ]
    filtered_attrs = _state._dir_cache[obj_type]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for attr_name in filtered_attrs:
            try:
                attr = getattr(item, attr_name)
            except Exception:
                continue
            attr_cls = type(attr)
            if attr_cls in [str, int, float, bool, np.ndarray]:
                continue
            if callable(attr) and not issubclass(attr_cls, nn.Module):
                continue
            if address == "":
                next_stack.append(
                    (attr, attr_name.strip("_"), address_full + [("attr", attr_name)])
                )
            else:
                next_stack.append(
                    (
                        attr,
                        f"{address}.{attr_name.strip('_')}",
                        address_full + [("attr", attr_name)],
                    )
                )


def get_attr_values_from_tensor_list(tensor_list: List[torch.Tensor], field_name: str) -> List[Any]:
    """For a list of tensors, gets the value of a given attribute from each tensor that has that attribute.

    Args:
        tensor_list: List of tensors to search.
        field_name: Name of the field to check in the tensor.

    Returns:
        List of marks from the tensors.
    """
    marks = []
    for tensor in tensor_list:
        mark = getattr(tensor, field_name, None)
        if mark is not None:
            marks.append(mark)
    return marks


def nested_getattr(obj: Any, attr: str) -> Any:
    """Helper function that takes in an object, and a string of attributes separated by '.' and recursively
    returns the attribute.

    Args:
        obj: Any object, e.g. "torch"
        attr: String specifying the nested attribute, e.g. "nn.functional"

    Returns:
        The attribute specified by the string.
    """
    if attr == "":
        return obj

    attributes = attr.split(".")
    for i, a in enumerate(attributes):
        if a in [
            "volatile",
            "T",
            "H",
            "mH",
            "mT",
        ]:  # avoid annoying warning; if there's more, make a list
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                obj = getattr(obj, a)
        else:
            obj = getattr(obj, a)
    return obj


def nested_assign(obj: Any, addr: List[tuple], val: Any) -> None:
    """Walk into a nested structure following an address path and assign a value at the final location.

    Args:
        obj: The root object to traverse.
        addr: A list of (kind, key) tuples specifying how to traverse the structure.
            Each tuple is either ("ind", key) for index/dict access (obj[key]) or
            ("attr", name) for attribute access (getattr(obj, name)).
        val: The value to assign at the destination.
    """
    for i, (entry_type, entry_val) in enumerate(addr):
        if i == len(addr) - 1:
            if entry_type == "ind":
                obj[entry_val] = val
            elif entry_type == "attr":
                setattr(obj, entry_val, val)
        else:
            if entry_type == "ind":
                obj = obj[entry_val]
            elif entry_type == "attr":
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    obj = getattr(obj, entry_val)


def iter_accessible_attributes(
    obj: Any, *, short_circuit: Optional[Callable[[Any, str], bool]] = None
):
    for attr_name in dir(obj):
        if short_circuit and short_circuit(obj, attr_name):
            continue

        # Attribute access can fail for any number of reasons, especially when
        # working with objects that we don't know anything about.  This
        # function makes a best-effort attempt to access every attribute, but
        # gracefully skips any that cause problems.

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                attr = getattr(obj, attr_name)
            except Exception:
                continue

        yield attr_name, attr


def remove_attributes_with_prefix(obj: Any, prefix: str) -> None:
    """Given an object, removes any attributes beginning with a given prefix.

    Args:
        obj: object from which to remove attributes
        prefix: string prefix that marks fields to remove
    """
    for field in dir(obj):
        if field.startswith(prefix):
            delattr(obj, field)


def tensor_all_nan(tensor: torch.Tensor) -> bool:
    """Returns True if tensor is all nans, False otherwise."""
    if torch.isnan(tensor).int().sum() == tensor.numel():
        return True
    else:
        return False


def tensor_nanequal(tensor_a: torch.Tensor, tensor_b: torch.Tensor, allow_tolerance=False) -> bool:
    """Returns True if the two tensors are equal, allowing for nans."""
    if tensor_a.shape != tensor_b.shape:
        return False

    if tensor_a.dtype != tensor_b.dtype:
        return False

    if not torch.equal(tensor_a.isinf(), tensor_b.isinf()):
        return False

    if tensor_a.is_complex():
        tensor_a_nonan = torch.view_as_complex(
            torch.nan_to_num(torch.view_as_real(tensor_a), 0.7234691827346)
        )
        tensor_b_nonan = torch.view_as_complex(
            torch.nan_to_num(torch.view_as_real(tensor_b), 0.7234691827346)
        )
    else:
        tensor_a_nonan = torch.nan_to_num(tensor_a, 0.7234691827346)
        tensor_b_nonan = torch.nan_to_num(tensor_b, 0.7234691827346)

    if torch.equal(tensor_a_nonan, tensor_b_nonan):
        return True

    if (
        allow_tolerance
        and (tensor_a_nonan.dtype != torch.bool)
        and (tensor_b_nonan.dtype != torch.bool)
        and ((tensor_a_nonan - tensor_b_nonan).abs().max() <= MAX_FLOATING_POINT_TOLERANCE)
    ):
        return True

    return False


def safe_to(obj: Any, device: str) -> Any:
    """Moves object to device if it's a tensor, does nothing otherwise.

    Args:
        obj: The object.
        device: which device to move to

    Returns:
        Object either moved to device if a tensor, same object if otherwise.
    """
    from ._state import pause_logging

    if type(obj) == torch.Tensor:
        with pause_logging():
            return obj.to(device)
    else:
        return obj


def get_tensor_memory_amount(t: torch.Tensor) -> int:
    """Returns the size of a tensor in bytes.

    Args:
        t: Tensor.

    Returns:
        Size of tensor in bytes.
    """
    from ._state import pause_logging

    try:
        with pause_logging():
            return t.nelement() * t.element_size()
    except Exception:
        return 0


def human_readable_size(size: int, decimal_places: int = 1) -> str:
    """Utility function to convert a size in bytes to a human-readable format.

    Args:
        size: Number of bytes.
        decimal_places: Number of decimal places to use.

    Returns:
        String with human-readable size.
    """
    for unit in ["B", "KB", "MB", "GB", "TB", "PB"]:
        if size < 1024.0 or unit == "PB":
            break
        size /= 1024.0
    if unit == "B":
        size = int(size)
    else:
        size = np.round(size, decimals=decimal_places)
    return f"{size} {unit}"


def print_override(t: torch.Tensor, func_name: str):
    """Overrides the __str__ and __repr__ methods of Tensor so as not to lead to any infinite recursion.

    Args:
        t: Tensor
        func_name: Either "__str__" or "__repr__"

    Returns:
        The string representation of the tensor.
    """
    from ._state import pause_logging

    with pause_logging():
        cpu_data = t.data.cpu()
        if cpu_data.dtype == torch.bfloat16:
            cpu_data = cpu_data.to(torch.float16)
    n = np.array(cpu_data)
    np_str = getattr(n, func_name)()
    np_str = np_str.replace("array", "tensor")
    np_str = np_str.replace("\n", "\n ")
    if t.grad_fn is not None:
        grad_fn_str = f", grad_fn={type(t.grad_fn).__name__})"
        np_str = np_str[0:-1] + grad_fn_str
    elif t.requires_grad:
        np_str = np_str[0:-1] + ", requires_grad=True)"
    return np_str


def safe_copy(x, detach_tensor: bool = False):
    """Utility function to make a copy of a tensor or parameter, or just copy
    the thing if it's not a tensor.  Uses ``pause_logging()`` so that
    clone / cpu / to calls don't get logged.

    Args:
        x: Input
        detach_tensor: Whether to detach the cloned tensor from the computational graph or not.

    Returns:
        Safely copied variant of the input with same values and same class, but different memory
    """
    from ._state import pause_logging

    if issubclass(type(x), (torch.Tensor, torch.nn.Parameter)):
        with pause_logging():
            if not detach_tensor:
                return x.clone()
            vals_cpu = x.data.cpu()
            if vals_cpu.dtype == torch.bfloat16:
                # numpy doesn't support bfloat16; convert to float16 first, then back after numpy round-trip
                vals_cpu = vals_cpu.to(torch.float16)
            vals_np = vals_cpu.numpy()
            vals_tensor = torch.from_numpy(vals_np)
            if hasattr(x, "tl_tensor_label_raw"):
                vals_tensor.tl_tensor_label_raw = x.tl_tensor_label_raw
            if type(x) == torch.Tensor:
                return vals_tensor
            elif type(x) == torch.nn.Parameter:
                return torch.nn.Parameter(vals_tensor)
    else:
        return copy.copy(x)


def in_notebook() -> bool:
    """Return True if the code is running inside a Jupyter notebook, False otherwise."""
    try:
        if "IPKernelApp" not in get_ipython().config:
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True


def warn_parallel():
    """
    Utility function to give raise error if it's being run in parallel processing.
    """
    if mp.current_process().name != "MainProcess":
        raise RuntimeError(
            "WARNING: It looks like you are using parallel execution; only run "
            "torchlens in the main process, since certain operations "
            "depend on execution order."
        )
