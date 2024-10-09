import base64
import copy
import multiprocessing as mp
import random
import secrets
import string
import warnings
from sys import getsizeof
from typing import Any, Dict, List, Optional, Type, Callable

import numpy as np
import torch
from IPython import get_ipython
from torch import nn

MAX_FLOATING_POINT_TOLERANCE = 5.5879e-09 + 0.0000001


def identity(x):
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
    if torch.cuda.is_available():
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
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(rng_states["torch_cuda"], "cuda")


def make_random_barcode(barcode_len: int = 8) -> str:
    """Generates a random integer hash for a layer to use as internal label (invisible from user side).

    Args:
        barcode_len: Length of the desired barcode

    Returns:
        Random hash.
    """
    alphabet = string.ascii_letters + string.digits
    barcode = "".join(secrets.choice(alphabet) for _ in range(barcode_len))
    return barcode


def make_short_barcode_from_input(
        things_to_hash: List[Any], barcode_len: int = 16
) -> str:
    """Utility function that takes a list of anything and returns a short hash of it.

    Args:
        things_to_hash: List of things to hash; they must all be convertible to a string.
        barcode_len:

    Returns:
        Short hash of the input.
    """
    barcode = "".join([str(x) for x in things_to_hash])
    barcode = str(hash(barcode))
    barcode = barcode.encode("utf-8")
    barcode = base64.urlsafe_b64encode(barcode)
    barcode = barcode.decode("utf-8")
    barcode = barcode[0:barcode_len]
    return barcode


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


def make_var_iterable(x):
    """Utility function to facilitate dealing with outputs:
    - If not a list, tuple, or dict, make it a list of length 1
    - If a dict, make it a list of the values
    - If a list or tuple, keep it.

    Args:
        x: Output of the function

    Returns:
        Iterable output
    """
    if any([issubclass(type(x), cls) for cls in [list, tuple, set]]):
        return x
    elif issubclass(type(x), dict):
        return list(x.values())
    else:
        return [x]


def index_nested(x: Any, indices: List[int]) -> Any:
    """Utility function to index into a nested list or tuple.

    Args:
        x: Nested list or tuple.
        indices: List of indices to use.

    Returns:
        Indexed object.
    """
    indices = make_var_iterable(indices)
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


def tuple_tolerant_assign(obj_: Any, ind: int, new_value: any):
    """Utility function to assign an entry of a list, tuple, or dict to a new value.

    Args:
        obj_: Tuple to change.
        ind: Index to change.
        new_value: The new value.

    Returns:
        Tuple with the new value swapped out.
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
    tensors_in_obj = []
    tensor_addresses = []
    tensor_addresses_full = []
    tensor_ids_in_obj = []
    for _ in range(search_depth):
        this_stack = search_stack_for_vars_of_type(
            this_stack,
            which_type,
            tensors_in_obj,
            tensor_addresses,
            tensor_addresses_full,
            tensor_ids_in_obj,
            subclass_exceptions,
            allow_repeats,
        )

    if return_addresses:
        return list(zip(tensors_in_obj, tensor_addresses, tensor_addresses_full))
    else:
        return tensors_in_obj


def search_stack_for_vars_of_type(
        current_stack: List,
        which_type: Type,
        tensors_in_obj: List,
        tensor_addresses: List,
        tensor_addresses_full: List,
        tensor_ids_in_obj: List,
        subclass_exceptions: List,
        allow_repeats: bool,
):
    """Helper function that searches current stack for vars of a given type, and
    returns the next stack to search.

    Args:
        current_stack: The current stack.
        which_type: Type of variable to pull out
        tensors_in_obj: List of tensors found in the object so far
        tensor_addresses: Addresses of the tensors found so far
        tensor_addresses_full: explicit instructions for indexing the obj
        tensor_ids_in_obj: List of tensor ids found in the object so far
        subclass_exceptions: Subclasses of tensors not to use.
        allow_repeats: whether to allow repeat tensors

    Returns:
        The next stack.
    """
    next_stack = []
    if len(current_stack) == 0:
        return current_stack
    while len(current_stack) > 0:
        item, address, address_full = current_stack.pop(0)
        item_class = type(item)
        if any(
                [issubclass(item_class, subclass) for subclass in subclass_exceptions]
        ) or ((id(item) in tensor_ids_in_obj) and not allow_repeats):
            continue
        if issubclass(item_class, which_type):
            tensors_in_obj.append(item)
            tensor_addresses.append(address)
            tensor_addresses_full.append(address_full)
            tensor_ids_in_obj.append(id(item))
            continue
        if item_class in [str, int, float, bool, np.ndarray, torch.tensor]:
            continue
        extend_search_stack_from_item(item, address, address_full, next_stack)
    return next_stack


def extend_search_stack_from_item(
        item: Any, address: str, address_full, next_stack: List
):
    """Utility function to iterate through a single item to populate the next stack to search for.

    Args:
        item: The item
        next_stack: Stack to add to
    """
    if type(item) in [list, tuple, set]:
        if address == "":
            next_stack.extend(
                [(x, f"{i}", address_full + [("ind", i)]) for i, x in enumerate(item)]
            )
        else:
            next_stack.extend(
                [
                    (x, f"{address}.{i}", address_full + [("ind", i)])
                    for i, x in enumerate(item)
                ]
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

    for attr_name in dir(item):
        if (attr_name.startswith("__")) or (attr_name == 'T') or ('grad' in attr_name):
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                attr = getattr(item, attr_name)
        except:
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
                    f'{address}.{attr_name.strip("_")}',
                    address_full + [("attr", attr_name)],
                )
            )


def get_attr_values_from_tensor_list(
        tensor_list: List[torch.Tensor], field_name: str
) -> List[Any]:
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
            'H',
            'mH',
            'mT'
        ]:  # avoid annoying warning; if there's more, make a list
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                obj = getattr(obj, a)
        else:
            obj = getattr(obj, a)
    return obj


def nested_assign(obj, addr, val):
    """Given object and an address in that object, assign value to that address."""
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


def iter_accessible_attributes(obj: Any, *, short_circuit: Optional[Callable[[Any, str], bool]] = None):
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


def remove_attributes_starting_with_str(obj: Any, s: str):
    """Given an object removes, any attributes for that object beginning with a given
    substring.

    Args:
        obj: object
        s: string that marks fields to remove
    """
    for field in dir(obj):
        if field.startswith(s):
            delattr(obj, field)


def tensor_all_nan(t: torch.Tensor) -> bool:
    """Returns True if  tensor is all nans, False otherwise."""
    if torch.isnan(t).int().sum() == t.numel():
        return True
    else:
        return False


def tensor_nanequal(t1: torch.Tensor, t2: torch.Tensor, allow_tolerance=False) -> bool:
    """Returns True if the two tensors are equal, allowing for nans."""
    if t1.shape != t2.shape:
        return False

    if t1.dtype != t2.dtype:
        return False

    if not torch.equal(t1.isinf(), t2.isinf()):
        return False

    t1_nonan = torch.nan_to_num(t1, 0.7234691827346)
    t2_nonan = torch.nan_to_num(t2, 0.7234691827346)

    if torch.equal(t1_nonan, t2_nonan):
        return True

    if (
            allow_tolerance
            and (t1_nonan.dtype != torch.bool)
            and (t2_nonan.dtype != torch.bool)
            and ((t1_nonan - t2_nonan).abs().max() <= MAX_FLOATING_POINT_TOLERANCE)
    ):
        return True

    return False


def safe_to(x: Any, device: str):
    """Moves object to device if it's a tensor, does nothing otherwise.

    Args:
        x: The object.
        device: which device to move to

    Returns:
        Object either moved to device if a tensor, same object if otherwise.
    """
    if type(x) == torch.Tensor:
        return clean_to(x, device)
    else:
        return x


def get_tensor_memory_amount(t: torch.Tensor) -> int:
    """Returns the size of a tensor in bytes.

    Args:
        t: Tensor.

    Returns:
        Size of tensor in bytes.
    """
    cpu_data = clean_cpu(t.data)
    if cpu_data.dtype == torch.bfloat16:
        cpu_data = clean_to(cpu_data, torch.float16)
    return getsizeof(np.array(cpu_data))


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


clean_from_numpy = copy.deepcopy(torch.from_numpy)
clean_new_param = copy.deepcopy(torch.nn.Parameter)
clean_clone = copy.deepcopy(torch.clone)
clean_cpu = copy.deepcopy(torch.Tensor.cpu)
clean_cuda = copy.deepcopy(torch.Tensor.cuda)
clean_to = copy.deepcopy(torch.Tensor.to)


def print_override(t: torch.Tensor, func_name: str):
    """Overrides the __str__ and __repr__ methods of Tensor so as not to lead to any infinite recursion.

    Args:
        t: Tensor
        func_name: Either "__str__" or "__repr__"

    Returns:
        The string representation of the tensor.
    """
    cpu_data = clean_cpu(t.data)
    if cpu_data.dtype == torch.bfloat16:
        cpu_data = clean_to(cpu_data, torch.float16)
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
    """Utility function to make a copy of a tensor or parameter when torch is in mutated mode, or just copy
    the thing if it's not a tensor.

    Args:
        x: Input
        detach_tensor: Whether to detach the cloned tensor from the computational graph or not.

    Returns:
        Safely copied variant of the input with same values and same class, but different memory
    """
    if issubclass(type(x), (torch.Tensor, torch.nn.Parameter)):
        if not detach_tensor:
            return clean_clone(x)
        vals_cpu = clean_cpu(x.data)
        if vals_cpu.dtype == torch.bfloat16:
            vals_cpu = clean_to(vals_cpu, torch.float16)
        vals_np = vals_cpu.numpy()
        vals_tensor = clean_from_numpy(vals_np)
        if hasattr(x, "tl_tensor_label_raw"):
            vals_tensor.tl_tensor_label_raw = x.tl_tensor_label_raw
        if type(x) == torch.Tensor:
            return vals_tensor
        elif type(x) == torch.nn.Parameter:
            return clean_new_param(vals_tensor)
    else:
        return copy.copy(x)


def in_notebook():
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
            "pytorch-xray in the main process, since certain operations "
            "depend on execution order."
        )
