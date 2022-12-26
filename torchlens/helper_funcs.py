import multiprocessing as mp
import random
import secrets
import string
from collections import OrderedDict
from sys import getsizeof
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import torch
from IPython import get_ipython
from torch import nn


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


def get_rng_states() -> Dict:
    """Utility function to fetch sufficient information from all RNG states to recover the same state later.

    Returns:
        Dict with sufficient information to recover all RNG states.
    """
    rng_dict = {'random': random.getstate(),
                'np': np.random.get_state(),
                'torch': torch.random.get_rng_state()}
    if torch.cuda.is_available():
        rng_dict['torch_cuda'] = torch.cuda.get_rng_state('cuda')
    return rng_dict


def set_rng_states(rng_states: Dict):
    """Utility function to set the state of random seeds to a cached value.

    Args:
        rng_states: Dict of rng_states saved by get_random_seed_states

    Returns:
        Nothing, but correctly sets all random seed states.
    """
    random.setstate(rng_states['random'])
    np.random.set_state(rng_states['np'])
    torch.random.set_rng_state(rng_states['torch'])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(rng_states['torch_cuda'], 'cuda')


def warn_parallel():
    """
    Utility function to give raise error if it's being run in parallel processing.
    """
    if mp.current_process().name != 'MainProcess':
        raise RuntimeError("WARNING: It looks like you are using parallel execution; only run "
                           "pytorch-xray in the main process, since certain operations "
                           "depend on execution order.")


def make_barcode() -> str:
    """Generates a random integer hash for a layer to use as internal label (invisible from user side).

    Returns:
        Random hash.
    """
    alphabet = string.ascii_letters + string.digits
    barcode = ''.join(secrets.choice(alphabet) for _ in range(6))
    return barcode


def make_var_iterabl(x):
    """Utility function to facilitate dealing with outputs:
    - If not a list, tuple, or dict, make it a list of length 1
    - If a dict, make it a list of the values
    - If a list or tuple, keep it.

    Args:
        x: Output of the function

    Returns:
        Iterable output
    """
    if type(x) in [tuple, list, set]:
        return x
    if issubclass(type(x), dict):
        return list(x.values())
    else:
        return [x]

def is_iterable(obj: Any) -> bool:
    """ Checks if an object is iterable.

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


def tuple_assign(tuple_: tuple, ind: int, new_value: any):
    """Utility function to assign an entry of a tuple to a new value.

    Args:
        tuple_: Tuple to change.
        ind: Index to change.
        new_value: The new value.

    Returns:
        Tuple with the new value swapped out.
    """
    list_ = list(tuple_)
    list_[ind] = new_value
    return tuple(list_)


def in_notebook():
    try:
        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True


def get_vars_of_type_from_obj(obj: Any,
                              which_type: Type,
                              subclass_exceptions: Optional[List] = None,
                              search_depth: int = 5) -> List:
    """Recursively finds all objects of a given type, or a subclass of that type,
    up to the given search depth.

    Args:
        obj: Object to search.
        which_type: type of object to pull out
        subclass_exceptions: subclasses that you don't want to pull out.
        search_depth: How many layers deep to search before giving up.

    Returns:
        List of objects of desired type found in the input object.
    """
    if subclass_exceptions is None:
        subclass_exceptions = []
    this_stack = [obj]
    tensors_in_obj = []
    tensor_ids_in_obj = []
    for _ in range(search_depth):
        next_stack = []
        if len(this_stack) == 0:
            break
        while len(this_stack) > 0:
            item = this_stack.pop()
            item_class = type(item)
            if any([issubclass(item_class, subclass) for subclass in subclass_exceptions]):
                continue
            if all([issubclass(item_class, which_type), id(item) not in tensor_ids_in_obj]):
                tensors_in_obj.append(item)
                tensor_ids_in_obj.append(id(item))
            if hasattr(item, 'shape'):
                continue
            if is_iterable(item):
                for i in item:
                    next_stack.append(i)
            for attr_name in dir(item):
                if attr_name.startswith('__'):
                    continue
                try:
                    attr = getattr(item, attr_name)
                except AttributeError:
                    continue
                attr_cls = type(attr)
                if attr_cls in [str, int, float, bool, np.ndarray]:
                    continue
                if callable(attr) and not issubclass(attr_cls, nn.Module):
                    continue
                next_stack.append(attr)
        this_stack = next_stack
    return tensors_in_obj


def mark_tensors_in_obj(x: Any, field_name: str, field_val: Any):
    """Marks all tensors in an object with a specified field attribute.

    Args:
        x: the input object
        field_name: name of the field to add to the tensor.
        field_val: value of the field to add to the tensor.

    Returns:
        Nothing.
    """
    input_tensors = get_vars_of_type_from_obj(x, torch.Tensor)
    for tensor in input_tensors:
        setattr(tensor, field_name, field_val)


def barcode_tensors_in_obj(x: Any):
    """Marks all tensors in the input saying they came as input.

    Args:
        x: Input to the model.

    Returns:
        Nothing.
    """
    input_tensors = get_vars_of_type_from_obj(x, torch.Tensor)
    for tensor in input_tensors:
        setattr(tensor, 'tl_barcode', make_barcode())


def tensor_in_obj_has_mark(x: Any, field_name: str, field_val: Any):
    """Checks if any tensors in the input have a given mark.

    Args:
        x: Input to the model.
        field_name: Name of the field to check in the tensor.
        field_val: Value of the field to check in the tensor.

    Returns:
        True if any tensors in the input have the given mark, False otherwise.
    """
    input_tensors = get_vars_of_type_from_obj(x, torch.Tensor)
    for tensor in input_tensors:
        if getattr(tensor, field_name, None) == field_val:
            return True
    return False


def get_tensors_in_obj_with_mark(x: Any, field_name: str, field_val: Any) -> List[torch.Tensor]:
    """Get all tensors in an object that have a mark with a given value.

    Args:
        x: Input object.
        field_name: Name of the field to check.
        field_val: Value of the field that the tensor must have.

    Returns:
        List of tensors with the given mark.
    """
    input_tensors = get_vars_of_type_from_obj(x, torch.Tensor)
    tensors_with_mark = []
    for tensor in input_tensors:
        if getattr(tensor, field_name, None) == field_val:
            tensors_with_mark.append(tensor)
    return tensors_with_mark


def get_marks_from_tensor_list(tensor_list: List[torch.Tensor], field_name: str) -> List[Any]:
    """Gets a list of marks from a list of tensors.

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


def remove_list_duplicates(list_: List) -> List:
    """Given a list, remove any duplicates preserving order of first apppearance of each element.
    Args:
        list_: List to remove duplicates from.
    Returns:
        List with duplicates removed.
    """
    return list(OrderedDict.fromkeys(list_))


def get_tensor_memory_amount(t: torch.Tensor) -> int:
    """Returns the size of a tensor in bytes.

    Args:
        t: Tensor.

    Returns:
        Size of tensor in bytes.
    """
    return getsizeof(t.storage())


def human_readable_size(size: int, decimal_places: int = 1) -> str:
    """Utility function to convert a size in bytes to a human-readable format.

    Args:
        size: Number of bytes.
        decimal_places: Number of decimal places to use.

    Returns:
        String with human-readable size.
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB', 'PB']:
        if size < 1024.0 or unit == 'PB':
            break
        size /= 1024.0
    if unit == 'B':
        size = int(size)
    else:
        size = np.round(size, decimals=decimal_places)
    return f"{size} {unit}"


def text_num_split(s: str) -> Tuple[str, int]:
    """Utility function that takes in a string that begins with letters and ends in a number,
    and splits it into the letter part and the number part, returning both in a tuple.

    Args:
        s: String

    Returns:
        Tuple containing the beginning string and ending number.
    """
    s = s.strip()
    num = ''
    while s[-1].isdigit():
        num = s[-1] + num
        s = s[:-1]
    return s, int(num)


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
        return ''
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
                ranges.append(f'{start}-{end}')
            start = int_list[i]
            end = int_list[i]
    if start == end:
        ranges.append(str(start))
    else:
        ranges.append(f'{start}-{end}')
    return ','.join(ranges)
