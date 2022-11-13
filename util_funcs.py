from collections import OrderedDict
from sys import getsizeof

import numpy as np
import random
import torch
from typing import Any, List, Optional, Tuple, Type
import copy
from IPython import get_ipython

from torch import nn

import os
import binascii

import string
import secrets


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


def pprint_tensor_record(history_dict, which_fields='all', show_tensor=True):
    """Debug sorta function to pretty print the tensor record.

    Args:
        history_dict: Dict with the history
        which_fields: which fields to show
        show_tensor: whether to show the actual tensor
    """
    fields_not_to_print = ['parent_params', 'creation_args', 'creation_kwargs', 'history_dict']
    tensor_log = history_dict['tensor_log']

    for key, val in history_dict.items():
        if key == 'tensor_log':
            continue
        print(f'{key}: {val}')
    for barcode, record in tensor_log.items():
        print(f"\n{barcode}")
        if which_fields == 'all':
            field_list = record.keys()
        else:
            field_list = which_fields

        for field in field_list:
            if field not in record:
                continue
            val = record[field]
            if field in fields_not_to_print:
                continue
            if (field == 'tensor_contents') and not show_tensor:
                continue
            if type(val) == torch.Tensor:
                val_nice = np.array(val.data).squeeze()
                print(f'{field}: \n{val_nice}')
            else:
                print(f'{field}: {val}')


def make_barcode() -> int:
    """Generates a random integer hash for a layer to use as internal label (invisible from user side).

    Returns:
        Random hash.
    """
    alphabet = string.ascii_letters + string.digits
    barcode = ''.join(secrets.choice(alphabet) for _ in range(6))
    return barcode


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
                              search_depth: int = 5) -> List[torch.Tensor]:
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


def obj_contains_tensors(obj: Any) -> bool:
    """Checks if an object contains any tensors.

    Args:
        obj: Object to search.

    Returns:
        True if object contains tensors, False otherwise.
    """
    return len(get_vars_of_type_from_obj(obj)) > 0


def unique_classes_in_list(list_: List[Any]) -> List[type]:
    """Returns a list of unique classes in a list.

    Args:
        l: List to search.

    Returns:
        List of unique classes in the list.
    """
    classes = []
    for item in list_:
        if type(item) not in classes:
            classes.append(type(item))
    return classes


def mark_tensors_in_obj(x: Any, field_name: str, field_val: Any):
    """Marks all tensors in the input saying they came as input.

    Args:
        x: Input to the model.
        field_name: Name of the field to add to the tensor.
        field_val: Value of the field to add to the tensor.

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
        setattr(tensor, 'xray_barcode', make_barcode())


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


def remove_list_duplicates(l: List) -> List:
    """Given a list, remove any duplicates preserving order of first apppearance of each element.
    Args:
        l: List to remove duplicates from.
    Returns:
        List with duplicates removed.
    """
    return list(OrderedDict.fromkeys(l))


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


def readable_tensor_size(t: torch.Tensor) -> str:
    """Returns the size of a tensor in human-readable format.

    Args:
        t: Tensor.

    Returns:
        Human-readable size of tensor.
    """
    return human_readable_size(get_tensor_memory_amount(t))


def do_lists_intersect(l1, l2) -> bool:
    """Utility function checking if two lists have any elements in common."""
    return len(list(set(l1) & set(l2))) > 0


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
