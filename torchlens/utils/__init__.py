"""Focused utility modules: RNG, tensor ops, argument handling, introspection, collections, hashing, display."""

from .rng import (
    set_random_seed,
    log_current_rng_states,
    set_rng_from_saved_states,
    log_current_autocast_state,
    AutocastRestore,
    _AUTOCAST_DEVICES,
)
from .tensor_utils import (
    MAX_FLOATING_POINT_TOLERANCE,
    _cuda_available,
    _is_cuda_available,
    tensor_all_nan,
    tensor_nanequal,
    safe_to,
    get_tensor_memory_amount,
    safe_copy,
    print_override,
)
from .arg_handling import (
    _safe_copy_arg,
    safe_copy_args,
    safe_copy_kwargs,
    _model_expects_single_arg,
    normalize_input_args,
)
from .introspection import (
    _ATTR_SKIP_SET,
    get_vars_of_type_from_obj,
    get_attr_values_from_tensor_list,
    nested_getattr,
    nested_assign,
    iter_accessible_attributes,
    remove_attributes_with_prefix,
    _get_func_call_stack,
)
from .collections import (
    is_iterable,
    ensure_iterable,
    index_nested,
    remove_entry_from_list,
    assign_to_sequence_or_dict,
)
from .hashing import (
    make_random_barcode,
    make_short_barcode_from_input,
)
from .display import (
    identity,
    int_list_to_compact_str,
    human_readable_size,
    in_notebook,
    warn_parallel,
)

__all__ = [
    "AutocastRestore",
    "MAX_FLOATING_POINT_TOLERANCE",
    "_ATTR_SKIP_SET",
    "_AUTOCAST_DEVICES",
    "_cuda_available",
    "_get_func_call_stack",
    "_is_cuda_available",
    "_model_expects_single_arg",
    "_safe_copy_arg",
    "assign_to_sequence_or_dict",
    "ensure_iterable",
    "get_attr_values_from_tensor_list",
    "get_tensor_memory_amount",
    "get_vars_of_type_from_obj",
    "human_readable_size",
    "identity",
    "in_notebook",
    "index_nested",
    "int_list_to_compact_str",
    "is_iterable",
    "iter_accessible_attributes",
    "log_current_autocast_state",
    "log_current_rng_states",
    "make_random_barcode",
    "make_short_barcode_from_input",
    "nested_assign",
    "nested_getattr",
    "normalize_input_args",
    "print_override",
    "remove_attributes_with_prefix",
    "remove_entry_from_list",
    "safe_copy",
    "safe_copy_args",
    "safe_copy_kwargs",
    "safe_to",
    "set_random_seed",
    "set_rng_from_saved_states",
    "tensor_all_nan",
    "tensor_nanequal",
    "warn_parallel",
]
