"""Functions for tracking tensor lineage, family relationships, and operation equivalence.

Handles backward hooks for gradient capture, parent-child-sibling-spouse linkage,
parameter pass tracking, and structural fingerprinting of operations for loop detection.
"""

import itertools as it
import weakref
from typing import Any, Dict, List, Set, Tuple, TYPE_CHECKING, Union

import torch

from ..utils.hashing import make_random_barcode, make_short_barcode_from_input
from ..data_classes.layer_pass_log import LayerPassLog

if TYPE_CHECKING:
    from ..data_classes.model_log import ModelLog


def _add_backward_hook(self: "ModelLog", t: torch.Tensor, tensor_label: str) -> None:
    """Adds a backward hook to the tensor that saves the gradients to ModelLog if specified.

    Args:
        t: tensor
        tensor_label: the raw tensor label used to look up the tensor's log entry
    """

    # GC-8: Use weakref to avoid preventing ModelLog garbage collection
    self_ref = weakref.ref(self)

    def log_grad_to_model_history(grad):
        model_log = self_ref()
        if model_log is not None:
            _log_tensor_grad(model_log, grad, tensor_label)

    if (t.grad_fn is not None) or t.requires_grad:
        t.register_hook(log_grad_to_model_history)


def _log_tensor_grad(self: "ModelLog", grad: torch.Tensor, tensor_label_raw: str) -> None:
    """Logs the gradient for a tensor during a backward pass.

    Args:
        grad: the gradient
        tensor_label_raw: the raw tensor label
    """
    self.has_saved_gradients = True
    tensor_label = self._raw_to_final_layer_labels[tensor_label_raw]
    layer_log_entry = self[tensor_label]
    layers_to_update = [tensor_label]
    if layer_log_entry.is_output_parent:  # also update any linked outputs
        for child_layer in layer_log_entry.child_layers:
            if self[child_layer].is_output_layer:
                layers_to_update.append(child_layer)

    for layer_label in layers_to_update:
        layer = self[layer_label]
        if layer_label not in self._saved_gradients_set:
            self._saved_gradients_set.add(layer_label)
            self.layers_with_saved_gradients.append(layer_label)
        layer.log_tensor_grad(grad)


def _locate_parent_tensors_in_args(
    self: "ModelLog",
    parent_log_entries: List[LayerPassLog],
    args: Tuple[Any],
    kwargs: Dict[Any, Any],
) -> Dict:
    """Returns a dict specifying where in the function call each parent tensor was used.

    Args:
        parent_log_entries: List of parent LayerPassLog entries.
        args: Tuple of function positional args.
        kwargs: Dict of function keyword args.

    Returns:
        Dict with two sub-dicts: 'args' and 'kwargs', each mapping arg positions to parent
        tensor labels.
    """
    tensor_all_arg_positions: Dict[str, Dict] = {"args": {}, "kwargs": {}}
    arg_struct_dict = {"args": args, "kwargs": kwargs}

    for parent_entry in parent_log_entries:
        for arg_type in ["args", "kwargs"]:
            arg_struct = arg_struct_dict[arg_type]
            _find_arg_positions_for_single_parent(
                parent_entry,
                arg_type,
                arg_struct,
                tensor_all_arg_positions,  # type: ignore[arg-type]
            )

    return tensor_all_arg_positions


def _find_arg_positions_for_single_parent(
    parent_entry: LayerPassLog,
    arg_type: str,
    arg_struct: Union[List, Tuple, Dict],
    tensor_all_arg_positions: Dict,
) -> None:
    """Helper function that finds where a single parent tensor is used in either the args or kwargs of a function,
    and updates a dict that tracks this information.

    Args:
        parent_entry: Parent tensor
        arg_type: 'args' or 'kwargs'
        arg_struct: args or kwargs
        tensor_all_arg_positions: dict tracking where the tensors are used
    """
    iteration_strategies = {
        "args": enumerate,
        "kwargs": lambda x: x.items(),
        list: enumerate,
        tuple: enumerate,
        dict: lambda x: x.items(),
    }
    iterfunc = iteration_strategies[arg_type]

    for arg_key, arg in iterfunc(arg_struct):  # type: ignore[operator]
        if getattr(arg, "tl_tensor_label_raw", -1) == parent_entry.tensor_label_raw:
            tensor_all_arg_positions[arg_type][arg_key] = parent_entry.tensor_label_raw
        elif type(arg) in [list, tuple, dict]:
            iterfunc2 = iteration_strategies[type(arg)]
            for sub_arg_key, sub_arg in iterfunc2(arg):  # type: ignore[operator]
                if getattr(sub_arg, "tl_tensor_label_raw", -1) == parent_entry.tensor_label_raw:
                    tensor_all_arg_positions[arg_type][(arg_key, sub_arg_key)] = (
                        parent_entry.tensor_label_raw
                    )


def _get_ancestors_from_parents(
    parent_entries: List[LayerPassLog],
) -> Tuple[Set[str], Set[str]]:
    """Utility function to get the ancestors of a tensor based on those of its parent tensors.

    Args:
        parent_entries: list of parent entries

    Returns:
        List of input ancestors and internally initialized ancestors.
    """
    input_ancestors = set()
    internally_initialized_ancestors = set()

    for parent_entry in parent_entries:
        input_ancestors.update(parent_entry.input_ancestors)
        internally_initialized_ancestors.update(parent_entry.internally_initialized_ancestors)
    return input_ancestors, internally_initialized_ancestors


def _update_tensor_family_links(self: "ModelLog", entry_to_update: LayerPassLog) -> None:
    """For a given tensor, updates family information for its links to parents, children, siblings, and
    spouses, in both directions (i.e., mutually adding the labels for each family pair).

    Args:
        entry_to_update: dict of information about the LayerPassLog to be created
    """
    tensor_label = entry_to_update.tensor_label_raw
    parent_tensor_labels = entry_to_update.parent_layers

    # Add the tensor as child to its parents

    for parent_tensor_label in parent_tensor_labels:
        parent_tensor = self[parent_tensor_label]
        if tensor_label not in parent_tensor.child_layers:
            parent_tensor.child_layers.append(tensor_label)
            parent_tensor.has_children = True

    # Set the parents of the tensor as spouses to each other

    for spouse1, spouse2 in it.combinations(parent_tensor_labels, 2):
        if spouse1 not in self[spouse2].spouse_layers:
            self[spouse2].spouse_layers.append(spouse1)
            self[spouse2].has_spouses = True
        if spouse2 not in self[spouse1].spouse_layers:
            self[spouse1].spouse_layers.append(spouse2)
            self[spouse1].has_spouses = True

    # Set the children of its parents as siblings to each other.

    for parent_tensor_label in parent_tensor_labels:
        _add_sibling_labels_for_new_tensor(self, entry_to_update, self[parent_tensor_label])


def _add_sibling_labels_for_new_tensor(
    self: "ModelLog", entry_to_update: LayerPassLog, parent_tensor: LayerPassLog
) -> None:
    """Given a tensor and specified parent tensor, adds sibling labels to that tensor, and
    adds itself as a sibling to all existing children.

    Args:
        entry_to_update: the new tensor
        parent_tensor: the parent tensor
    """
    new_tensor_label = entry_to_update.tensor_label_raw
    for sibling_tensor_label in parent_tensor.child_layers:
        if sibling_tensor_label == new_tensor_label:
            continue
        sibling_tensor = self[sibling_tensor_label]
        if new_tensor_label not in sibling_tensor.sibling_layers:
            sibling_tensor.sibling_layers.append(new_tensor_label)
            sibling_tensor.has_siblings = True
        if sibling_tensor_label not in entry_to_update.sibling_layers:
            entry_to_update.sibling_layers.append(sibling_tensor_label)
            entry_to_update.has_siblings = True


def _process_parent_param_passes(
    arg_parameters: List[torch.nn.Parameter],
) -> Dict[str, int]:
    """Utility function to mark the parameters with barcodes, and log which pass they're on.

    Args:
        arg_parameters: List of arg parameters

    Returns:
        Dict mapping each parameter's barcode to its current pass number.
    """
    parent_param_passes = {}
    for param in arg_parameters:
        if not hasattr(param, "tl_param_barcode"):
            param_barcode = make_random_barcode()
            param.tl_param_barcode = param_barcode  # type: ignore[attr-defined]
            param.tl_pass_num = 1  # type: ignore[attr-defined]
        else:
            param_barcode = param.tl_param_barcode  # type: ignore[attr-defined]
            param.tl_pass_num += 1  # type: ignore[attr-defined]
        parent_param_passes[param_barcode] = param.tl_pass_num  # type: ignore[attr-defined]
    return parent_param_passes


def _make_raw_param_group_barcode(indiv_param_barcodes: List[str], layer_type: str) -> str:
    """Given list of param barcodes and layer type, returns the raw barcode for the
    param_group; e.g., conv2d_abcdef_uvwxyz

    Args:
        indiv_param_barcodes: List of barcodes for each individual parameter tensor
        layer_type: The layer type.

    Returns:
        Raw barcode for the param group
    """
    param_group_barcode = f"{layer_type}_{'_'.join(sorted(indiv_param_barcodes))}"
    return param_group_barcode


def _get_operation_equivalence_type(
    args: Tuple, kwargs: Dict, i: int, layer_type: str, fields_dict: Dict
) -> str:
    """Builds a string key that uniquely identifies an operation's structural equivalence class.

    Two invocations of the same function with the same non-tensor arguments, output index,
    and containing module are considered equivalent (same layer across passes).

    Args:
        args: Positional arguments to the function call.
        kwargs: Keyword arguments to the function call.
        i: Index of this output tensor within a multi-output call (0 for single outputs).
        layer_type: The operation name (e.g. 'conv2d', 'relu').
        fields_dict: Partial fields dict for the tensor being logged; must contain
            'is_part_of_iterable_output' and 'containing_module_origin'.

    Returns:
        A string key uniquely identifying this operation's equivalence class.
    """
    arg_hash = _get_hash_from_args(args, kwargs)
    operation_equivalence_type = f"{layer_type}_{arg_hash}"
    if fields_dict["is_part_of_iterable_output"]:
        operation_equivalence_type += f"_outindex{i}"
    if fields_dict["containing_module_origin"] is not None:
        module_str = fields_dict["containing_module_origin"][0]
        operation_equivalence_type += f"_module{module_str}"
    return operation_equivalence_type


def _get_hash_from_args(args, kwargs) -> str:
    """Get a hash from the args and kwargs of a function call, excluding any tracked tensors.

    Preserves positional arg indices, kwarg names, and dict keys to avoid collisions.

    Args:
        args: Positional arguments to hash.
        kwargs: Keyword arguments to hash.

    Returns:
        A short hash string, or 'no_args' if no non-tensor arguments are present.
    """
    args_to_hash: List[Any] = []
    for a, arg in enumerate(args):
        _append_arg_hash(arg, f"pos{a}", args_to_hash)
    for key, arg in kwargs.items():
        _append_arg_hash(arg, f"kw_{key}", args_to_hash)

    if len(args_to_hash) == 0:
        return "no_args"
    return make_short_barcode_from_input(args_to_hash)


def _append_arg_hash(arg, prefix: str, args_to_hash: list, _depth: int = 0) -> None:
    """Append structural fingerprint tokens for a single argument to the accumulator list.

    Builds an ``operation_equivalence_type`` -- a structural fingerprint of the operation's
    argument types and shapes (not a content hash). This fingerprint is used by loop
    detection to identify operations that are structurally identical across passes.

    For tensors, only shape and dtype are recorded (not values). Containers (dicts, lists,
    tuples, sets) are recursed into with depth-limited traversal. Parameters are excluded.

    Args:
        arg: The argument value to fingerprint.
        prefix: String prefix encoding the argument's position/key path.
        args_to_hash: Accumulator list that fingerprint tokens are appended to.
        _depth: Recursion depth guard; stops at 10 to prevent infinite recursion.
    """
    if _depth > 10:
        args_to_hash.append(f"{prefix}_deep")
        return
    if isinstance(arg, torch.nn.Parameter):
        pass  # exclude parameters from hash — must check before Tensor (Parameter is a subclass)
    elif isinstance(arg, torch.Tensor):
        # Use shape/dtype only — formatting a tensor can trigger wrapped
        # methods (item, __format__) which re-enter logging and cause
        # infinite recursion.
        args_to_hash.append(f"{prefix}_tensor{arg.shape}")
    elif isinstance(arg, dict):
        for k, v in arg.items():
            _append_arg_hash(v, f"{prefix}_dk{k}", args_to_hash, _depth + 1)
    elif isinstance(arg, (list, tuple, set)):
        for i, elem in enumerate(arg):
            _append_arg_hash(elem, f"{prefix}_i{i}", args_to_hash, _depth + 1)
    else:
        args_to_hash.append(f"{prefix}_{arg}")


def _update_tensor_containing_modules(layer_entry: LayerPassLog) -> List[str]:
    """Utility function that updates the containing modules of a Tensor by starting from the containing modules
    as of the last function call, then looks at the sequence of module transitions (in or out of a module) as of
    the last module it saw, and updates accordingly.

    Args:
        layer_entry: Log entry of tensor to check

    Returns:
        List of the updated containing modules.
    """
    containing_modules = layer_entry.containing_modules_origin_nested[:]
    thread_modules = layer_entry.module_entry_exit_thread_output[:]
    for thread_module in thread_modules:
        if thread_module[0] == "+":
            containing_modules.append(thread_module[1:])
        elif (thread_module[0] == "-") and (thread_module[1:] in containing_modules):
            containing_modules.remove(thread_module[1:])
    return containing_modules
