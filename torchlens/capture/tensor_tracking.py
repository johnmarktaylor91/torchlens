"""Functions for tracking tensor lineage, family relationships, and operation equivalence.

Handles backward hooks for gradient capture, parent-child-sibling-spouse linkage,
parameter pass tracking, and structural fingerprinting of operations for loop detection.

Key concepts:

**Family links** (parent/child/sibling/spouse):
    When a new tensor is created by a function, its input tensors become parents,
    co-parents become spouses, and children of the same parent become siblings.
    All links are bidirectional and updated immediately at creation time.

**Operation equivalence type** (``_get_operation_equivalence_type``):
    A structural fingerprint string that identifies operations as "the same layer"
    across loop iterations.  Used by loop detection to group operations into
    equivalence classes.  For parameterized ops, the fingerprint is based on the
    parameter barcodes + op type (e.g. ``"conv2d_abc123_def456"``).  For
    non-parameterized ops, it hashes non-tensor args, output index, and
    containing module.

**Backward hooks** (``_add_backward_hook``):
    Uses ``weakref.ref(ModelLog)`` to avoid preventing garbage collection of the
    ModelLog after the user is done with it.  The hook closure captures the weakref
    and the raw tensor label (a string, not the tensor itself).

**Parent arg position tracking** (``_locate_parent_tensors_in_args``):
    Records where each parent tensor appeared in the function's args/kwargs,
    supporting up to 2 levels of nesting (e.g., ``args[0]`` or ``args[1][2]``).
    Deeper nesting is not tracked.
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
    """Register a backward hook on ``t`` that captures its gradient into ModelLog.

    The hook closure captures a ``weakref`` to ModelLog (not a strong reference)
    so that the hook doesn't prevent GC of the ModelLog after the user drops it
    (GC-8).  The closure also captures ``tensor_label`` (a string) rather than
    the tensor itself, avoiding circular references.

    Only tensors that participate in autograd (have grad_fn or require_grad)
    get hooks — others would never receive gradients.

    Args:
        t: The tensor to hook.
        tensor_label: Raw tensor label (e.g. ``"conv2d_3_47_raw"``) used to
            look up the corresponding log entry when the gradient arrives.
    """
    # Weak reference prevents ModelLog → tensor → hook → ModelLog ref cycle.
    self_ref = weakref.ref(self)

    def log_grad_to_model_history(grad):
        model_log = self_ref()
        if model_log is not None:
            _log_tensor_grad(model_log, grad, tensor_label)

    if (t.grad_fn is not None) or t.requires_grad:
        t.register_hook(log_grad_to_model_history)


def _log_tensor_grad(self: "ModelLog", grad: torch.Tensor, tensor_label_raw: str) -> None:
    """Callback invoked during backward pass to save a tensor's gradient.

    Resolves the raw label to a final label, then saves the gradient on the
    layer entry.  If the layer is an output parent, its output-layer children
    also receive the gradient (since output layers are identity wrappers that
    share the same gradient).

    Args:
        grad: The gradient tensor from autograd.
        tensor_label_raw: Raw tensor label used to look up the final label.
    """
    self.has_saved_gradients = True
    tensor_label = self._raw_to_final_layer_labels[tensor_label_raw]
    layer_log_entry = self[tensor_label]
    layers_to_update = [tensor_label]
    # Output layers are identity wrappers; propagate gradient to them too.
    if layer_log_entry.is_output_parent:
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
    """Map each parent tensor to its position in the function's args/kwargs.

    Supports up to 2 levels of nesting:
      - Top-level: ``args[i]`` maps to key ``i``
      - Nested: ``args[i][j]`` maps to key ``(i, j)``
    Deeper nesting is not tracked (would require recursive search).

    This mapping is stored as ``parent_layer_arg_locs`` on the child's log
    entry, and is used by:
      - ``_get_parent_contents``: to retrieve pre-call parent values from arg_copies
      - Validation replay: to reconstruct the function call

    Returns:
        ``{"args": {pos: label, ...}, "kwargs": {key: label, ...}}``
    """
    tensor_all_arg_positions: Dict[str, Dict] = {"args": {}, "kwargs": {}}
    arg_struct_dict = {"args": args, "kwargs": kwargs}

    for parent_entry in parent_log_entries:
        for arg_type in ["args", "kwargs"]:
            arg_struct = arg_struct_dict[arg_type]
            _find_arg_positions_for_single_parent(
                parent_entry,
                arg_type,
                arg_struct,  # type: ignore[arg-type]
                tensor_all_arg_positions,
            )

    return tensor_all_arg_positions


def _find_arg_positions_for_single_parent(
    parent_entry: LayerPassLog,
    arg_type: str,
    arg_struct: Union[List, Tuple, Dict],
    tensor_all_arg_positions: Dict,
) -> None:
    """Locate a single parent tensor within args or kwargs (up to 2 nesting levels).

    Scans the top-level args/kwargs and one level of sub-containers (lists,
    tuples, dicts).  For top-level matches, the key is a scalar (int index or
    kwarg name).  For nested matches, the key is a tuple ``(outer_key, inner_key)``.

    Args:
        parent_entry: The parent tensor's log entry.
        arg_type: ``"args"`` or ``"kwargs"``.
        arg_struct: The actual args tuple or kwargs dict.
        tensor_all_arg_positions: Accumulator dict; mutated in place.
    """
    # Polymorphic iteration: enumerate for positional, .items() for keyword.
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
            # Second level of nesting (e.g., torch.cat([tensor_a, tensor_b])).
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
    """Update bidirectional family links for a newly created tensor.

    All four relationship types are updated symmetrically:
      - **Parent → Child**: new tensor added to each parent's ``child_layers``.
      - **Spouse ↔ Spouse**: all pairs of parents become spouses (they co-parent).
      - **Sibling ↔ Sibling**: existing children of each parent become siblings
        of the new tensor (and vice versa).

    Args:
        entry_to_update: The newly created LayerPassLog entry.
    """
    tensor_label = entry_to_update.tensor_label_raw
    parent_tensor_labels = entry_to_update.parent_layers

    # Parent → Child (bidirectional: child already knows its parents from fields_dict).
    for parent_tensor_label in parent_tensor_labels:
        parent_tensor = self[parent_tensor_label]
        if tensor_label not in parent_tensor.child_layers:
            parent_tensor.child_layers.append(tensor_label)
            parent_tensor.has_children = True

    # Spouse ↔ Spouse: co-parents are spouses of each other.
    for spouse1, spouse2 in it.combinations(parent_tensor_labels, 2):
        if spouse1 not in self[spouse2].spouse_layers:
            self[spouse2].spouse_layers.append(spouse1)
            self[spouse2].has_spouses = True
        if spouse2 not in self[spouse1].spouse_layers:
            self[spouse1].spouse_layers.append(spouse2)
            self[spouse1].has_spouses = True

    # Sibling ↔ Sibling: children of the same parent are siblings.
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
    """Assign persistent barcodes to parameters and track their pass number.

    On first encounter, each parameter gets a random barcode (``tl_param_barcode``)
    and pass number 1.  On subsequent encounters (same parameter used again in
    a later loop iteration), the pass number is incremented.

    The barcode is a random string that uniquely identifies the parameter tensor
    across the entire logging session.  It's used (together with layer_type) to
    build the ``operation_equivalence_type`` for parameterized operations, which
    is how loop detection recognizes "conv2d with weights W" as the same layer
    on pass 1 and pass 2.

    Args:
        arg_parameters: Parameter tensors found in the function's arguments.

    Returns:
        Dict mapping each parameter's barcode to its current pass number.
    """
    parent_param_passes = {}
    for param in arg_parameters:
        if not hasattr(param, "tl_param_barcode"):
            # First time seeing this parameter — assign a unique barcode.
            param_barcode = make_random_barcode()
            param.tl_param_barcode = param_barcode  # type: ignore[attr-defined]
            param.tl_pass_num = 1  # type: ignore[attr-defined]
        else:
            # Same parameter seen again (loop iteration) — increment pass.
            param_barcode = param.tl_param_barcode  # type: ignore[attr-defined]
            param.tl_pass_num += 1  # type: ignore[attr-defined]
        parent_param_passes[param_barcode] = param.tl_pass_num  # type: ignore[attr-defined]
    return parent_param_passes


def _make_raw_param_group_barcode(indiv_param_barcodes: List[str], layer_type: str) -> str:
    """Build an operation_equivalence_type string for a parameterized operation.

    Combines the layer type with sorted parameter barcodes to produce a
    canonical fingerprint.  Sorting ensures order-independence (e.g., weight
    and bias can appear in either order).

    The layer_type prefix is critical: different operations using the same
    parameters (e.g., ``isinf(weight)`` vs ``expand(weight)``) must NOT be
    grouped as the same layer.

    Example: ``"conv2d_abc123_def456"``

    Args:
        indiv_param_barcodes: Barcodes for each parameter tensor.
        layer_type: The normalized operation name.

    Returns:
        Canonical fingerprint string for this parameterized operation.
    """
    param_group_barcode = f"{layer_type}_{'_'.join(sorted(indiv_param_barcodes))}"
    return param_group_barcode


def _get_operation_equivalence_type(
    args: Tuple, kwargs: Dict, i: int, layer_type: str, fields_dict: Dict
) -> str:
    """Build an operation_equivalence_type string for a NON-parameterized operation.

    For ops that don't use parameters (e.g., ``relu``, ``cat``, ``add``), the
    fingerprint is built from:
      1. ``layer_type``: the normalized function name.
      2. ``arg_hash``: hash of non-tensor arguments (shapes, scalar values, etc.).
      3. ``outindex`` suffix: disambiguates outputs of multi-output functions.
      4. ``module`` suffix: disambiguates identical ops in different submodules.

    This fingerprint is used by loop detection.  Two operations with the same
    fingerprint are candidates for being "the same layer on different passes."

    Note: non-parameterized ops default to ``pass_num=1`` because without
    parameters to track reuse, there's no reliable way to count passes.

    Args:
        args: Positional arguments to the function call.
        kwargs: Keyword arguments to the function call.
        i: Index of this output tensor within a multi-output call.
        layer_type: The normalized operation name.
        fields_dict: Must contain ``is_part_of_iterable_output`` and
            ``containing_module_origin``.

    Returns:
        A string key identifying this operation's equivalence class.
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
    """Compute a structural hash of non-tensor arguments for equivalence fingerprinting.

    Tensor arguments are excluded (they define graph edges, not structural identity).
    Parameters are also excluded (they have their own barcode system).
    The hash preserves positional indices and kwarg keys to avoid collisions
    between e.g. ``f(a=1, b=2)`` and ``f(a=2, b=1)``.

    Returns:
        A short deterministic hash string, or ``"no_args"`` if no non-tensor
        arguments are present.
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
    """Compute a tensor's current module nesting by replaying entry/exit transitions.

    Each tensor records:
      - ``containing_modules_origin_nested``: the module stack at creation time.
      - ``module_entry_exit_thread_output``: a sequence of transitions like
        ``"+encoder.layer1:1"`` (entered) or ``"-encoder.layer1:1"`` (exited).

    This function starts from the origin stack and applies each transition to
    produce the current containing-module list.  Used by ``_get_input_module_info``
    to determine which module a new operation is inside.

    Args:
        layer_entry: Log entry whose module context to compute.

    Returns:
        Current containing-module stack as a list of module pass strings.
    """
    containing_modules = layer_entry.containing_modules_origin_nested[:]
    thread_modules = layer_entry.module_entry_exit_thread_output[:]
    for thread_module in thread_modules:
        if thread_module[0] == "+":
            containing_modules.append(thread_module[1:])
        elif (thread_module[0] == "-") and (thread_module[1:] in containing_modules):
            containing_modules.remove(thread_module[1:])
    return containing_modules
