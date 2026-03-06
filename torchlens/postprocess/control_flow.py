"""Steps 5-7: Conditional branches, module annotation fixes, buffer layer fixes.

Step 5 (_mark_conditional_branches): Starting from terminal boolean tensors,
    backtracks to find where conditional branches diverge from the main graph.
Step 6 (_fix_modules_for_internal_tensors): Internally-generated tensors
    (constants, arange, etc.) don't know their containing module. This infers
    module containment by tracing backward from known input-descendant tensors.
    Also appends module path suffixes to operation_equivalence_type — this is
    INTENTIONAL and affects loop detection (Step 8), ensuring operations in
    different modules are never grouped as the same layer.
Step 7 (_fix_buffer_layers): Connects buffer parents, deduplicates identical
    buffers (same module, same value, same parent), and assigns buffer pass numbers.
"""

from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Set, Tuple

import torch

from ..utils.display import identity
from ..data_classes.layer_pass_log import LayerPassLog

if TYPE_CHECKING:
    from ..data_classes.model_log import ModelLog


def _mark_conditional_branches(self) -> None:
    """Step 5: Detect conditional branches by backtracking from terminal booleans.

    Conditional branches in PyTorch models (e.g., ``if x > 0:``) produce boolean
    tensors that are consumed by Python control flow but never reach an output.
    These boolean tensors are identified as "internally terminated bool layers"
    during Step 3 (orphan removal).

    Starting from these terminal booleans, this function floods backward and
    forward through parent_layers and child_layers. When it encounters a node
    that IS an output ancestor (is_output_ancestor=True), that node is the
    "branch start" — the point where the conditional branch diverges from the
    main computation graph. All non-output-ancestor nodes traversed are marked
    as ``in_cond_branch=True``.
    """
    terminal_bool_nodes = self.internally_terminated_bool_layers[:]

    nodes_seen: Set[str] = set()
    node_stack = terminal_bool_nodes.copy()
    while len(node_stack) > 0:
        node_label = node_stack.pop()
        node = self[node_label]
        if node_label in nodes_seen:
            continue
        for next_tensor_label in node.parent_layers + node.child_layers:
            next_node = self[next_tensor_label]
            if next_node.is_output_ancestor:  # we found the beginning of a conditional branch
                next_node.cond_branch_start_children.append(node_label)
                next_node.in_cond_branch = False
                nodes_seen.add(next_tensor_label)
                self.conditional_branch_edges.append((next_tensor_label, node_label))
            else:
                if next_tensor_label in nodes_seen:
                    continue
                next_node.in_cond_branch = True
                node_stack.append(next_tensor_label)

        nodes_seen.add(node_label)


def _fix_modules_for_internal_tensors(self) -> None:
    """Step 6: Infer module containment for internally-generated tensors.

    Internally-initialized tensors (constants, torch.arange results, etc.) are
    created without any module context. This function starts from nodes where
    internal branches merge with input-descendant tensors and propagates module
    containment backward (to parents) and forward (to non-input-descendant
    children) using the module entry/exit thread metadata recorded during the
    forward pass.

    After fixing module containment, appends the module path as a suffix to
    each tensor's ``operation_equivalence_type``. This is INTENTIONAL and
    critical for Step 8 (loop detection): it ensures that operations with
    identical function signatures but in different modules (e.g., relu in
    linear1 vs relu in linear2) are never grouped as the same layer.
    """
    # Start from nodes where internally-initialized branches first merge with
    # an input-descendant tensor — these nodes have known module containment
    # that can be propagated backward to their internal ancestors.
    node_stack = self._layers_where_internal_branches_merge_with_input[:]

    nodes_seen: Set[str] = set()
    while len(node_stack) > 0:
        node_label = node_stack.pop()
        node = self[node_label]
        # Propagate modules backward to internal-only parent nodes:
        for parent_label in node.parent_layers:
            parent_node = self[parent_label]
            if (not parent_node.has_input_ancestor) and (parent_label not in nodes_seen):
                _fix_modules_for_single_internal_tensor(
                    node, parent_node, "parent", node_stack, nodes_seen
                )

        # Propagate forward to internally-generated child nodes:
        for child_label in node.child_layers:
            child_node = self[child_label]
            if (
                node.has_input_ancestor
                or child_node.has_input_ancestor
                or child_label in nodes_seen
                or child_node.is_output_layer
            ):
                continue
            _fix_modules_for_single_internal_tensor(
                node, child_node, "child", node_stack, nodes_seen
            )

    # Append module path suffix to operation_equivalence_type for ALL tensors.
    # This ensures loop detection (Step 8) treats same-function operations in
    # different modules as distinct equivalence types.
    for layer in self:
        module_str = "_".join(
            [module_pass[0] for module_pass in layer.containing_modules_origin_nested]
        )
        layer.operation_equivalence_type += module_str


def _fix_modules_for_single_internal_tensor(
    starting_node: LayerPassLog,
    node_to_fix: LayerPassLog,
    node_type_to_fix: str,
    node_stack: List[str],
    nodes_seen: Set[str],
) -> None:
    """Fix the containing modules for a single internally-generated tensor.

    Copies the module containment from ``starting_node`` (which has known module
    info) to ``node_to_fix``, then adjusts by replaying the module entry/exit
    thread in reverse. For a parent, module entries (+) on the child mean we
    should REMOVE that module from the parent's containment (going backward
    undoes the entry); module exits (-) mean we should ADD it. For a child,
    the logic is direct (entries add, exits remove).

    Args:
        starting_node: Source node with correct module containment.
        node_to_fix: The internally-generated tensor to fix.
        node_type_to_fix: 'parent' (fix going backward) or 'child' (fix going forward).
        node_stack: Stack to push node_to_fix onto for further propagation.
        nodes_seen: Set of already-processed nodes.
    """
    node_to_fix_label = node_to_fix.tensor_label_raw
    node_to_fix.containing_modules_origin_nested = (
        starting_node.containing_modules_origin_nested.copy()
    )
    if node_type_to_fix == "parent":
        thread_modules = starting_node.module_entry_exit_threads_inputs[
            node_to_fix.tensor_label_raw
        ]
        step_val = -1
    elif node_type_to_fix == "child":
        thread_modules = node_to_fix.module_entry_exit_threads_inputs[
            starting_node.tensor_label_raw
        ]
        step_val = 1
    else:
        raise ValueError("node_type_to_fix must be 'parent' or 'child'")

    for enter_or_exit, module_address, module_pass in thread_modules[::step_val]:
        module_pass_label = (module_address, module_pass)
        if node_type_to_fix == "parent":
            if (enter_or_exit == "+") and (
                module_pass_label in node_to_fix.containing_modules_origin_nested
            ):
                node_to_fix.containing_modules_origin_nested.remove(module_pass_label)
            elif enter_or_exit == "-":
                node_to_fix.containing_modules_origin_nested.append(module_pass_label)
        elif node_type_to_fix == "child":
            if enter_or_exit == "+":
                node_to_fix.containing_modules_origin_nested.append(module_pass_label)
            elif enter_or_exit == "-":
                if module_pass_label in node_to_fix.containing_modules_origin_nested:
                    node_to_fix.containing_modules_origin_nested.remove(module_pass_label)
    node_stack.append(node_to_fix_label)
    nodes_seen.add(node_to_fix_label)


def _fix_buffer_layers(self) -> None:
    """Step 7: Connect buffer parents, merge duplicates, and assign pass numbers.

    Buffer tensors (nn.Module registered buffers) are logged as source tensors
    during the forward pass but may lack proper parent connections. This function:

    1. Connects each buffer to its buffer_parent (the tensor that produced the
       buffer's value), updating parent/child links and ancestry.
    2. Deduplicates buffers: buffers with the same containing module, same parent,
       same buffer_address, AND same tensor value are merged into a single node.
       The dedup hash is (containing_modules + buffer_parent + buffer_address).
    3. Assigns sequential buffer_pass numbers per buffer_address.

    Note: Buffer sibling_layers are always empty — the sibling iteration in
    _merge_buffer_entries is effectively dead code for buffers (#2).
    """
    buffer_counter: Dict[str, int] = defaultdict(lambda: 1)
    buffer_hash_groups: Dict[str, List[str]] = defaultdict(list)

    for layer_label in self.buffer_layers:
        layer = self[layer_label]
        if layer.buffer_parent is not None:
            layer.parent_layers.append(layer.buffer_parent)
            layer.has_parents = True
            self[layer.buffer_parent].child_layers.append(layer_label)
            self[layer.buffer_parent].has_children = True
            layer.func_applied = identity
            layer.func_applied_name = "identity"
            layer.has_input_ancestor = True
            layer.input_ancestors.update(self[layer.buffer_parent].input_ancestors)
            layer.orig_ancestors.remove(layer.tensor_label_raw)
            layer.orig_ancestors.update(self[layer.buffer_parent].orig_ancestors)
            layer.parent_layer_arg_locs["args"][0] = layer.buffer_parent
            if (self[layer.buffer_parent].tensor_contents is not None) and (
                layer.creation_args is not None
            ):
                layer.creation_args.append(
                    self[layer.buffer_parent].tensor_contents.detach().clone()
                )

        buffer_hash = (
            str(layer.containing_modules_origin_nested)
            + str(layer.buffer_parent)
            + layer.buffer_address
        )
        buffer_hash_groups[buffer_hash].append(layer_label)

    # Merge buffers with the same hash AND the same tensor value.
    # Buffers sharing the same hash but different values are kept as separate
    # unique buffers (the for/else clause appends unmatched buffers to unique_buffers).
    for _, buffers_orig in buffer_hash_groups.items():
        buffers = buffers_orig[1:]
        unique_buffers = buffers_orig[:1]
        for b, buffer_label in enumerate(buffers):
            buffer = self[buffer_label]
            for unique_buffer_label in unique_buffers:
                unique_buffer = self[unique_buffer_label]
                if (
                    (buffer.tensor_contents is not None)
                    and (unique_buffer.tensor_contents is not None)
                    and (torch.equal(buffer.tensor_contents, unique_buffer.tensor_contents))
                ):
                    _merge_buffer_entries(self, unique_buffer, buffer)
                    break
            else:
                unique_buffers.append(buffer_label)

    # And relabel the buffer passes.

    for layer_label in self.buffer_layers:
        layer = self[layer_label]
        buffer_address = layer.buffer_address
        layer.buffer_pass = buffer_counter[buffer_address]
        self.buffer_num_passes[buffer_address] = buffer_counter[buffer_address]
        buffer_counter[buffer_address] += 1


def _merge_buffer_entries(
    self, source_buffer: LayerPassLog, buffer_to_remove: LayerPassLog
) -> None:
    """Merge a duplicate buffer into a source buffer, rewiring all edges.

    Transfers all child and parent connections from ``buffer_to_remove`` to
    ``source_buffer``, updates parent_layer_arg_locs in children to point to
    the source buffer, fixes internally_initialized_parents/ancestors references
    across the graph, and removes the duplicate from the layer dict.
    """
    for child_layer in buffer_to_remove.child_layers:
        if child_layer not in source_buffer.child_layers:
            source_buffer.child_layers.append(child_layer)
        self[child_layer].parent_layers.remove(buffer_to_remove.tensor_label_raw)
        self[child_layer].parent_layers.append(source_buffer.tensor_label_raw)
        if buffer_to_remove.tensor_label_raw in self[child_layer].internally_initialized_parents:
            self[child_layer].internally_initialized_parents.remove(
                buffer_to_remove.tensor_label_raw
            )
            self[child_layer].internally_initialized_parents.append(source_buffer.tensor_label_raw)

        for arg_type in ["args", "kwargs"]:
            for arg_label, arg_val in self[child_layer].parent_layer_arg_locs[arg_type].items():
                if arg_val == buffer_to_remove.tensor_label_raw:
                    self[child_layer].parent_layer_arg_locs[arg_type][arg_label] = (
                        source_buffer.tensor_label_raw
                    )

    for parent_layer in buffer_to_remove.parent_layers:
        if parent_layer not in source_buffer.parent_layers:
            source_buffer.parent_layers.append(parent_layer)
        self[parent_layer].child_layers.remove(buffer_to_remove.tensor_label_raw)
        self[parent_layer].child_layers.append(source_buffer.tensor_label_raw)

    for parent_layer in buffer_to_remove.internally_initialized_parents:
        if parent_layer not in source_buffer.internally_initialized_parents:
            source_buffer.internally_initialized_parents.append(parent_layer)

    if buffer_to_remove.tensor_label_raw in source_buffer.spouse_layers:
        source_buffer.spouse_layers.remove(buffer_to_remove.tensor_label_raw)

    if buffer_to_remove.tensor_label_raw in source_buffer.sibling_layers:
        source_buffer.sibling_layers.remove(buffer_to_remove.tensor_label_raw)

    for spouse_layer in buffer_to_remove.spouse_layers:
        if buffer_to_remove.tensor_label_raw in self[spouse_layer].spouse_layers:
            self[spouse_layer].spouse_layers.remove(buffer_to_remove.tensor_label_raw)
            self[spouse_layer].spouse_layers.append(source_buffer.tensor_label_raw)

    # Note: sibling_layers iteration removed — buffers always have empty sibling_layers (#2)

    self._raw_layer_labels_list.remove(buffer_to_remove.tensor_label_raw)
    self._raw_layer_dict.pop(buffer_to_remove.tensor_label_raw)

    for layer in self:
        if buffer_to_remove.tensor_label_raw in layer.orig_ancestors:
            layer.orig_ancestors.remove(buffer_to_remove.tensor_label_raw)
            layer.orig_ancestors.add(source_buffer.tensor_label_raw)
        if buffer_to_remove.tensor_label_raw in layer.internally_initialized_ancestors:
            layer.internally_initialized_ancestors.remove(buffer_to_remove.tensor_label_raw)
            layer.internally_initialized_ancestors.add(source_buffer.tensor_label_raw)

    self._remove_log_entry(buffer_to_remove, remove_references=True)
