"""Steps 5-7: Conditional branches, module annotation fixes, buffer layer fixes."""

from collections import defaultdict
from typing import TYPE_CHECKING, List, Set

import torch

from ..helper_funcs import identity
from ..data_classes.tensor_log import TensorLog

if TYPE_CHECKING:
    from ..data_classes.model_log import ModelLog


def _mark_conditional_branches(self):
    """Starting from any terminal boolean nodes, backtracks until it finds the beginning of any
    conditional branches.
    """
    terminal_bool_nodes = self.internally_terminated_bool_layers[:]

    nodes_seen = set()
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


def _fix_modules_for_internal_tensors(self):
    """
    Since internally initialized tensors don't automatically know what module they're in,
    this function infers this by tracing back from tensors that came from the input.
    """
    # Fetch nodes where internally initialized branches first meet a tensor computed from the input:
    node_stack = self._layers_where_internal_branches_merge_with_input[:]

    # Now go through the stack and work backwards up the internally initialized branch, fixing the
    # module containment labels as we go.

    nodes_seen = set()
    while len(node_stack) > 0:
        node_label = node_stack.pop()
        node = self[node_label]
        # Propagate modules for any parent nodes:
        for parent_label in node.parent_layers:
            parent_node = self[parent_label]
            if (not parent_node.has_input_ancestor) and (parent_label not in nodes_seen):
                _fix_modules_for_single_internal_tensor(
                    node, parent_node, "parent", node_stack, nodes_seen
                )

        # And for any internally generated child nodes:
        for child_label in node.child_layers:
            child_node = self[child_label]
            if any(
                [
                    node.has_input_ancestor,
                    child_node.has_input_ancestor,
                    child_label in nodes_seen,
                    child_node.is_output_layer,
                ]
            ):
                continue
            _fix_modules_for_single_internal_tensor(
                node, child_node, "child", node_stack, nodes_seen
            )

    # Now that the module containment is fixed, add this to the operation equivalence types.
    for layer in self:
        module_str = "_".join(
            [module_pass[0] for module_pass in layer.containing_modules_origin_nested]
        )
        layer.operation_equivalence_type += module_str


def _fix_modules_for_single_internal_tensor(
    starting_node: TensorLog,
    node_to_fix: TensorLog,
    node_type_to_fix: str,
    node_stack: List[str],
    nodes_seen: Set[str],
):
    """Helper function to fix the containing modules for a single internally generated tensor.
    The rule is, start from the child node, and apply in reverse any modules that were entered or exited.

    Args:
        starting_node: Source node that has the correct module information
        node_to_fix: Parent of the source node
        node_type_to_fix: either 'child' or 'parent'
        node_stack: Stack of nodes to consider
        nodes_seen: Nodes seen so far
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


def _fix_buffer_layers(self):
    """Connect the buffer parents, merge duplicate buffer nodes, and label buffer passes correctly.
    Buffers are duplicates if they happen in the same module, have the same value, and have the same parents.
    """
    buffer_counter = defaultdict(lambda: 1)
    buffer_hash_groups = defaultdict(list)

    for layer_label in self.buffer_layers:
        layer = self[layer_label]
        if layer.buffer_parent is not None:
            layer.parent_layers.append(layer.buffer_parent)
            self[layer.buffer_parent].child_layers.append(layer_label)
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

    # Now go through and merge any layers with the same hash and the same value.
    for _, buffers_orig in buffer_hash_groups.items():
        buffers = buffers_orig[1:]
        unique_buffers = buffers_orig[:1]
        for b, buffer_label in enumerate(buffers):
            for unique_buffer_label in unique_buffers:
                buffer = self[buffer_label]
                unique_buffer = self[unique_buffer_label]
                if (
                    (buffer.tensor_contents is not None)
                    and (unique_buffer.tensor_contents is not None)
                    and (torch.equal(buffer.tensor_contents, unique_buffer.tensor_contents))
                ):
                    _merge_buffer_entries(self, unique_buffer, buffer)
                    break
                unique_buffers.append(buffer_label)

    # And relabel the buffer passes.

    for layer_label in self.buffer_layers:
        layer = self[layer_label]
        buffer_address = layer.buffer_address
        layer.buffer_pass = buffer_counter[buffer_address]
        self.buffer_num_passes[buffer_address] = buffer_counter[buffer_address]
        buffer_counter[buffer_address] += 1


def _merge_buffer_entries(self, source_buffer: TensorLog, buffer_to_remove: TensorLog):
    """Merges two identical buffer layers."""
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

    for sibling_layer in buffer_to_remove.sibling_layers:
        if buffer_to_remove.tensor_label_raw in self[sibling_layer].sibling_layers:
            self[sibling_layer].sibling_layers.remove(buffer_to_remove.tensor_label_raw)
            self[sibling_layer].sibling_layers.append(source_buffer.tensor_label_raw)

    self._raw_tensor_labels_list.remove(buffer_to_remove.tensor_label_raw)
    self._raw_tensor_dict.pop(buffer_to_remove.tensor_label_raw)

    for layer in self:
        if buffer_to_remove.tensor_label_raw in layer.orig_ancestors:
            layer.orig_ancestors.remove(buffer_to_remove.tensor_label_raw)
            layer.orig_ancestors.add(source_buffer.tensor_label_raw)
        if buffer_to_remove.tensor_label_raw in layer.internally_initialized_ancestors:
            layer.internally_initialized_ancestors.remove(buffer_to_remove.tensor_label_raw)
            layer.internally_initialized_ancestors.add(source_buffer.tensor_label_raw)

    self._remove_log_entry(buffer_to_remove, remove_references=True)
