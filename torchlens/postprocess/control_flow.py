"""Steps 5-7: Conditional branches, module annotation fixes, buffer layer fixes.

Step 5 (_mark_conditional_branches): Starting from terminal boolean tensors,
    backtracks through parent_layers only to find where conditional branches
    diverge from the main graph. Optionally detects THEN branches via AST
    analysis when save_source_context=True.
Step 6 (_fix_modules_for_internal_tensors): Internally-generated tensors
    (constants, arange, etc.) don't know their containing module. This infers
    module containment by tracing backward from known input-descendant tensors.
    Also appends module path suffixes to operation_equivalence_type — this is
    INTENTIONAL and affects loop detection (Step 8), ensuring operations in
    different modules are never grouped as the same layer.
Step 7 (_fix_buffer_layers): Connects buffer parents, deduplicates identical
    buffers (same module, same value, same parent), and assigns buffer pass numbers.
"""

import ast
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

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

    Starting from these terminal booleans, this function floods **backward only**
    through parent_layers. When it encounters a node that IS an output ancestor
    (is_output_ancestor=True), that node is the "branch start" — the point where
    the conditional branch diverges from the main computation graph. All
    non-output-ancestor nodes traversed are marked as ``in_cond_branch=True``.

    Bug #88 fix: The original algorithm flooded bidirectionally (parents + children),
    which caused non-conditional children of output-ancestor nodes to be falsely
    marked as in_cond_branch. The fix restricts flooding to parent_layers only.

    After IF detection, if save_source_context is enabled, THEN branches are
    detected via AST analysis of the source file containing the ``if`` statement.
    """
    terminal_bool_nodes = self.internally_terminated_bool_layers[:]

    nodes_seen: Set[str] = set()
    node_stack = terminal_bool_nodes.copy()
    while len(node_stack) > 0:
        node_label = node_stack.pop()
        node = self[node_label]
        if node_label in nodes_seen:
            continue
        for next_tensor_label in node.parent_layers:  # backward only (Bug #88 fix)
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

    # THEN branch detection (requires source context for AST analysis)
    if self.save_source_context:
        _detect_then_branches(self, terminal_bool_nodes)


def _detect_then_branches(self, terminal_bool_nodes: List[str]) -> None:
    """Detect THEN branches by matching branch-start children to AST ``if`` body ranges.

    For each branch-start node (node with cond_branch_start_children), finds a
    terminal boolean in its condition chain, uses that boolean's func_call_stack
    to locate the ``if`` statement in source, parses the AST to get the if-body
    line range, then checks each output-ancestor child of the branch-start to see
    if its source line falls within that range.

    Args:
        terminal_bool_nodes: Labels of terminal boolean nodes (the condition tensors).
    """
    ast_cache: Dict[str, Optional[ast.Module]] = {}
    terminal_bool_set = set(terminal_bool_nodes)
    # Track which branch starts have a confirmed ast.If (i.e., the bool IS
    # used for control flow). Branch starts without an ast.If match have their
    # IF markings cleared in post-validation.
    branch_starts_with_ast_if: Set[str] = set()

    for start_label in self._raw_layer_labels_list:
        start_node = self[start_label]
        if not start_node.cond_branch_start_children:
            continue

        # Find a terminal boolean reachable from this branch start's condition chain.
        # DFS through the IF children (cond_branch_start_children) and their
        # non-output-ancestor children to find a terminal bool with a call stack.
        terminal_bool = _find_terminal_bool_in_chain(
            self, start_node.cond_branch_start_children, terminal_bool_set
        )
        if terminal_bool is None:
            continue

        # Get the terminal boolean's source location
        if not terminal_bool.func_call_stack:
            continue
        user_frame = terminal_bool.func_call_stack[0]
        file_b = user_frame.file
        line_b = user_frame.line_number

        # Parse AST for this file (cached)
        if file_b not in ast_cache:
            try:
                with open(file_b, "r") as f:
                    source = f.read()
                ast_cache[file_b] = ast.parse(source)
            except Exception:
                ast_cache[file_b] = None

        tree = ast_cache[file_b]
        if tree is None:
            continue

        # Find the ast.If node at or near line_b
        if_node = _find_if_node_near_line(tree, line_b)
        if if_node is None:
            continue

        # The bool IS used in an actual if-statement — keep IF markings even
        # if no THEN children found (e.g., else branch executed instead).
        branch_starts_with_ast_if.add(start_label)

        # Get the if-body line range
        body_start = if_node.body[0].lineno
        body_end = _get_end_lineno(if_node.body[-1])

        # Check each output-ancestor child that is NOT already an IF child
        if_children_set = set(start_node.cond_branch_start_children)
        for child_label in start_node.child_layers:
            if child_label in if_children_set:
                continue
            child_node = self[child_label]
            if not child_node.is_output_ancestor:
                continue
            # Check if this child's source line falls within the if-body
            if not child_node.func_call_stack:
                continue
            child_frame = _find_frame_in_file(child_node.func_call_stack, file_b)
            if child_frame is None:
                continue
            if body_start <= child_frame.line_number <= body_end:
                start_node.cond_branch_then_children.append(child_label)
                self.conditional_then_edges.append((start_label, child_label))

    # Post-validation: clear IF markings for branch-starts where no ast.If was
    # found (the bool was computed but not used for control flow). Branch-starts
    # WITH a confirmed ast.If keep their IF markings even if THEN is empty
    # (e.g., else branch executed — ELSE detection is a future TODO).
    for label in self._raw_layer_labels_list:
        node = self[label]
        if not node.cond_branch_start_children:
            continue
        if label in branch_starts_with_ast_if:
            continue  # confirmed if-statement — keep IF markings
        # No ast.If found — clear false IF markings
        seen: Set[str] = set()
        stack = list(node.cond_branch_start_children)
        while stack:
            cond_label = stack.pop()
            if cond_label in seen:
                continue
            seen.add(cond_label)
            cond_node = self[cond_label]
            cond_node.in_cond_branch = False
            for parent_label in cond_node.parent_layers:
                parent = self[parent_label]
                if parent.in_cond_branch:
                    stack.append(parent_label)
        # Clear branch-start fields
        self.conditional_branch_edges = [
            edge for edge in self.conditional_branch_edges if edge[0] != label
        ]
        node.cond_branch_start_children.clear()


def _find_if_node_near_line(tree: ast.Module, target_line: int) -> Optional[ast.If]:
    """Find the ast.If node whose test covers the target line number.

    Walks the entire AST looking for If nodes where the test expression's line
    range contains the target line, or the If node itself is on the target line.
    Returns the most specific (deepest nested) match.
    """
    best: Optional[ast.If] = None
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            test_start = node.test.lineno
            test_end = getattr(node.test, "end_lineno", test_start)
            # The if statement itself or its test expression covers the target line
            if test_start <= target_line <= test_end or node.lineno == target_line:
                # Prefer deeper (later-found in walk order, or more specific)
                if best is None or node.lineno >= best.lineno:
                    best = node
    return best


def _get_end_lineno(node: ast.AST) -> int:
    """Get the end line number of an AST node, handling nested blocks."""
    end = getattr(node, "end_lineno", None)
    if end is not None:
        return end
    # Fallback: walk children to find the maximum line number
    max_line = getattr(node, "lineno", 0)
    for child in ast.walk(node):
        child_end = getattr(child, "end_lineno", getattr(child, "lineno", 0))
        if child_end > max_line:
            max_line = child_end
    return max_line


def _find_terminal_bool_in_chain(
    self, if_children: List[str], terminal_bool_set: Set[str]
) -> Optional[LayerPassLog]:
    """Find a terminal boolean node reachable from the given IF children.

    DFS through IF children and their non-output-ancestor children (the condition
    chain) to find a node in ``terminal_bool_set`` that has a func_call_stack.
    """
    seen: Set[str] = set()
    stack = list(if_children)
    while stack:
        label = stack.pop()
        if label in seen:
            continue
        seen.add(label)
        node = self[label]
        if label in terminal_bool_set and node.func_call_stack:
            return node
        # Follow children that are part of the condition chain (not output ancestors)
        for child_label in node.child_layers:
            child = self[child_label]
            if not child.is_output_ancestor and child_label not in seen:
                stack.append(child_label)
    return None


def _find_frame_in_file(func_call_stack, target_file: str):
    """Find the first FuncCallLocation in the stack that matches the target file."""
    for frame in func_call_stack:
        if frame.file == target_file:
            return frame
    return None


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
    _module_str_cache = {}
    for layer in self:
        cm_key = tuple(layer.containing_modules)
        if cm_key not in _module_str_cache:
            _module_str_cache[cm_key] = "_".join(
                [module_pass[0] for module_pass in layer.containing_modules]
            )
        layer.operation_equivalence_type += _module_str_cache[cm_key]


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
    node_to_fix.containing_modules = starting_node.containing_modules.copy()
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
            if (enter_or_exit == "+") and (module_pass_label in node_to_fix.containing_modules):
                node_to_fix.containing_modules.remove(module_pass_label)
            elif enter_or_exit == "-":
                node_to_fix.containing_modules.append(module_pass_label)
        elif node_type_to_fix == "child":
            if enter_or_exit == "+":
                node_to_fix.containing_modules.append(module_pass_label)
            elif enter_or_exit == "-":
                if module_pass_label in node_to_fix.containing_modules:
                    node_to_fix.containing_modules.remove(module_pass_label)
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
            self[layer.buffer_parent].child_layers.append(layer_label)
            self[layer.buffer_parent].has_children = True
            layer.func_applied = identity
            layer.func_name = "identity"
            layer.has_input_ancestor = True
            layer.input_ancestors.update(self[layer.buffer_parent].input_ancestors)
            layer.root_ancestors.remove(layer.tensor_label_raw)
            layer.root_ancestors.update(self[layer.buffer_parent].root_ancestors)
            layer.parent_layer_arg_locs["args"][0] = layer.buffer_parent
            if (self[layer.buffer_parent].activation is not None) and (
                layer.captured_args is not None
            ):
                layer.captured_args.append(self[layer.buffer_parent].activation.detach().clone())

        buffer_hash = (
            str(layer.containing_modules) + str(layer.buffer_parent) + layer.buffer_address
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
                    (buffer.activation is not None)
                    and (unique_buffer.activation is not None)
                    and (torch.equal(buffer.activation, unique_buffer.activation))
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

    self._raw_layer_labels_list.remove(buffer_to_remove.tensor_label_raw)
    self._raw_layer_dict.pop(buffer_to_remove.tensor_label_raw)

    for layer in self:
        if buffer_to_remove.tensor_label_raw in layer.root_ancestors:
            layer.root_ancestors.remove(buffer_to_remove.tensor_label_raw)
            layer.root_ancestors.add(source_buffer.tensor_label_raw)
        if buffer_to_remove.tensor_label_raw in layer.internally_initialized_ancestors:
            layer.internally_initialized_ancestors.remove(buffer_to_remove.tensor_label_raw)
            layer.internally_initialized_ancestors.add(source_buffer.tensor_label_raw)

    self._remove_log_entry(buffer_to_remove, remove_references=True)
