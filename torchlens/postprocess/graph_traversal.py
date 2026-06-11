"""Steps 1-4: Output nodes, ancestry tracing, orphan removal, distance marking.

Step 1 (_add_output_layers): Creates dedicated output Op nodes, copying
    metadata from the original output tensors but stripping params and module info.
Step 2 (_find_output_ancestors): DFS backward from outputs marking has_output_descendant.
Step 3 (_remove_orphan_nodes): Bidirectional flood from inputs AND outputs to find
    connected nodes; any node unreachable from both is removed as an orphan.
Step 4 (_mark_layer_depths): Optional forward/backward BFS recording
    min/max hop counts from input and output nodes.
"""

import dataclasses
from collections import OrderedDict
from typing import TYPE_CHECKING, cast

import torch

from ..quantities import Bytes, Duration
from ..utils.display import identity
from ..utils.rng import log_current_rng_states
from ..utils.tensor_utils import safe_copy, safe_to, tensor_nanequal
from ..utils.introspection import _get_code_context
from ..data_classes.op import Op

if TYPE_CHECKING:
    from ..data_classes.trace import Trace


def _resolve_output_parent_labels(
    self: "Trace", output_tensors: list[torch.Tensor]
) -> list["str | None"]:
    """Resolve the raw graph parent label for every model output tensor.

    When every output was attributed during capture, ``self.output_layers``
    already pairs positionally with the output tensors and is returned as-is.
    Otherwise some outputs have no graph entry, and the historical positional
    pairing silently shifted every value/address binding. This walks the
    outputs explicitly, keeping capture-attributed labels in output order, and
    handles the one attributable gap: a registered buffer returned directly
    from ``forward()`` without ever being touched by a traced op. Such a
    buffer has no label and no graph node, so a buffer-only model used to
    "trace" with an empty ``output_layers`` list and then fail
    ``validate_forward_pass`` with "No output layers found". For exactly that
    case the buffer is logged as a late source node through the same pathway
    as capture-time buffer reads, so Step 1 can bind a proper ``output_N``
    node to the buffer's value.

    Must run BEFORE Step 0 event materialization: the late source node is
    appended to the capture event stream and materializes with every other op.

    Parameters
    ----------
    self:
        Trace being postprocessed.
    output_tensors:
        The actual output tensors returned by the model's ``forward()``.

    Returns
    -------
    list[str | None]
        One raw parent label per output tensor; ``None`` for output tensors
        TorchLens cannot attribute (e.g. user-injected tensors), which Step 1
        skips exactly as before.
    """
    from collections import deque

    from ..backends.torch._tl import clear_tensor_label, get_tensor_label
    from ..backends.torch.sources import log_source_tensor

    # Common case: capture attributed every output -- the existing positional
    # pairing is exact, and nothing needs synthesizing.
    if len(self.output_layers) == len(output_tensors):
        return list(self.output_layers)

    capture_events = getattr(self, "capture_events", None)
    pending_labels = deque(self.output_layers)
    buffer_label_set = set(self.buffer_layers)
    parent_labels: list[str | None] = []
    buffer_addresses_by_id: dict[int, str] | None = None
    late_buffer_labels_by_tensor_id: dict[int, str | None] = {}
    for output_tensor in output_tensors:
        parent_label = get_tensor_label(output_tensor)
        if parent_label is not None:
            # Attributed at capture and still tagged; keep the queue in step.
            if pending_labels and pending_labels[0] == parent_label:
                pending_labels.popleft()
            parent_labels.append(parent_label)
            continue

        # Lazily index the source model's registered buffers by identity.
        if buffer_addresses_by_id is None:
            model_ref = getattr(self, "_source_model_ref", None)
            model = model_ref() if model_ref is not None else None
            buffer_addresses_by_id = (
                {id(buffer): address for address, buffer in model.named_buffers()}
                if model is not None
                else {}
            )
        buffer_address = buffer_addresses_by_id.get(id(output_tensor))
        if buffer_address is None:
            # Untraceable user-injected output: skip, exactly as before.
            parent_labels.append(None)
            continue

        # The tensor IS a registered buffer with no live label. Session
        # cleanup strips capture labels from model state, so a capture-time
        # attribution for this position would be waiting at the queue head as
        # a buffer node for the same address.
        head_label = pending_labels[0] if pending_labels else None
        if (
            head_label is not None
            and head_label in buffer_label_set
            and _event_buffer_address_matches(self, head_label, buffer_address)
        ):
            parent_labels.append(pending_labels.popleft())
            continue

        if capture_events is None:
            parent_labels.append(None)
            continue

        tensor_id = id(output_tensor)
        if tensor_id in late_buffer_labels_by_tensor_id:
            parent_labels.append(late_buffer_labels_by_tensor_id[tensor_id])
            continue

        # Genuinely unlogged static buffer returned as a model output: log it
        # as a late buffer source so it materializes with everything else.
        log_source_tensor(self, output_tensor, "buffer", buffer_address)
        parent_label = get_tensor_label(output_tensor)
        late_buffer_labels_by_tensor_id[tensor_id] = parent_label
        # Don't leak this session's label onto the model's live buffer: a
        # stale label would make the NEXT capture skip re-registering it.
        clear_tensor_label(output_tensor)
        # Mirror capture-time output marking on the synthesized event.
        if parent_label is not None:
            event = capture_events.op_event_by_label_raw.get(parent_label)
            if event is not None:
                updated_event = dataclasses.replace(event, is_output_parent=True)
                capture_events.op_event_by_label_raw[parent_label] = updated_event
                for index, existing_event in enumerate(capture_events.op_events):
                    if existing_event.label_raw == parent_label:
                        capture_events.op_events[index] = updated_event
                        break
        parent_labels.append(parent_label)
    return parent_labels


def _event_buffer_address_matches(self: "Trace", label_raw: str, buffer_address: str) -> bool:
    """Return whether a raw buffer node was logged for one buffer address.

    Parameters
    ----------
    self:
        Trace being postprocessed.
    label_raw:
        Raw label of a candidate buffer source node.
    buffer_address:
        Registered-buffer address to compare against.

    Returns
    -------
    bool
        ``True`` when the node's buffer equivalence class names the address.
    """

    capture_events = getattr(self, "capture_events", None)
    if capture_events is None:
        return False
    event = capture_events.op_event_by_label_raw.get(label_raw)
    equivalence_class = getattr(event, "equivalence_class", None) if event is not None else None
    if not equivalence_class or not equivalence_class.startswith("buffer_"):
        return False
    candidate = equivalence_class.removeprefix("buffer_")
    # The equivalence class may carry a module-stack suffix appended directly
    # after the address, so accept a prefix match as well as an exact one.
    return candidate == buffer_address or candidate.startswith(buffer_address)


def _add_output_layers(
    self: "Trace",
    output_tensors: list[torch.Tensor],
    output_addresses: list[str],
    output_parent_labels: "list[str | None] | None" = None,
) -> None:
    """Step 1: Add dedicated output nodes to the graph.

    For each tensor in the model's output, creates a new Op that acts
    as a terminal "output" node. The new node copies tensor metadata from the
    original output tensor but resets function, parameter, and module information
    to reflect that this is a synthetic bookkeeping node (func=identity,
    no params, no containing module). The original output tensor becomes the
    parent of the new output node.

    Output tensors are paired with their graph parents through
    ``output_parent_labels`` (one entry per output tensor, ``None`` for
    unattributable tensors, which are skipped). This keeps tensors, addresses,
    and parents aligned even when some outputs have no graph entry --
    previously a leading unlabeled output silently shifted every pairing.

    Also detects child_tensor_variations: if the actual output tensor differs
    from the parent's saved out (e.g., due to in-place ops or transform),
    the difference is recorded for validation.
    """
    if output_parent_labels is None:
        # Legacy alignment: callers that predate per-tensor parent resolution
        # paired ``self.output_layers`` positionally with the output tensors.
        output_parent_labels = list(self.output_layers)

    paired_outputs = [
        (parent_label, output_tensor, output_address)
        for parent_label, output_tensor, output_address in zip(
            output_parent_labels, output_tensors, output_addresses
        )
        if parent_label is not None
    ]
    new_output_layers = []
    for i, (output_layer_label, output_tensor, output_address_suffix) in enumerate(paired_outputs):
        output_node = self[output_layer_label]
        new_output_node = cast(Op, output_node.copy())
        new_output_node.layer_type = "output"
        new_output_node.is_output = True
        new_output_node.is_input = False
        new_output_node.is_buffer = False
        if i == len(paired_outputs) - 1:
            new_output_node.is_final_output = True
        self._layer_counter += 1
        new_output_node._label_raw = f"output_{i + 1}_raw"
        new_output_node._layer_label_raw = new_output_node._label_raw
        new_output_node.raw_index = self._layer_counter
        output_address = "output"
        if output_address_suffix != "":
            output_address += f".{output_address_suffix}"
        new_output_node.io_role = output_address
        container_path_meta = getattr(self, "_output_container_specs_by_raw_label", {}).get(
            output_node._label_raw
        )
        if container_path_meta is not None:
            new_output_node.container_path = container_path_meta[0]
            new_output_node.container_spec = container_path_meta[1]

        # Fix function information:

        new_output_node.func = identity
        new_output_node.func_name = "none"
        new_output_node.code_context = _get_code_context(
            self.num_context_lines,
            source_loading_enabled=self.save_code_context,
            disable_col_offset=False,
        )
        new_output_node.func_duration = Duration(0)
        new_output_node.func_rng_states = (
            log_current_rng_states(torch_only=True) if self.save_rng_states else {}
        )
        new_output_node.arg_names = tuple([])
        new_output_node.num_args_total = 0
        new_output_node.num_pos_args = 0
        new_output_node.num_kwargs = 0
        new_output_node.non_tensor_pos_args = []
        new_output_node.non_tensor_kwargs = {}
        new_output_node.func_non_tensor_args = []
        new_output_node.grad_fn_class_name = None
        new_output_node.autograd_memory = None
        new_output_node.num_autograd_tensors = None
        new_output_node.bytes_delta_at_call = Bytes(0)
        new_output_node.bytes_peak_at_call = Bytes(0)
        new_output_node._internal_set("saved_args", [output_tensor])
        new_output_node._internal_set("saved_kwargs", {})

        # Strip any params:

        new_output_node.parent_params = []
        new_output_node._param_barcodes = []
        new_output_node.parent_param_ops = {}
        new_output_node._param_logs = []
        new_output_node.param_shapes = []
        new_output_node.num_params = int(0)
        new_output_node.num_params_trainable = 0
        new_output_node.num_params_frozen = 0
        new_output_node.param_memory = Bytes(0)

        # Strip module info:

        new_output_node.module = None
        new_output_node.modules = []
        new_output_node.module_call_stack = []
        new_output_node.input_to_module_calls = []
        new_output_node.output_of_modules = [mod_pass[0] for mod_pass in output_node.modules]
        new_output_node.output_of_module_calls = output_node.modules
        new_output_node.is_module_output = False
        new_output_node.is_atomic_module = False
        new_output_node.atomic_module_call = None

        # Fix ancestry information:

        new_output_node.is_internal_source = False
        new_output_node.has_output_descendant = True
        new_output_node.output_descendants = {new_output_node._label_raw}
        new_output_node.children = []
        new_output_node.has_children = False
        new_output_node.parents = [output_node._label_raw]
        new_output_node.parent_arg_positions = {
            "args": {0: output_node._label_raw},
            "kwargs": {},
        }
        new_output_node._edge_uses = []

        # Clear func_config on synthetic output nodes:
        new_output_node.func_config = {}

        # Fix layer equivalence information:
        new_output_node.recurrent_ops = []
        equiv_type = (
            f"output_{'_'.join(tuple(str(s) for s in new_output_node.shape))}_"
            f"{str(new_output_node.dtype)}"
        )
        new_output_node.equivalence_class = equiv_type
        self.op_equivalence_classes[equiv_type].add(new_output_node._label_raw)

        # Track child tensor variations for output nodes.
        new_output_node.has_out_variations = False
        new_output_node.out_versions_by_child = {}
        if output_node.has_saved_activation:
            actual_output = safe_copy(output_tensor)
            if output_node.output_device not in [str(actual_output.device), "same"]:
                actual_output = safe_to(actual_output, output_node.output_device)
            actual_output_raw = actual_output
            if self.activation_transform is not None:
                actual_output = output_node._apply_transform(
                    actual_output,
                    self.activation_transform,
                    transform_kind="out",
                    streaming_active=getattr(self, "_out_writer", None) is not None,
                )
                output_node._validate_streaming_transform_output(
                    actual_output,
                    transform_kind="out",
                    streaming_active=getattr(self, "_out_writer", None) is not None,
                )
            comparison_output = (
                output_node.out if output_node.out is not None else output_node.transformed_out
            )
            if comparison_output is not None and not tensor_nanequal(
                actual_output, comparison_output
            ):
                output_node.out_versions_by_child[new_output_node._label_raw] = actual_output
                output_node.has_out_variations = True
                if output_node.out is None:
                    new_output_node._internal_set("transformed_out", actual_output)
                else:
                    new_output_node._internal_set("out", actual_output_raw)

        # Change original output node:

        output_node.children.append(new_output_node._label_raw)

        self._raw_layer_dict[new_output_node._label_raw] = new_output_node
        self._raw_layer_labels_list.append(new_output_node._label_raw)

        new_output_layers.append(new_output_node._label_raw)

    self.output_layers = new_output_layers


def _find_output_ancestors(self: "Trace") -> None:
    """Step 2: Mark every node that is an ancestor of an output node.

    Uses a LIFO stack (DFS) starting from output nodes. For each node popped,
    checks its children — if any child has_output_descendant, this node is too,
    and it inherits the child's output_descendants. Then pushes all unseen parents
    onto the stack.

    Note: A node may be pushed onto the stack multiple times if it's shared by
    sibling paths. The second pop is redundant (nodes_seen prevents re-pushing
    parents) but harmless — it may beneficially propagate output_descendants
    from newly-marked children on the second visit.

    The boolean has_output_descendant is always correct after this function; the
    output_descendants set may be incomplete for multi-output graphs, but Step 4's
    flood corrects it if distance marking is enabled.
    """
    node_stack = self.output_layers[:]
    nodes_seen = set()
    while len(node_stack) > 0:
        node_label = node_stack.pop()
        nodes_seen.add(node_label)
        node = self[node_label]
        for child_node_label in node.children:
            if self[child_node_label].has_output_descendant:
                node.has_output_descendant = True
                node.output_descendants.update(self[child_node_label].output_descendants)
        for parent_node_label in node.parents:
            if parent_node_label not in nodes_seen:
                node_stack.append(parent_node_label)


def _remove_orphan_nodes(self: "Trace") -> None:
    """Step 3: Remove orphan nodes unreachable from both inputs and outputs.

    Floods BIDIRECTIONALLY from input and output nodes simultaneously. A node is
    reachable if it can be reached by following children OR parents from
    any starting node. This bidirectional approach is necessary because:
    - Forward-only (from inputs) would miss nodes reachable only backward from outputs
      (e.g., internally-initialized tensors that only feed into outputs).
    - Backward-only (from outputs) would miss input-side dead ends.

    Any non-output node with no children is logged as an internally-terminated tensor
    (it produced a value that was never used by downstream computation reaching an output).
    """
    orig_nodes = set(self._raw_layer_labels_list)
    nodes_seen = set()
    # Seed with inputs, outputs, and written buffer-version nodes. Written
    # buffers such as BatchNorm.num_batches_tracked are state transitions even
    # when the updated value is not read later in the forward graph.
    written_buffer_layers = [
        label
        for label in self.buffer_layers
        if getattr(self._raw_layer_dict[label], "buffer_write_kind", None) is not None
    ]
    node_stack = self.input_layers + self.output_layers + written_buffer_layers
    while len(node_stack) > 0:
        tensor_label = node_stack.pop()
        nodes_seen.add(tensor_label)
        layer_entry = self._raw_layer_dict[tensor_label]
        if (len(layer_entry.children) == 0) and (not layer_entry.is_output):
            _log_internally_terminated_tensor(self, tensor_label)
        # Follow BOTH directions to ensure full bidirectional reachability.
        for next_label in layer_entry.children + layer_entry.parents:
            if next_label not in nodes_seen:
                node_stack.append(next_label)

    nodes_seen = _expand_seen_nodes_to_complete_func_call_groups(self, nodes_seen)
    orphan_nodes = orig_nodes - nodes_seen
    self._orphan_labels = [label for label in self._raw_layer_labels_list if label in orphan_nodes]
    self._orphan_logs = tuple(self._raw_layer_dict[label] for label in self._orphan_labels)
    self.orphan_records = [
        {
            "raw_label": orphan._label_raw,
            "label": orphan.label,
            "payload_ref": orphan.out_ref
            if getattr(orphan, "out_ref", None) is not None
            else orphan.out,
        }
        for orphan in self._orphan_logs
        if getattr(orphan, "has_saved_activation", False)
    ]
    if getattr(self, "keep_orphans", False):
        for orphan_label in orphan_nodes:
            self._raw_layer_dict[orphan_label].is_orphan = True
        return

    # Batch-remove orphaned nodes and rebuild the ordered layer dict/list.
    orphan_entries = [self._raw_layer_dict[label] for label in orphan_nodes]
    self._batch_remove_log_entries(orphan_entries, remove_references=True)

    new_layer_dict = OrderedDict()
    new_layer_list = []
    for tensor_label in self._raw_layer_labels_list:
        if tensor_label not in orphan_nodes:
            new_layer_dict[tensor_label] = self._raw_layer_dict[tensor_label]
            new_layer_list.append(tensor_label)
    self._raw_layer_labels_list = new_layer_list
    self._raw_layer_dict = new_layer_dict


def _expand_seen_nodes_to_complete_func_call_groups(
    self: "Trace", nodes_seen: set[str]
) -> set[str]:
    """Add raw-label siblings for any surviving ``func_call_id`` group.

    Parameters
    ----------
    nodes_seen:
        Raw labels reachable from the input/output flood.

    Returns
    -------
    set[str]
        Reachable raw labels expanded so multi-output wrapper calls are kept
        atomically.
    """

    func_groups: dict[int, set[str]] = {}
    for raw_label in self._raw_layer_labels_list:
        func_call_id = getattr(self._raw_layer_dict[raw_label], "func_call_id", None)
        if func_call_id is not None:
            func_groups.setdefault(func_call_id, set()).add(raw_label)

    expanded_seen = set(nodes_seen)
    changed = True
    while changed:
        changed = False
        for raw_labels in func_groups.values():
            if expanded_seen.intersection(raw_labels) and not raw_labels.issubset(expanded_seen):
                expanded_seen.update(raw_labels)
                changed = True
    return expanded_seen


def _mark_layer_depths(self: "Trace") -> None:
    """Step 4: Compute min/max hop distances from inputs and outputs.

    Runs two unidirectional floods: forward from inputs (following children)
    and backward from outputs (following parents). Each flood records
    min_distance_from_{input,output} and max_distance_from_{input,output} on
    every reachable node.

    This step is CONDITIONAL on ``self.mark_layer_depths`` — it is
    skipped when the user doesn't need distance metadata.
    """
    _flood_graph_from_input_or_output_nodes(self, "input")
    _flood_graph_from_input_or_output_nodes(self, "output")


def _flood_graph_from_input_or_output_nodes(self: "Trace", mode: str) -> None:
    """Flood the graph from input or output nodes, tracking min/max distance.

    Traverses unidirectionally from starting nodes (input or output), recording
    each node's min and max hop count from the start. Also marks each node's
    ancestry (input_ancestors or output_descendants).

    Unlike the bidirectional flood in Step 3, this flood is UNIDIRECTIONAL:
    from inputs it follows children (forward), from outputs it follows
    parents (backward). This ensures hop counts reflect actual data-flow
    distance, not arbitrary graph traversal paths.

    Nodes are processed once in topological order: forward from inputs, or
    backward from outputs. Each visited node propagates finalized distance and
    lineage state to its data-flow successors.

    Args:
        mode: 'input' to flood forward from inputs, 'output' to flood backward from outputs.
    """
    if mode == "input":
        starting_nodes = self.input_layers[:]
        min_field = "min_distance_from_input"
        max_field = "max_distance_from_input"
        marker_field = "has_input_ancestor"
        layer_logging_field = "input_ancestors"
        forward_field = "children"
        traversal_order = self._raw_layer_labels_list
    elif mode == "output":
        starting_nodes = self.output_layers[:]
        min_field = "min_distance_to_output"
        max_field = "max_distance_to_output"
        marker_field = "has_output_descendant"
        layer_logging_field = "output_descendants"
        forward_field = "parents"
        traversal_order = reversed(self._raw_layer_labels_list)
    else:
        raise ValueError("Mode but be either 'input' or 'output'")

    for starting_node_label in starting_nodes:
        starting_node = self[starting_node_label]
        _update_node_distance_vals(starting_node, min_field, max_field, 0)
        setattr(starting_node, marker_field, True)
        getattr(starting_node, layer_logging_field).add(starting_node_label)

    for current_node_label in traversal_order:
        current_node = self[current_node_label]
        current_min = getattr(current_node, min_field)
        current_max = getattr(current_node, max_field)
        if current_min is None or current_max is None:
            continue

        current_lineage = getattr(current_node, layer_logging_field)
        for next_node_label in getattr(current_node, forward_field):
            next_node = self[next_node_label]
            _update_node_distance_vals(next_node, min_field, max_field, current_min + 1)
            _update_node_distance_vals(next_node, min_field, max_field, current_max + 1)
            setattr(next_node, marker_field, True)
            getattr(next_node, layer_logging_field).update(current_lineage)


def _update_node_distance_vals(
    current_node: Op,
    min_field: str,
    max_field: str,
    nodes_since_start: int,
) -> None:
    """Update a node's min/max distance fields if the current hop count is a new extreme."""
    if getattr(current_node, min_field) is None:
        setattr(current_node, min_field, nodes_since_start)
    else:
        setattr(
            current_node,
            min_field,
            min(nodes_since_start, getattr(current_node, min_field)),
        )

    if getattr(current_node, max_field) is None:
        setattr(current_node, max_field, nodes_since_start)
    else:
        setattr(
            current_node,
            max_field,
            max(nodes_since_start, getattr(current_node, max_field)),
        )


def _log_internally_terminated_tensor(self: "Trace", tensor_label: str) -> None:
    """Mark a tensor as terminated inside the model (no children reaching an output node)."""
    layer_entry = self[tensor_label]
    layer_entry.is_internal_sink = True
    if tensor_label not in self.internal_sink_ops:
        self.internal_sink_ops.append(tensor_label)
        if layer_entry.is_scalar_bool and (tensor_label not in self.internally_terminated_bool_ops):
            self.internally_terminated_bool_ops.append(tensor_label)
            layer_entry.is_terminal_bool = True
