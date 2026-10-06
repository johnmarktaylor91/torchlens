"""Steps 1-4: Output nodes, ancestry tracing, orphan removal, distance marking.

Step 1 (_add_output_layers): Creates dedicated output Op nodes, copying
    metadata from the original output tensors but stripping params and module info.
Step 2 (_find_output_ancestors): DFS backward from outputs marking has_output_descendant.
Step 3 (_remove_orphan_nodes): Bidirectional flood from inputs AND outputs to find
    connected nodes; any node unreachable from both is removed as an orphan.
Step 4 (_mark_layer_depths): Optional forward/backward BFS recording
    min/max hop counts from input and output nodes.
"""

from collections import OrderedDict
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, cast

import torch

from ..data_classes.op import Op, _dtype_or_none, _memory_or_none, _shape_or_none
from ..ir.op_record import amend_late_buffer_output_parent
from ..quantities import Bytes, Duration
from ..utils.display import identity
from ..utils.introspection import _get_code_context
from ..utils.rng import log_current_rng_states
from ..utils.tensor_utils import (
    get_memory_amount_from_metadata,
    safe_copy,
    safe_to,
    tensor_nanequal,
)
from ._materialize import _recorded_buffer_address

if TYPE_CHECKING:
    from ..data_classes.trace import Trace


def output_payload_aliases_parent(returned: Any, retained: Any) -> bool:
    """True iff the returned output tensor IS the producer's retained payload value.

    The output pseudo-row may share the producer's ONE retained payload
    (HONESTY 13-R1) exactly when the tensor the model returned equals the
    producer's retained ``out`` bit-for-bit under the same shape, dtype and
    device (NaN-equal, so a NaN output is not a spurious "difference"). Any
    other case -- unretained producer, meta payloads, a cross-device
    ``output_device`` policy, an in-place mutation after the producer, a view
    whose base shape differs -- keeps the historical physical copy so the
    variation is recorded, never hidden.
    """

    if not (torch.is_tensor(returned) and torch.is_tensor(retained)):
        return False
    if returned.is_meta or retained.is_meta:
        return False
    if (
        tuple(returned.shape) != tuple(retained.shape)
        or returned.dtype != retained.dtype
        or str(returned.device) != str(retained.device)
    ):
        return False
    return bool(tensor_nanequal(returned, retained))


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
        One raw parent label per output tensor. ``None`` is retained only for
        legacy callers; the normal torch backend fails loud on non-buffer
        unattributable outputs before Step 1.
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
            from .._errors import OutputAttributionError

            try:
                shape_text = str(tuple(output_tensor.shape))
            except RuntimeError:
                # Shapeless variants (nested) raise from ``.shape``; the refusal
                # must stay typed instead of crashing on its own message (R65 --
                # same guard as the backend.py twin, which also discloses the
                # output address this site does not hold).
                shape_text = "<unavailable>"
            raise OutputAttributionError(
                "TorchLens could not attribute a model output tensor to any traced op "
                f"(shape={shape_text}, dtype={output_tensor.dtype}). "
                "This may indicate an opaque execution boundary or a pre-bound torch "
                "function that escaped wrapping.",
                code="output_attribution_failed",
                remedy=(
                    "use ordinary torch module attributes during forward, or "
                    "bind/import torch functions after TorchLens has wrapped torch"
                ),
            )

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
                capture_events.append_amendment(
                    amend_late_buffer_output_parent(event.seq, parent_label, is_output_parent=True)
                )
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
    if event is None or event.kind != "source" or event.layer_type != "buffer":
        return False
    return _recorded_buffer_address(event) == buffer_address


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
        # That list may be SHORTER than the outputs (unattributed outputs have
        # no entry), so normalize it to one slot per output tensor here. Padding
        # with None is behavior-identical to the truncating zip this replaces:
        # the padded slots are dropped by the ``is not None`` filter below, and
        # surplus labels were already discarded. Normalizing lets the pairing be
        # strict, which is what actually matters -- a labels/tensors/addresses
        # length mismatch on the modern path must fail loud rather than yield a
        # silently shorter, entirely plausible set of output nodes.
        legacy_labels = list(self.output_layers)[: len(output_tensors)]
        output_parent_labels = legacy_labels + [None] * (len(output_tensors) - len(legacy_labels))

    paired_outputs = [
        (parent_label, output_tensor, output_address)
        for parent_label, output_tensor, output_address in zip(
            output_parent_labels, output_tensors, output_addresses, strict=True
        )
        if parent_label is not None
    ]
    new_output_layers = []
    _core = self.__dict__.get("_trace_core")
    _op_store = _core.ops if _core is not None else None
    if _op_store is not None and _op_store.frozen:
        _op_store = None
    for i, (output_layer_label, output_tensor, output_address_suffix) in enumerate(paired_outputs):
        output_node = self[output_layer_label]
        # Internal output-node synthesis is a builder row append on the
        # trace's own store (detached only for legacy/preview traces).
        new_output_node = cast(Op, output_node._copy_for_output(_store=_op_store))
        new_output_node.layer_type = "output"
        new_output_node.is_output = True
        new_output_node.is_input = False
        new_output_node.is_buffer = False
        new_output_node._internal_set("interventions", [])
        new_output_node._internal_set("intervention_replaced", False)
        if i == len(paired_outputs) - 1:
            new_output_node.is_final_output = True
        self._raw_graph_ws.layer_counter += 1
        new_output_node._label_raw = f"output_{i + 1}_raw"
        new_output_node._layer_label_raw = new_output_node._label_raw
        new_output_node.raw_index = self._raw_graph_ws.layer_counter
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

        # Tensor metadata must describe the tensor the model actually returned,
        # not the parent op's recorded output: after an in-place mutation
        # through a view (``y = x[...]; y.zero_(); return x``) the returned
        # base tensor's label is advanced to the mutating op, whose recorded
        # output is the VIEW — copying its shape would make the output node
        # claim the view's shape for the full base tensor.
        new_output_node.shape = tuple(output_tensor.shape)
        new_output_node.dtype = output_tensor.dtype
        new_output_node.activation_memory = Bytes(
            get_memory_amount_from_metadata(
                output_tensor, new_output_node.shape, new_output_node.dtype
            )
        )

        # Fix function information:

        new_output_node.func = identity
        new_output_node.func_name = "none"
        # Identity partition (A1): a synthetic output node is a BOUNDARY
        # bookkeeping row -- it executed nothing, so it OWNS no compute. The
        # wholesale clone previously inherited the producing op's FLOPs,
        # double-counting the terminal op in every total (+31% on gpt2, where
        # the duplicated row is the tied vocabulary projection). The producing
        # op owns the output tensor and its compute; the alias row displays
        # shape only and carries not-applicable (None) compute.
        new_output_node.flops_forward = None
        new_output_node.flops_backward = None
        new_output_node.code_context = _get_code_context(
            self.num_context_lines,
            source_loading_enabled=self.save_code_context,
            disable_col_offset=False,
        )
        new_output_node.var_names = []
        new_output_node.func_duration = Duration(0)
        new_output_node.func_rng_states = (
            log_current_rng_states(torch_only=True) if self.save_rng_states else {}
        )
        new_output_node.arg_names = ()
        new_output_node.num_args_total = 0
        new_output_node.num_pos_args = 0
        new_output_node.num_kwargs = 0
        new_output_node.non_tensor_pos_args = []
        new_output_node.non_tensor_kwargs = {}
        new_output_node.func_non_tensor_args = []
        new_output_node.grad_fn_class_name = None
        new_output_node.autograd_memory = None
        new_output_node.num_autograd_tensors = None
        # An output pseudo-node runs no kernel: "not measured" is None, never
        # a fabricated 0 B (observe item 15 zero-site sweep).
        new_output_node.bytes_delta_at_call = None
        new_output_node.bytes_peak_at_call = None
        new_output_node._internal_set("saved_args", [output_tensor])
        new_output_node._internal_set("saved_kwargs", {})

        # Strip any params:

        new_output_node.parent_params = []
        new_output_node._param_barcodes = []
        new_output_node.parent_param_ops = {}
        new_output_node._param_logs = []
        new_output_node.param_shapes = []
        new_output_node.num_params = 0
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
        # internal_source_parents is a DIRECT-PARENT relation ("the subset of MY parents
        # whose producers carry internal-source ancestry"), so it must be re-derived like
        # parents/parent_arg_positions above. It used to survive the wholesale clone
        # verbatim, which was wrong in BOTH directions on every model with a buffer or
        # factory-tensor ancestry (i.e. any BatchNorm net): it named labels that are not
        # parents of the output node at all, and omitted the output node's ONE real
        # parent, which does carry that ancestry.
        new_output_node.internal_source_parents = (
            [output_node._label_raw] if output_node.has_internal_source_ancestor else []
        )
        # root_ancestors must be re-derived the same way: the output node is NOT an
        # internal source (set above), so the armed ancestry-closure convention
        # requires root_ancestors == input_ancestors | internal_source_ancestors.
        # The wholesale clone inherited the DIRECT parent's root_ancestors verbatim,
        # which is the empty set when that parent is a parentless factory source
        # (arange/zeros/... returned straight from forward) — the one source-minting
        # convention that leaves root_ancestors empty on the exempt source row.
        new_output_node.root_ancestors = set(output_node.input_ancestors) | set(
            output_node.internal_source_ancestors
        )
        new_output_node._edge_uses = []

        # Synthetic output nodes start with an EMPTY annotations namespace: the
        # wholesale clone otherwise inherits the terminal op's per-op payloads,
        # and when the model returns a collective boundary's result directly
        # that duplicated the portable collective_boundary_v1 payload INCLUDING
        # its correlation key onto a second op record (N ops advertising N-1
        # boundaries, breaking per-rank correlation-key uniqueness). The
        # journal's op_labels_raw is the boundary->op mapping authority; a
        # bookkeeping node is never a boundary carrier.
        new_output_node.annotations = {}

        # Clear func_config on synthetic output nodes:
        new_output_node.func_config = {}
        new_output_node.is_transform = False
        new_output_node.transform_kind = None
        new_output_node.transform_chain = ()
        new_output_node.transform_config = {}
        new_output_node.transform_fn_name = None
        new_output_node.transform_fn_qualname = None
        new_output_node.transform_fn_source = None
        new_output_node.unattributed_tensor_args = ()
        new_output_node.dropped_edge_tensor_args = ()

        # Fix layer equivalence information:
        new_output_node.pass_index = 1
        new_output_node.num_passes = 1
        new_output_node.equivalent_ops = {new_output_node._label_raw}
        new_output_node.recurrent_ops = []
        equiv_type = (
            f"output_{i + 1}_{'_'.join(tuple(str(s) for s in new_output_node.shape))}_"
            f"{str(new_output_node.dtype)}"
        )
        new_output_node.equivalence_class = equiv_type
        self.op_equivalence_classes[equiv_type].add(new_output_node._label_raw)

        # Track child tensor variations for output nodes.
        new_output_node.has_out_variations = False
        new_output_node.out_versions_by_child = {}
        if output_node.has_saved_activation:
            # The recomputed payload must inherit the PARENT payload's
            # detachment state: cooked/sparse traces store detached payloads
            # (a retained graph would poison later captures), while live
            # traces keep the graph-attached output — it is the very handle
            # log_backward() differentiates through.
            _parent_payload = (
                output_node.out if output_node.out is not None else output_node.transformed_out
            )
            _detach_payload = not (
                torch.is_tensor(_parent_payload) and _parent_payload.grad_fn is not None
            )
            if output_payload_aliases_parent(output_tensor, output_node.out):
                # HONESTY 13-R1: the producer already retains this exact value,
                # so the output pseudo-row rides that ONE retained payload
                # instead of a second physical copy (the byte model charged
                # both: a full duplicate of the logits on large models). The
                # A1 identity partition already says the producer owns the
                # bytes; the payload now agrees. Differing values (an in-place
                # mutation after the producer) still copy below and record
                # the variation.
                actual_output_raw = output_node.out
            else:
                actual_output_raw = safe_copy(output_tensor, detach_tensor=_detach_payload)
                if output_node.output_device not in [str(actual_output_raw.device), "same"]:
                    actual_output_raw = safe_to(actual_output_raw, output_node.output_device)
            actual_output_transformed = None
            if self.activation_transform is not None:
                actual_output_transformed = output_node._apply_transform(
                    actual_output_raw,
                    self.activation_transform,
                    transform_kind="out",
                    streaming_active=getattr(self, "_out_writer", None) is not None,
                )
                output_node._validate_streaming_transform_output(
                    actual_output_transformed,
                    transform_kind="out",
                    streaming_active=getattr(self, "_out_writer", None) is not None,
                )
            raw_retained = output_node.out is not None
            new_output_node._internal_set("out", actual_output_raw if raw_retained else None)
            new_output_node._internal_set("transformed_out", actual_output_transformed)
            new_output_node.transformed_out_shape = _shape_or_none(actual_output_transformed)
            new_output_node.transformed_out_dtype = _dtype_or_none(actual_output_transformed)
            new_output_node.transformed_activation_memory = _memory_or_none(
                actual_output_transformed
            )

            comparison_output = output_node.out if raw_retained else output_node.transformed_out
            actual_comparison = actual_output_raw if raw_retained else actual_output_transformed
            if (
                comparison_output is not None
                and actual_comparison is not None
                and not tensor_nanequal(actual_comparison, comparison_output)
            ):
                output_node.out_versions_by_child[new_output_node._label_raw] = actual_comparison
                output_node.has_out_variations = True

        # Change original output node:

        output_node.children.append(new_output_node._label_raw)

        self._raw_graph_ws.raw_layer_dict[new_output_node._label_raw] = new_output_node
        self._raw_graph_ws.raw_layer_labels_list.append(new_output_node._label_raw)

        new_output_layers.append(new_output_node._label_raw)

    self.output_layers = new_output_layers


def _find_output_ancestors(self: "Trace") -> None:
    """Step 2: Mark every node that is an ancestor of an output node.

    Walks the capture DAG in reverse topological order so every child's complete
    descendant set is finalized before it is propagated to a parent. This keeps
    ancestry metadata complete even when optional Step 4 distance computation is
    disabled.
    """

    for node_label in reversed(self._raw_graph_ws.raw_layer_labels_list):
        node = self[node_label]
        for child_node_label in node.children:
            child = self[child_node_label]
            if child.has_output_descendant:
                node.has_output_descendant = True
                node.output_descendants.update(child.output_descendants)


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
    orig_nodes = set(self._raw_graph_ws.raw_layer_labels_list)
    nodes_seen = set()
    # Seed with inputs, outputs, and written buffer-version nodes. Written
    # buffers such as BatchNorm.num_batches_tracked are state transitions even
    # when the updated value is not read later in the forward graph.
    written_buffer_layers = [
        label
        for label in self.buffer_layers
        if getattr(self._raw_graph_ws.raw_layer_dict[label], "buffer_write_kind", None) is not None
    ]
    node_stack = self.input_layers + self.output_layers + written_buffer_layers
    # Shadow sets over the two trace-level list[str] sink ledgers: the former
    # per-node `label not in <list>` scans were O(k^2) in sink count
    # (hunt-6 R52-2). The lists stay the portable source of truth.
    seen_sink_labels = set(self.internal_sink_ops)
    seen_terminated_bool_labels = set(self.internally_terminated_bool_ops)
    while len(node_stack) > 0:
        tensor_label = node_stack.pop()
        nodes_seen.add(tensor_label)
        layer_entry = self._raw_graph_ws.raw_layer_dict[tensor_label]
        if (len(layer_entry.children) == 0) and (not layer_entry.is_output):
            _log_internally_terminated_tensor(
                self,
                tensor_label,
                seen_sink_labels=seen_sink_labels,
                seen_terminated_bool_labels=seen_terminated_bool_labels,
            )
        # Follow BOTH directions to ensure full bidirectional reachability.
        for next_label in layer_entry.children + layer_entry.parents:
            if next_label not in nodes_seen:
                node_stack.append(next_label)

    nodes_seen = _expand_seen_nodes_to_complete_func_call_groups(self, nodes_seen)
    orphan_nodes = orig_nodes - nodes_seen
    self._orphan_labels = [
        label for label in self._raw_graph_ws.raw_layer_labels_list if label in orphan_nodes
    ]
    self._orphan_logs = tuple(
        self._raw_graph_ws.raw_layer_dict[label] for label in self._orphan_labels
    )
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
    # Record pruned torch-RNG ops that DROVE control flow before the orphan
    # metadata is stripped: an ``if torch.rand(()) > 0.5`` predicate is
    # input-disconnected and orphaned away, so the runnable descriptor would
    # otherwise never learn the taken branch was nondeterministic + unwitnessed.
    _record_pruned_rng_control_flow(self, orphan_nodes)
    # Record pruned in-place ops that mutated an UNLABELLED (invisible ``.data`` / foreign)
    # alias: their write targets storage the sparse DAG cannot model, so dropping the op
    # silently loses the mutation. Recording it lets the runnable descriptor stay honestly
    # UNVERIFIABLE instead of VERIFYING a replay that omits the write.
    _record_pruned_alias_mutation(self, orphan_nodes)
    if getattr(self, "keep_orphans", False):
        for orphan_label in orphan_nodes:
            self._raw_graph_ws.raw_layer_dict[orphan_label].is_orphan = True
        return

    # Record the ``func_call_id``s of the ops being INTENTIONALLY orphan-pruned
    # here (dead computation that never reaches an output). The completeness
    # backstop uses this so that captured-then-orphan-pruned dispatchable ops are
    # accounted for on the captured side rather than false-firing as an untraced
    # dispatch. This is a plain runtime attribute (never serialized / in a
    # ``*_FIELD_ORDER``) and is only recorded on the removal path -- ``keep_orphans``
    # leaves the ops in the final trace, so nothing is pruned. Read now, before
    # removal, while the raw entries still carry their capture-time ``func_call_id``.
    self._orphan_pruned_func_call_ids = {
        func_call_id
        for label in orphan_nodes
        for func_call_id in (
            getattr(self._raw_graph_ws.raw_layer_dict[label], "func_call_id", None),
        )
        if isinstance(func_call_id, int)
    }

    # Read the source-less-argument witness of the ops being pruned NOW: removal
    # strips their fields, and the provenance disclosure step
    # (``_warn_unattributed_tensor_args``) runs later, so a source-less tensor whose
    # only consumer is pruned (``G.sum().item()``, a branch predicate) would lose
    # its witness. ``keep_orphans`` leaves the ops (and their witness) in the trace.
    orphan_witness = [
        f"{label} ({', '.join(positions)}, pruned)"
        for label in self._raw_graph_ws.raw_layer_labels_list
        if label in orphan_nodes
        for positions in (
            tuple(
                getattr(self._raw_graph_ws.raw_layer_dict[label], "unattributed_tensor_args", ())
                or ()
            ),
        )
        if positions
    ]
    if orphan_witness:
        self.__dict__["_orphan_unattributed_tensor_args"] = orphan_witness

    # Batch-remove orphaned nodes and rebuild the ordered layer dict/list.
    orphan_entries = [self._raw_graph_ws.raw_layer_dict[label] for label in orphan_nodes]
    self._batch_remove_log_entries(orphan_entries, remove_references=True)

    new_layer_dict = OrderedDict()
    new_layer_list = []
    for tensor_label in self._raw_graph_ws.raw_layer_labels_list:
        if tensor_label not in orphan_nodes:
            new_layer_dict[tensor_label] = self._raw_graph_ws.raw_layer_dict[tensor_label]
            new_layer_list.append(tensor_label)
    self._raw_graph_ws.raw_layer_labels_list = new_layer_list
    self._raw_graph_ws.raw_layer_dict = new_layer_dict


def _child_edge_totally_overwrites(self: "Trace", walked_label: str, child_label: str) -> bool:
    """Return whether a child op TOTALLY overwrites the walked tensor's bytes (r53 hon_2).

    The sanitized pruned-orphan walk stops at a total-writer edge whose
    DESTINATION is the walked tensor (``copy_``/``zero_``/``fill_``/RNG-fill
    receiver, or an ``out=`` destination): the post-write value is independent
    of the walked nondeterministic bytes, so a pruned ``empty -> fill_(1) ->
    branch`` chain must NOT be flagged (empty-plus-fill is THE idiomatic
    scratch init). The walk CONTINUES when the walked tensor is the VALUE
    SOURCE of the writer (``dst.copy_(walked)``). Unknown or partial in-place
    writers never stop the walk (fail closed).
    """

    from ..utils.rng import qualname_is_uninit_total_writer

    op = self._raw_graph_ws.raw_layer_dict.get(child_label)
    if op is None:
        return False
    func_id = getattr(op, "func_id", None)
    namespace = getattr(func_id, "namespace", None)
    qualname = getattr(func_id, "qualname", None)
    arg_locs = getattr(op, "parent_arg_positions", None)
    args_locs = arg_locs.get("args", {}) if isinstance(arg_locs, dict) else {}
    kwargs_locs = arg_locs.get("kwargs", {}) if isinstance(arg_locs, dict) else {}
    if (
        namespace in ("torch", "torch.Tensor", "torch.nn.functional")
        and kwargs_locs.get("out") == walked_label
    ):
        return True
    return qualname_is_uninit_total_writer(namespace, qualname) and args_locs.get(0) == walked_label


def _rng_orphan_drove_control_or_output(
    self: "Trace", start_label: str, escape_sources: frozenset[str]
) -> bool:
    """Return whether an orphaned nondeterministic op's value steered control or output.

    Walks the forward (child) chain from ``start_label``. The nondeterministic
    result (a seeded torch-RNG draw or an uninitialized-memory family product,
    r53 hon_2) "influenced control" when the op itself or any descendant is a
    recorded tensor->host escape source (its value was read by
    ``bool()``/``.item()`` to steer pure-Python control flow); it "influenced
    output" when a descendant is an output node (defensive: an output-reaching
    op is not normally orphaned). The walk is SANITIZED: it never crosses a
    total-writer edge whose destination is the walked tensor
    (``_child_edge_totally_overwrites``), so an empty-then-fully-written
    scratch chain stays clean. A genuinely-dead draw whose result feeds nothing
    reaches neither and returns ``False``, so it stays VERIFIED.
    """

    seen: set[str] = set()
    stack = [start_label]
    while stack:
        label = stack.pop()
        if label in seen:
            continue
        seen.add(label)
        op = self._raw_graph_ws.raw_layer_dict.get(label)
        if op is None:
            continue
        if label in escape_sources or getattr(op, "is_output", False):
            return True
        for child_label in getattr(op, "children", ()) or ():
            if child_label in seen:
                continue
            if _child_edge_totally_overwrites(self, label, child_label):
                continue
            stack.append(child_label)
    return False


def _orphan_is_uninit_alloc_source(self: "Trace", op: Any) -> bool:
    """Return whether an orphaned op is an uninitialized-memory value source (r53 hon_2).

    Keys on the shared closed family table (``utils/rng.py``): the ``empty``
    factory family taints unless its product has zero elements; a resize
    spelling taints only when it GREW its receiver beyond the pre-call element
    count (shrink/same-size preserves the prefix), failing closed to tainted
    when either shape is unreadable. The whole family is clean when the
    capture-time ambient snapshot proves deterministic fill.
    """

    from ..utils.rng import (
        deterministic_fill_governs,
        qualname_is_uninit_growth_resize,
        qualname_is_uninitialized_alloc,
    )

    func_id = getattr(op, "func_id", None)
    namespace = getattr(func_id, "namespace", None)
    qualname = getattr(func_id, "qualname", None)
    is_factory = qualname_is_uninitialized_alloc(namespace, qualname)
    is_resize = qualname_is_uninit_growth_resize(namespace, qualname)
    if not is_factory and not is_resize:
        return False
    snapshot = self._runnable.capture_ambient
    if isinstance(snapshot, dict) and deterministic_fill_governs(
        snapshot.get("deterministic_algorithms"),
        snapshot.get("fill_uninitialized_memory"),
    ):
        return False

    def _numel(shape: Any) -> int | None:
        """Element count for a fully-static integer shape tuple, else ``None``."""

        if not isinstance(shape, tuple):
            return None
        numel = 1
        for dim in shape:
            if not isinstance(dim, int):
                return None
            numel *= dim
        return numel

    out_numel = _numel(getattr(op, "shape", None))
    if is_factory:
        return out_numel is None or out_numel > 0
    arg_locs = getattr(op, "parent_arg_positions", None)
    args_locs = arg_locs.get("args", {}) if isinstance(arg_locs, dict) else {}
    receiver = self._raw_graph_ws.raw_layer_dict.get(args_locs.get(0, ""))
    pre_numel = _numel(getattr(receiver, "shape", None)) if receiver is not None else None
    return pre_numel is None or out_numel is None or out_numel > pre_numel


def _record_pruned_rng_control_flow(self: "Trace", orphan_nodes: set[str]) -> None:
    """Flag pruned nondeterministic-source ops that drove control flow (runnable descriptor).

    An orphaned op is a nondeterministic value source when its captured
    callable maps to an ATen ``nondeterministic_seeded`` overload OR to the
    uninitialized-memory family (r53 hon_2, shared predicate). When such an
    op's value reached a control decision (or, defensively, an output) through
    the sanitized child walk, the recorded taken branch is nondeterministic yet
    fully pruned from the visible graph. Recording its raw label in the
    weak-keyed side table lets the runnable producer downgrade witness
    completeness so the model is honestly UNVERIFIABLE + NOT_APPLICABLE
    instead of falsely VERIFIED + ATTESTED. This only reads orphan metadata and
    records a side-channel fact; it never alters the visible graph.
    """

    from ..backends.torch.completeness_witness import (
        host_escape_source_labels,
        record_pruned_rng_control_source,
    )
    from ..utils.rng import aten_qualname_is_seeded_rng

    escape_sources = host_escape_source_labels(self)
    for label in orphan_nodes:
        op = self._raw_graph_ws.raw_layer_dict.get(label)
        if op is None:
            continue
        func_id = getattr(op, "func_id", None)
        if func_id is None:
            continue
        if not aten_qualname_is_seeded_rng(
            getattr(func_id, "namespace", None), getattr(func_id, "qualname", None)
        ) and not _orphan_is_uninit_alloc_source(self, op):
            continue
        if _rng_orphan_drove_control_or_output(self, label, escape_sources):
            record_pruned_rng_control_source(self, label)


def _record_pruned_alias_mutation(self: "Trace", orphan_nodes: set[str]) -> None:
    """Flag orphan-pruned in-place ops that mutated an unlabelled alias, for the runnable descriptor.

    Capture records (in a weak-keyed side table) every in-place op whose mutation TARGET carried no
    resolvable capture label -- an invisible ``.data`` / foreign alias (``y.data.add_(5.0)``). Such
    an op's output slot feeds nothing in the tensor graph, so orphan removal drops it and the write
    to the aliased storage is silently lost: a sparse replay recomputes the PRE-mutation value and
    would falsely report VERIFIED. Recording the raw label of each candidate that is ACTUALLY pruned
    lets the runnable producer downgrade witness completeness so the model is honestly UNVERIFIABLE +
    NOT_APPLICABLE. A candidate op that survives pruning is graph-represented (replayed) and never
    recorded. This only reads orphan metadata and records a side-channel fact; it never alters the
    visible graph.
    """

    from ..backends.torch.completeness_witness import (
        alias_mutation_candidate_labels,
        record_pruned_alias_mutation_source,
    )

    candidates = alias_mutation_candidate_labels(self)
    if not candidates:
        return
    for label in candidates & orphan_nodes:
        record_pruned_alias_mutation_source(self, label)


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
    for raw_label in self._raw_graph_ws.raw_layer_labels_list:
        func_call_id = getattr(self._raw_graph_ws.raw_layer_dict[raw_label], "func_call_id", None)
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
    traversal_order: Iterable[str]
    if mode == "input":
        starting_nodes = self.input_layers[:]
        min_field = "min_distance_from_input"
        max_field = "max_distance_from_input"
        marker_field = "has_input_ancestor"
        layer_logging_field = "input_ancestors"
        forward_field = "children"
        traversal_order = self._raw_graph_ws.raw_layer_labels_list
    elif mode == "output":
        starting_nodes = self.output_layers[:]
        min_field = "min_distance_to_output"
        max_field = "max_distance_to_output"
        marker_field = "has_output_descendant"
        layer_logging_field = "output_descendants"
        forward_field = "parents"
        traversal_order = reversed(self._raw_graph_ws.raw_layer_labels_list)
    else:
        raise ValueError("Mode must be either 'input' or 'output'")

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


def _log_internally_terminated_tensor(
    self: "Trace",
    tensor_label: str,
    *,
    seen_sink_labels: set[str],
    seen_terminated_bool_labels: set[str],
) -> None:
    """Mark a tensor as terminated inside the model (no children reaching an output node).

    ``seen_sink_labels`` / ``seen_terminated_bool_labels`` are the caller's
    persistent shadow sets over ``internal_sink_ops`` /
    ``internally_terminated_bool_ops``; the membership guards read them
    instead of rescanning the growing lists once per visited node
    (hunt-6 R52-2).
    """
    layer_entry = self[tensor_label]
    layer_entry.is_internal_sink = True
    if tensor_label not in seen_sink_labels:
        seen_sink_labels.add(tensor_label)
        self.internal_sink_ops.append(tensor_label)
        if layer_entry.is_scalar_bool and (tensor_label not in seen_terminated_bool_labels):
            seen_terminated_bool_labels.add(tensor_label)
            self.internally_terminated_bool_ops.append(tensor_label)
            layer_entry.is_terminal_bool = True
