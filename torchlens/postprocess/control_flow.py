"""Steps 5-7: Conditional branches, module annotation fixes, buffer layer fixes.

Step 5 (_mark_conditional_branches) now runs a six-phase conditional pipeline:
    5a. Build AST file indexes for files referenced by terminal scalar bools.
    5b. Classify terminal bools into branch/non-branch contexts.
    5c. Materialize dense conditional events from structural AST keys.
    5d. Backward-flood IF edges from branch-participating bools only.
    5e. Attribute executed ops to THEN/ELIF/ELSE arms across every forward edge.
    5f. Materialize derived compatibility views from the new primary structures.
Step 6 (_fix_modules_for_internal_tensors): Append module path suffixes to
    equivalence_class. This is INTENTIONAL and affects loop detection (Step 8),
    ensuring operations in different modules are never grouped as the same layer.
Step 7 (_fix_buffer_layers): Connects buffer parents, deduplicates identical
    buffers (same module, same value, same parent), and assigns buffer pass numbers.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import chain
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

import torch

from ..data_classes.op_log import OpLog
from ..utils.display import identity
from ..utils.tensor_utils import safe_copy
from . import ast_branches

if TYPE_CHECKING:
    from ..data_classes.func_call_location import FuncCallLocation
    from ..data_classes.model_log import ConditionalEvent, Trace


_BRANCH_CONTEXT_KINDS = frozenset({"if_test", "elif_test", "ifexp"})


def _mark_conditional_branches(self: "Trace") -> None:
    """Step 5: Classify bools, materialize events, and attribute conditional edges.

    The public Step 5 entry point is preserved for the postprocess orchestrator,
    but the implementation now delegates to six internal phases:

    1. Build AST file indexes for all files touched by terminal scalar bools.
    2. Classify terminal bools and collect observed structural conditional keys.
    3. Materialize dense ``ConditionalEvent`` records from those keys.
    4. Backward-flood IF edges from branch-participating bools only.
    5. Attribute executed ops and forward edges to conditional branch arms.
    6. Rebuild compatibility views derived from the new primary structures.

    Performance fast-path: when no terminal scalar bools were captured, the
    model has no conditional branches the pipeline can attribute, so we skip
    the AST file indexing, per-bool classification, and per-op
    ``attribute_op()`` work. All Trace-level conditional collections are
    already initialized empty in :meth:`Trace.__init__`, and per-layer
    conditional fields are initialized to their empty defaults during
    capture (see ``capture/output_tensors.py``). The fast-path is verified
    against the slow-path defaults via ``tests/test_perf_bundle.py``.
    """

    if _can_fast_skip_step5(self):
        return

    file_indexes = _build_file_indexes(self)
    conditional_keys, bool_classifications = _classify_bool_layers(self)
    # Defensive guard: if no terminal bool produced a structural conditional
    # key, attribution will produce zero edges, matching the fast-skip output.
    # This invariant lets future refactors of the bool detector trip an
    # explicit assertion rather than silently make the fast-path miss work.
    if not bool_classifications:
        assert not conditional_keys, (
            "Internally-terminated bool layers were absent but conditional "
            "keys were produced; the fast-skip precondition is stale."
        )
    events_by_key = _materialize_conditional_records(
        self,
        file_indexes,
        conditional_keys,
        bool_classifications,
    )
    _mark_conditional_branches_if_backward_flood(self, bool_classifications)
    _attribute_branches_forward(self, events_by_key)
    _materialize_derived_views(self)


def _can_fast_skip_step5(self: "Trace") -> bool:
    """Return True when Step 5 has no work to do.

    The slow path's only branch-attributing inputs are the Trace's
    ``internally_terminated_bool_ops``: if no terminal scalar bool was
    captured, ``_iter_terminal_scalar_bool_labels`` yields nothing, so
    every downstream collection (events, edges, per-layer arm children)
    would resolve to its empty default. Skipping the slow path is then
    semantically equivalent to running it.

    The function also checks the Trace-level conditional collections
    (``conditional_records``, ``conditional_branch_edges``,
    ``conditional_arm_entry_edges``, ``conditional_edge_call_indices``). They are initialized empty in
    :meth:`Trace.__init__`, and the slow path resets them on entry.
    Any caller that pre-populated these would change the user-visible
    output if we skipped, so we conservatively run the slow path in that
    (pathological) case as well.
    """

    if self.internally_terminated_bool_ops:
        return False
    if self.conditional_records:
        return False
    if self.conditional_branch_edges:
        return False
    if self.conditional_arm_entry_edges:
        return False
    if self.conditional_edge_call_indices:
        return False
    return True


def _build_file_indexes(
    self: "Trace",
) -> Dict[str, Optional[ast_branches.FileIndex]]:
    """Phase 5a: Build cached AST indexes for files touched by terminal bools.

    Parameters
    ----------
    self:
        Model log being postprocessed.

    Returns
    -------
    Dict[str, Optional[ast_branches.FileIndex]]
        Mapping from filename to the cached AST index, or ``None`` when the
        file could not be parsed or loaded.
    """

    file_indexes: Dict[str, Optional[ast_branches.FileIndex]] = {}
    for bool_label in _iter_terminal_scalar_bool_labels(self):
        bool_layer = self[bool_label]
        for frame in bool_layer.code_context:
            if frame.file in file_indexes:
                continue
            file_indexes[frame.file] = ast_branches.get_file_index(frame.file)
    return file_indexes


def _classify_bool_layers(
    self: "Trace",
) -> Tuple[List[ast_branches.ConditionalKey], Dict[str, ast_branches.BoolClassification]]:
    """Phase 5b: Classify terminal scalar bools and collect observed conditionals.

    Parameters
    ----------
    self:
        Model log being postprocessed.

    Returns
    -------
    Tuple[List[ast_branches.ConditionalKey], Dict[str, ast_branches.BoolClassification]]
        First-seen ordered conditional keys plus per-bool classification results
        keyed by raw layer label.
    """

    bool_classifications: Dict[str, ast_branches.BoolClassification] = {}
    ordered_conditional_keys: Dict[ast_branches.ConditionalKey, None] = {}

    for bool_label in _iter_terminal_scalar_bool_labels(self):
        bool_layer = self[bool_label]
        classification = ast_branches.BoolClassification("unknown", None, None, None)
        for frame in reversed(bool_layer.code_context):
            frame_classification = ast_branches.classify_bool(
                frame.file,
                frame.line_number,
                frame.col_offset,
            )
            if frame_classification.kind == "unknown":
                continue
            classification = frame_classification
            break

        is_terminal_conditional_bool = (
            classification.kind in _BRANCH_CONTEXT_KINDS
            and classification.conditional_key is not None
        )
        bool_layer.conditional_context_kind = classification.kind
        bool_layer.conditional_wrapper_kind = classification.wrapper_kind
        bool_layer.is_terminal_conditional_bool = is_terminal_conditional_bool
        bool_layer.terminal_conditional_id = None
        bool_classifications[bool_label] = classification

        if is_terminal_conditional_bool:
            conditional_key = classification.conditional_key
            if conditional_key is None:
                raise ValueError("Branch-participating bool classification must include a key.")
            assert conditional_key is not None  # mypy narrowing
            ordered_conditional_keys.setdefault(conditional_key, None)

    return list(ordered_conditional_keys.keys()), bool_classifications


def _materialize_conditional_records(
    self: "Trace",
    file_indexes: Dict[str, Optional[ast_branches.FileIndex]],
    conditional_keys: List[ast_branches.ConditionalKey],
    bool_classifications: Dict[str, ast_branches.BoolClassification],
) -> Dict[ast_branches.ConditionalKey, "ConditionalEvent"]:
    """Phase 5c: Materialize dense conditional events and translate bool keys.

    Parameters
    ----------
    self:
        Model log being postprocessed.
    file_indexes:
        Cached AST file indexes from phase 5a.
    conditional_keys:
        Ordered structural conditional keys observed in phase 5b.
    bool_classifications:
        Per-bool classifications keyed by raw layer label.

    Returns
    -------
    Dict[ast_branches.ConditionalKey, ConditionalEvent]
        Mapping from structural conditional key to the dense event object.
    """

    from ..data_classes.model_log import ConditionalEvent

    record_lookup = _build_conditional_record_lookup(file_indexes)
    self.conditional_records = []

    events_by_key: Dict[ast_branches.ConditionalKey, ConditionalEvent] = {}
    for conditional_id, conditional_key in enumerate(conditional_keys):
        if conditional_key not in record_lookup:
            raise ValueError(
                f"Observed conditional key was not found in the AST index: {conditional_key!r}"
            )
        record, function_qualname = record_lookup[conditional_key]
        event = ConditionalEvent(
            id=conditional_id,
            kind=record.kind,
            source_file=record.source_file,
            function_qualname=function_qualname,
            function_span=record.function_span,
            if_stmt_span=record.if_stmt_span,
            test_span=record.test_span,
            branch_ranges=record.branch_ranges,
            branch_test_spans=record.branch_test_spans,
            call_depth=record.call_depth,
            parent_conditional_id=None,
            parent_branch_kind=record.parent_branch_kind,
        )
        events_by_key[conditional_key] = event
        self.conditional_records.append(event)

    for conditional_key in conditional_keys:
        record, _function_qualname = record_lookup[conditional_key]
        event = events_by_key[conditional_key]
        parent_conditional_key = record.parent_conditional_key
        if parent_conditional_key is not None and parent_conditional_key in events_by_key:
            event.parent_conditional_id = events_by_key[parent_conditional_key].id

    for bool_label, classification in bool_classifications.items():
        bool_layer = self[bool_label]
        bool_conditional_key: Optional[ast_branches.ConditionalKey] = classification.conditional_key
        if bool_conditional_key is None or bool_conditional_key not in events_by_key:
            bool_layer.terminal_conditional_id = None
            continue
        event = events_by_key[bool_conditional_key]
        bool_layer.terminal_conditional_id = event.id
        event.bool_layers.append(bool_label)

    for bool_label in _iter_terminal_scalar_bool_labels(self):
        assert not hasattr(self[bool_label], "_bool_conditional_key")

    return events_by_key


def _mark_conditional_branches_if_backward_flood(
    self: "Trace",
    bool_classifications: Dict[str, ast_branches.BoolClassification],
) -> None:
    """Phase 5d: Backward-flood IF edges from branch-participating bools only.

    Parameters
    ----------
    self:
        Model log being postprocessed.
    bool_classifications:
        Per-bool classifications keyed by raw layer label.
    """

    self.conditional_branch_edges = []
    for layer in self:
        if getattr(layer, "is_orphan", False):
            continue
        layer.conditional_entry_children = []
        layer.is_in_conditional_body = False

    branch_bool_labels = [
        bool_label
        for bool_label in _iter_terminal_scalar_bool_labels(self)
        if bool_classifications[bool_label].conditional_key is not None
        and self[bool_label].is_terminal_conditional_bool
    ]

    nodes_seen: Set[str] = set()
    node_stack = branch_bool_labels.copy()
    while node_stack:
        node_label = node_stack.pop()
        node = self[node_label]
        if node_label in nodes_seen:
            continue
        for parent_label in node.parents:
            parent_layer = self[parent_label]
            if parent_layer.has_output_descendant:
                parent_layer.conditional_entry_children.append(node_label)
                parent_layer.is_in_conditional_body = False
                nodes_seen.add(parent_label)
                self.conditional_branch_edges.append((parent_label, node_label))
            else:
                if parent_label in nodes_seen:
                    continue
                parent_layer.is_in_conditional_body = True
                node_stack.append(parent_label)

        nodes_seen.add(node_label)


def _attribute_branches_forward(
    self: "Trace",
    events_by_key: Dict[ast_branches.ConditionalKey, "ConditionalEvent"],
) -> None:
    """Phase 5e: Attribute executed ops and forward edges to conditional arms.

    Parameters
    ----------
    self:
        Model log being postprocessed.
    events_by_key:
        Structural-to-dense conditional event lookup created in phase 5c.
    """

    conditional_arm_entry_edges: Dict[Tuple[int, str], List[Tuple[str, str]]] = defaultdict(list)
    conditional_edge_call_indices: Dict[Tuple[str, str, int, str], List[int]] = defaultdict(list)

    for layer_label in self._raw_layer_labels_list:
        layer = self[layer_label]
        if getattr(layer, "is_orphan", False):
            continue
        layer.conditional_branch_stack = _translate_conditional_stack(
            layer.code_context,
            events_by_key,
        )
        layer.conditional_branch_depth = len(layer.conditional_branch_stack)
        layer.conditional_arm_children = {}

    for parent_label in self._raw_layer_labels_list:
        parent_layer = self[parent_label]
        if getattr(parent_layer, "is_orphan", False):
            continue
        for child_label in parent_layer.children:
            child_layer = self[child_label]
            if getattr(child_layer, "is_orphan", False):
                continue
            gained_entries = _get_gained_branch_entries(
                parent_layer.conditional_branch_stack,
                child_layer.conditional_branch_stack,
            )
            for conditional_id, branch_kind in gained_entries:
                parent_layer.conditional_arm_children.setdefault(conditional_id, {}).setdefault(
                    branch_kind, []
                ).append(child_label)
                conditional_arm_entry_edges[(conditional_id, branch_kind)].append(
                    (parent_layer._label_raw, child_layer._label_raw)
                )
                conditional_edge_call_indices[
                    (
                        parent_layer._label_raw.split(":", 1)[0],
                        child_layer._label_raw.split(":", 1)[0],
                        conditional_id,
                        branch_kind,
                    )
                ].append(parent_layer.call_index)

    self.conditional_arm_entry_edges = dict(conditional_arm_entry_edges)
    self.conditional_edge_call_indices = dict(conditional_edge_call_indices)


def _materialize_derived_views(self: "Trace") -> None:
    """Phase 5f: Rebuild compatibility views derived from primary conditional data.

    Parameters
    ----------
    self:
        Model log being postprocessed.
    """

    self.conditional_edge_call_indices = {
        key: sorted(set(call_indexs))
        for key, call_indexs in self.conditional_edge_call_indices.items()
    }

    for layer_label in self._raw_layer_labels_list:
        layer = self[layer_label]
        if getattr(layer, "is_orphan", False):
            continue
        layer.conditional_then_children = sorted(
            set(
                chain.from_iterable(
                    branch_children.get("then", [])
                    for branch_children in layer.conditional_arm_children.values()
                )
            )
        )

        elif_children: Dict[int, Set[str]] = defaultdict(set)
        for branch_children in layer.conditional_arm_children.values():
            for branch_kind, child_labels in branch_children.items():
                if not branch_kind.startswith("elif_"):
                    continue
                elif_index = int(branch_kind.split("_", 1)[1])
                elif_children[elif_index].update(child_labels)
        layer.conditional_elif_children = {
            elif_index: sorted(child_labels)
            for elif_index, child_labels in sorted(elif_children.items())
        }

        layer.conditional_else_children = sorted(
            set(
                chain.from_iterable(
                    branch_children.get("else", [])
                    for branch_children in layer.conditional_arm_children.values()
                )
            )
        )


def _iter_terminal_scalar_bool_labels(self: "Trace") -> List[str]:
    """Return terminal scalar bool labels in deterministic execution order.

    Parameters
    ----------
    self:
        Model log being postprocessed.

    Returns
    -------
    List[str]
        Raw tensor labels for terminal scalar bool layers, ordered by first-seen
        execution order in the model log.
    """

    terminal_bool_labels = set(self.internally_terminated_bool_ops)
    return [
        layer_label
        for layer_label in self._raw_layer_labels_list
        if layer_label in terminal_bool_labels
        and self[layer_label].is_scalar_bool
        and not getattr(self[layer_label], "is_orphan", False)
    ]


def _build_conditional_record_lookup(
    file_indexes: Dict[str, Optional[ast_branches.FileIndex]],
) -> Dict[ast_branches.ConditionalKey, Tuple[ast_branches.ConditionalRecord, str]]:
    """Build a structural-key lookup for materializing dense conditional events.

    Parameters
    ----------
    file_indexes:
        Cached file indexes produced in phase 5a.

    Returns
    -------
    Dict[ast_branches.ConditionalKey, Tuple[ast_branches.ConditionalRecord, str]]
        Mapping from structural conditional key to its record and owning
        function qualname.
    """

    record_lookup: Dict[
        ast_branches.ConditionalKey, Tuple[ast_branches.ConditionalRecord, str]
    ] = {}
    for file_index in file_indexes.values():
        if file_index is None:
            continue
        for scope in file_index.scopes:
            for conditional_record in scope.conditionals:
                record_lookup[conditional_record.key] = (conditional_record, scope.qualname)
    return record_lookup


def _translate_conditional_stack(
    code_context: List["FuncCallLocation"],
    events_by_key: Dict[ast_branches.ConditionalKey, "ConditionalEvent"],
) -> List[Tuple[int, str]]:
    """Translate a structural AST branch stack into dense conditional IDs.

    Parameters
    ----------
    code_context:
        Captured runtime call stack for one operation.
    events_by_key:
        Structural-to-dense conditional event lookup created in phase 5c.

    Returns
    -------
    List[Tuple[int, str]]
        Dense ``(conditional_id, branch_kind)`` pairs ordered outer-to-inner.
        Structural keys that were never materialized are dropped.
    """

    translated_stack: List[Tuple[int, str]] = []
    for conditional_key, branch_kind in _attribute_op_with_scope_fallback(code_context):
        if conditional_key not in events_by_key:
            continue
        translated_stack.append((events_by_key[conditional_key].id, branch_kind))
    return translated_stack


def _attribute_op_with_scope_fallback(
    code_context: List["FuncCallLocation"],
) -> List[Tuple[ast_branches.ConditionalKey, str]]:
    """Attribute an op, retrying decorated-function scope resolution when needed.

    Parameters
    ----------
    code_context:
        Captured runtime call stack for one operation.

    Returns
    -------
    List[Tuple[ast_branches.ConditionalKey, str]]
        Structural ``(conditional_key, branch_kind)`` pairs ordered
        outer-to-inner.
    """

    branch_stack = ast_branches.attribute_op(code_context)
    if branch_stack:
        return branch_stack

    fallback_branch_stack: List[Tuple[ast_branches.ConditionalKey, str]] = []
    for frame in code_context:
        file_index = ast_branches.get_file_index(frame.file)
        if file_index is None:
            continue

        scope = _resolve_scope_with_decorator_fallback(file_index, frame)
        if scope is None:
            continue

        for conditional_key, branch_kind, _depth in scope.query_intervals(
            frame.line_number,
            frame.col_offset,
        ):
            entry = (conditional_key, branch_kind)
            if not fallback_branch_stack or fallback_branch_stack[-1] != entry:
                fallback_branch_stack.append(entry)

    return fallback_branch_stack


def _resolve_scope_with_decorator_fallback(
    file_index: ast_branches.FileIndex,
    frame: "FuncCallLocation",
) -> Optional[ast_branches.ScopeEntry]:
    """Resolve a frame, tolerating decorator-line ``co_firstlineno`` offsets.

    Parameters
    ----------
    file_index:
        AST index for the frame's source file.
    frame:
        Runtime frame metadata captured in ``FuncCallLocation`` form.

    Returns
    -------
    Optional[ast_branches.ScopeEntry]
        Resolved scope entry, or ``None`` when the fallback still fails closed.
    """

    resolved_scope = file_index.resolve_scope(
        code_firstlineno=frame.code_firstlineno,
        func_name=frame.func_name,
        code_qualname=frame.code_qualname,
    )
    if resolved_scope is not None:
        return resolved_scope

    candidate_firstlineno = frame.code_firstlineno + 1
    if frame.code_qualname is not None:
        qualname_matches = [
            scope
            for scope in file_index.scopes
            if scope.qualname == frame.code_qualname
            and scope.code_firstlineno == candidate_firstlineno
        ]
        if len(qualname_matches) == 1:
            return qualname_matches[0]
        return None

    name_matches = [
        scope
        for scope in file_index.scopes
        if scope.func_name == frame.func_name and scope.code_firstlineno == candidate_firstlineno
    ]
    if len(name_matches) == 1:
        return name_matches[0]
    return None


def _get_gained_branch_entries(
    parent_stack: List[Tuple[int, str]],
    child_stack: List[Tuple[int, str]],
) -> List[Tuple[int, str]]:
    """Return child stack entries gained across one forward edge.

    Parameters
    ----------
    parent_stack:
        Parent operation branch stack, ordered outer-to-inner.
    child_stack:
        Child operation branch stack, ordered outer-to-inner.

    Returns
    -------
    List[Tuple[int, str]]
        Entries present in the child's stack beyond the shared prefix with the
        parent, preserving outer-to-inner order.
    """

    shared_prefix_len = 0
    max_shared = min(len(parent_stack), len(child_stack))
    while shared_prefix_len < max_shared:
        if parent_stack[shared_prefix_len] != child_stack[shared_prefix_len]:
            break
        shared_prefix_len += 1
    return child_stack[shared_prefix_len:]


def _fix_modules_for_internal_tensors(self: "Trace") -> None:
    """Step 6: Append module-path suffix to equivalence_class.

    Module containment for every captured op is set at op-creation time by
    the wrap-forward stack helper (see torchlens.decoration._module_stack).
    This step's only remaining job is appending the canonical module-path
    suffix to ``equivalence_class`` so loop detection (Step 8) treats
    same-function ops in different modules as distinct equivalence types.

    It is legitimate for an op to have empty ``modules`` here (input,
    output, buffer, root-module, post-hook ops, orphan nodes). Such ops
    receive an empty suffix.
    """
    module_str_cache: dict[tuple, str] = {}
    for layer in self:
        if getattr(layer, "is_orphan", False):
            continue
        cm_key = tuple(layer.modules)
        if cm_key not in module_str_cache:
            module_str_cache[cm_key] = "_".join(module_pass[0] for module_pass in layer.modules)
        layer.equivalence_class += module_str_cache[cm_key]


def _fix_buffer_layers(self: "Trace") -> None:
    """Step 7: Connect buffer parents, merge duplicates, and assign pass numbers.

    Buffer tensors (nn.Module registered buffers) are logged as source tensors
    during the forward pass but may lack proper parent connections. This function:

    1. Connects each buffer to its buffer_parent (the tensor that produced the
       buffer's value), updating parent/child links and ancestry.
    2. Deduplicates buffers: buffers with the same containing module, same parent,
       same buffer_address, AND same tensor value are merged into a single node.
       The dedup hash is (modules + buffer_parent + buffer_address).
    3. Assigns sequential buffer_pass numbers per buffer_address.

    Note: Buffer siblings are always empty — the sibling iteration in
    _merge_buffer_entries is effectively dead code for buffers (#2).
    """
    buffer_counter: Dict[str, int] = defaultdict(lambda: 1)
    buffer_hash_groups: Dict[str, List[str]] = defaultdict(list)

    for layer_label in self.buffer_layers:
        layer = self[layer_label]
        if layer.buffer_parent is not None:
            layer.parents.append(layer.buffer_parent)
            self[layer.buffer_parent].children.append(layer_label)
            self[layer.buffer_parent].has_children = True
            layer.func = identity
            layer.func_name = "identity"
            layer.has_input_ancestor = True
            layer.input_ancestors.update(self[layer.buffer_parent].input_ancestors)
            layer.root_ancestors.remove(layer._label_raw)
            layer.root_ancestors.update(self[layer.buffer_parent].root_ancestors)
            layer.parent_arg_positions["args"][0] = layer.buffer_parent
            if (self[layer.buffer_parent].out is not None) and (layer.saved_args is not None):
                layer.saved_args.append(
                    safe_copy(self[layer.buffer_parent].out, detach_tensor=True)
                )

        buffer_hash = str(layer.modules) + str(layer.buffer_parent) + layer.buffer_address
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
                    (buffer.out is not None)
                    and (unique_buffer.out is not None)
                    and (torch.equal(buffer.out, unique_buffer.out))
                ):
                    _merge_buffer_entries(self, unique_buffer, buffer)
                    break
            else:
                unique_buffers.append(buffer_label)

    # And relabel the buffer ops.

    for layer_label in self.buffer_layers:
        layer = self[layer_label]
        buffer_address = layer.buffer_address
        layer.buffer_pass = buffer_counter[buffer_address]
        self.buffer_num_calls[buffer_address] = buffer_counter[buffer_address]
        buffer_counter[buffer_address] += 1


def _merge_buffer_entries(self: "Trace", source_buffer: OpLog, buffer_to_remove: OpLog) -> None:
    """Merge a duplicate buffer into a source buffer, rewiring all edges.

    Transfers all child and parent connections from ``buffer_to_remove`` to
    ``source_buffer``, updates parent_arg_positions in children to point to
    the source buffer, fixes internal_source_parents/ancestors references
    across the graph, and removes the duplicate from the layer dict.
    """
    for child_layer in buffer_to_remove.children:
        if child_layer not in source_buffer.children:
            source_buffer.children.append(child_layer)
        self[child_layer].parents.remove(buffer_to_remove._label_raw)
        self[child_layer].parents.append(source_buffer._label_raw)
        if buffer_to_remove._label_raw in self[child_layer].internal_source_parents:
            self[child_layer].internal_source_parents.remove(buffer_to_remove._label_raw)
            self[child_layer].internal_source_parents.append(source_buffer._label_raw)

        for arg_type in ["args", "kwargs"]:
            for arg_label, arg_val in self[child_layer].parent_arg_positions[arg_type].items():
                if arg_val == buffer_to_remove._label_raw:
                    self[child_layer].parent_arg_positions[arg_type][arg_label] = (
                        source_buffer._label_raw
                    )

    for parent_layer in buffer_to_remove.parents:
        if parent_layer not in source_buffer.parents:
            source_buffer.parents.append(parent_layer)
        self[parent_layer].children.remove(buffer_to_remove._label_raw)
        self[parent_layer].children.append(source_buffer._label_raw)

    for parent_layer in buffer_to_remove.internal_source_parents:
        if parent_layer not in source_buffer.internal_source_parents:
            source_buffer.internal_source_parents.append(parent_layer)

    self._raw_layer_labels_list.remove(buffer_to_remove._label_raw)
    self._raw_layer_dict.pop(buffer_to_remove._label_raw)

    for layer in self:
        if buffer_to_remove._label_raw in layer.root_ancestors:
            layer.root_ancestors.remove(buffer_to_remove._label_raw)
            layer.root_ancestors.add(source_buffer._label_raw)
        if buffer_to_remove._label_raw in layer.internal_source_ancestors:
            layer.internal_source_ancestors.remove(buffer_to_remove._label_raw)
            layer.internal_source_ancestors.add(source_buffer._label_raw)

    self._remove_log_entry(buffer_to_remove, remove_references=True)
