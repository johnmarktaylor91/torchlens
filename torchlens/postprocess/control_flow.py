"""Steps 5-7: Conditional branches, module annotation fixes, buffer layer fixes.

Step 5 (_mark_conditional_branches) now runs a six-phase conditional pipeline:
    5a. Build AST file indexes for files referenced by terminal scalar bools.
    5b. Classify terminal bools into branch/non-branch contexts.
    5c. Materialize dense conditional events from structural AST keys.
    5d. Backward-flood IF edges from branch-participating bools only.
    5e. Attribute executed ops to THEN/ELIF/ELSE arms across every forward edge.
    5f. Materialize derived compatibility views from the new primary structures.
Step 6 (_fix_modules_for_internal_tensors): Internally-generated tensors
    (constants, arange, etc.) don't know their containing module. This infers
    module containment by tracing backward from known input-descendant tensors.
    Also appends module path suffixes to operation_equivalence_type — this is
    INTENTIONAL and affects loop detection (Step 8), ensuring operations in
    different modules are never grouped as the same layer.
Step 7 (_fix_buffer_layers): Connects buffer parents, deduplicates identical
    buffers (same module, same value, same parent), and assigns buffer pass numbers.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import chain
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

import torch

from ..data_classes.layer_pass_log import LayerPassLog
from ..utils.display import identity
from ..utils.tensor_utils import safe_copy
from . import ast_branches

if TYPE_CHECKING:
    from ..data_classes.func_call_location import FuncCallLocation
    from ..data_classes.model_log import ConditionalEvent, ModelLog


_BRANCH_CONTEXT_KINDS = frozenset({"if_test", "elif_test", "ifexp"})


def _mark_conditional_branches(self: "ModelLog") -> None:
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
    ``attribute_op()`` work. All ModelLog-level conditional collections are
    already initialized empty in :meth:`ModelLog.__init__`, and per-layer
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
    events_by_key = _materialize_conditional_events(
        self,
        file_indexes,
        conditional_keys,
        bool_classifications,
    )
    _mark_conditional_branches_if_backward_flood(self, bool_classifications)
    _attribute_branches_forward(self, events_by_key)
    _materialize_derived_views(self)


def _can_fast_skip_step5(self: "ModelLog") -> bool:
    """Return True when Step 5 has no work to do.

    The slow path's only branch-attributing inputs are the ModelLog's
    ``internally_terminated_bool_layers``: if no terminal scalar bool was
    captured, ``_iter_terminal_scalar_bool_labels`` yields nothing, so
    every downstream collection (events, edges, per-layer arm children)
    would resolve to its empty default. Skipping the slow path is then
    semantically equivalent to running it.

    The function also checks the ModelLog-level conditional collections
    (``conditional_events``, ``conditional_branch_edges``,
    ``conditional_arm_edges``, ``conditional_edge_passes``). They are initialized empty in
    :meth:`ModelLog.__init__`, and the slow path resets them on entry.
    Any caller that pre-populated these would change the user-visible
    output if we skipped, so we conservatively run the slow path in that
    (pathological) case as well.
    """

    if self.internally_terminated_bool_layers:
        return False
    if self.conditional_events:
        return False
    if self.conditional_branch_edges:
        return False
    if self.conditional_arm_edges:
        return False
    if self.conditional_edge_passes:
        return False
    return True


def _build_file_indexes(
    self: "ModelLog",
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
        for frame in bool_layer.func_call_stack:
            if frame.file in file_indexes:
                continue
            file_indexes[frame.file] = ast_branches.get_file_index(frame.file)
    return file_indexes


def _classify_bool_layers(
    self: "ModelLog",
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
        for frame in reversed(bool_layer.func_call_stack):
            frame_classification = ast_branches.classify_bool(
                frame.file,
                frame.line_number,
                frame.col_offset,
            )
            if frame_classification.kind == "unknown":
                continue
            classification = frame_classification
            break

        bool_is_branch = (
            classification.kind in _BRANCH_CONTEXT_KINDS
            and classification.conditional_key is not None
        )
        bool_layer.bool_context_kind = classification.kind
        bool_layer.bool_wrapper_kind = classification.wrapper_kind
        bool_layer.bool_is_branch = bool_is_branch
        bool_layer.bool_conditional_id = None
        bool_classifications[bool_label] = classification

        if bool_is_branch:
            conditional_key = classification.conditional_key
            if conditional_key is None:
                raise ValueError("Branch-participating bool classification must include a key.")
            assert conditional_key is not None  # mypy narrowing
            ordered_conditional_keys.setdefault(conditional_key, None)

    return list(ordered_conditional_keys.keys()), bool_classifications


def _materialize_conditional_events(
    self: "ModelLog",
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
    self.conditional_events = []

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
            nesting_depth=record.nesting_depth,
            parent_conditional_id=None,
            parent_branch_kind=record.parent_branch_kind,
        )
        events_by_key[conditional_key] = event
        self.conditional_events.append(event)

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
            bool_layer.bool_conditional_id = None
            continue
        event = events_by_key[bool_conditional_key]
        bool_layer.bool_conditional_id = event.id
        event.bool_layers.append(bool_label)

    for bool_label in _iter_terminal_scalar_bool_labels(self):
        assert not hasattr(self[bool_label], "_bool_conditional_key")

    return events_by_key


def _mark_conditional_branches_if_backward_flood(
    self: "ModelLog",
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
        layer.cond_branch_start_children = []
        layer.in_cond_branch = False

    branch_bool_labels = [
        bool_label
        for bool_label in _iter_terminal_scalar_bool_labels(self)
        if bool_classifications[bool_label].conditional_key is not None
        and self[bool_label].bool_is_branch
    ]

    nodes_seen: Set[str] = set()
    node_stack = branch_bool_labels.copy()
    while node_stack:
        node_label = node_stack.pop()
        node = self[node_label]
        if node_label in nodes_seen:
            continue
        for parent_label in node.parent_layers:
            parent_layer = self[parent_label]
            if parent_layer.is_output_ancestor:
                parent_layer.cond_branch_start_children.append(node_label)
                parent_layer.in_cond_branch = False
                nodes_seen.add(parent_label)
                self.conditional_branch_edges.append((parent_label, node_label))
            else:
                if parent_label in nodes_seen:
                    continue
                parent_layer.in_cond_branch = True
                node_stack.append(parent_label)

        nodes_seen.add(node_label)


def _attribute_branches_forward(
    self: "ModelLog",
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

    conditional_arm_edges: Dict[Tuple[int, str], List[Tuple[str, str]]] = defaultdict(list)
    conditional_edge_passes: Dict[Tuple[str, str, int, str], List[int]] = defaultdict(list)

    for layer_label in self._raw_layer_labels_list:
        layer = self[layer_label]
        layer.conditional_branch_stack = _translate_conditional_stack(
            layer.func_call_stack,
            events_by_key,
        )
        layer.conditional_branch_depth = len(layer.conditional_branch_stack)
        layer.cond_branch_children_by_cond = {}

    for parent_label in self._raw_layer_labels_list:
        parent_layer = self[parent_label]
        for child_label in parent_layer.child_layers:
            child_layer = self[child_label]
            gained_entries = _get_gained_branch_entries(
                parent_layer.conditional_branch_stack,
                child_layer.conditional_branch_stack,
            )
            for conditional_id, branch_kind in gained_entries:
                parent_layer.cond_branch_children_by_cond.setdefault(conditional_id, {}).setdefault(
                    branch_kind, []
                ).append(child_label)
                conditional_arm_edges[(conditional_id, branch_kind)].append(
                    (parent_layer.tensor_label_raw, child_layer.tensor_label_raw)
                )
                conditional_edge_passes[
                    (
                        parent_layer.tensor_label_raw.split(":", 1)[0],
                        child_layer.tensor_label_raw.split(":", 1)[0],
                        conditional_id,
                        branch_kind,
                    )
                ].append(parent_layer.pass_num)

    self.conditional_arm_edges = dict(conditional_arm_edges)
    self.conditional_edge_passes = dict(conditional_edge_passes)


def _materialize_derived_views(self: "ModelLog") -> None:
    """Phase 5f: Rebuild compatibility views derived from primary conditional data.

    Parameters
    ----------
    self:
        Model log being postprocessed.
    """

    self.conditional_edge_passes = {
        key: sorted(set(pass_nums)) for key, pass_nums in self.conditional_edge_passes.items()
    }

    for layer_label in self._raw_layer_labels_list:
        layer = self[layer_label]
        layer.cond_branch_then_children = sorted(
            set(
                chain.from_iterable(
                    branch_children.get("then", [])
                    for branch_children in layer.cond_branch_children_by_cond.values()
                )
            )
        )

        elif_children: Dict[int, Set[str]] = defaultdict(set)
        for branch_children in layer.cond_branch_children_by_cond.values():
            for branch_kind, child_labels in branch_children.items():
                if not branch_kind.startswith("elif_"):
                    continue
                elif_index = int(branch_kind.split("_", 1)[1])
                elif_children[elif_index].update(child_labels)
        layer.cond_branch_elif_children = {
            elif_index: sorted(child_labels)
            for elif_index, child_labels in sorted(elif_children.items())
        }

        layer.cond_branch_else_children = sorted(
            set(
                chain.from_iterable(
                    branch_children.get("else", [])
                    for branch_children in layer.cond_branch_children_by_cond.values()
                )
            )
        )


def _iter_terminal_scalar_bool_labels(self: "ModelLog") -> List[str]:
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

    terminal_bool_labels = set(self.internally_terminated_bool_layers)
    return [
        layer_label
        for layer_label in self._raw_layer_labels_list
        if layer_label in terminal_bool_labels and self[layer_label].is_scalar_bool
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
    func_call_stack: List["FuncCallLocation"],
    events_by_key: Dict[ast_branches.ConditionalKey, "ConditionalEvent"],
) -> List[Tuple[int, str]]:
    """Translate a structural AST branch stack into dense conditional IDs.

    Parameters
    ----------
    func_call_stack:
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
    for conditional_key, branch_kind in _attribute_op_with_scope_fallback(func_call_stack):
        if conditional_key not in events_by_key:
            continue
        translated_stack.append((events_by_key[conditional_key].id, branch_kind))
    return translated_stack


def _attribute_op_with_scope_fallback(
    func_call_stack: List["FuncCallLocation"],
) -> List[Tuple[ast_branches.ConditionalKey, str]]:
    """Attribute an op, retrying decorated-function scope resolution when needed.

    Parameters
    ----------
    func_call_stack:
        Captured runtime call stack for one operation.

    Returns
    -------
    List[Tuple[ast_branches.ConditionalKey, str]]
        Structural ``(conditional_key, branch_kind)`` pairs ordered
        outer-to-inner.
    """

    branch_stack = ast_branches.attribute_op(func_call_stack)
    if branch_stack:
        return branch_stack

    fallback_branch_stack: List[Tuple[ast_branches.ConditionalKey, str]] = []
    for frame in func_call_stack:
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


def _fix_modules_for_internal_tensors(self: "ModelLog") -> None:
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


def _fix_buffer_layers(self: "ModelLog") -> None:
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
                layer.captured_args.append(
                    safe_copy(self[layer.buffer_parent].activation, detach_tensor=True)
                )

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
    self: "ModelLog", source_buffer: LayerPassLog, buffer_to_remove: LayerPassLog
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
