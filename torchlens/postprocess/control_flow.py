"""Steps 5-6: Conditional branches and buffer layer fixes.

Step 5 (_mark_conditional_branches) runs a six-phase conditional pipeline:
    5a. Build AST file indexes for files referenced by terminal scalar bools.
    5b. Classify terminal bools into branch/non-branch contexts.
    5c. Materialize dense conditional events from structural AST keys.
    5d. Backward-flood IF edges from branch-participating bools only.
    5e. Attribute executed ops to THEN/ELIF/ELSE arms across every forward edge.
    5f. Materialize derived compatibility views from primary structures.
Step 6 (_fix_buffer_layers): Connects buffer sources, deduplicates identical
    buffers (same module, same value, same parent), and assigns buffer pass numbers.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import chain
from typing import TYPE_CHECKING, Any

import torch

from .._state import pause_logging
from ..data_classes.op import Op
from ..utils.display import identity
from ..utils.tensor_utils import safe_copy
from . import ast_branches

if TYPE_CHECKING:
    from ..data_classes.func_call_location import FuncCallLocation
    from ..data_classes.trace import ConditionalEvent, Trace


_BRANCH_CONTEXT_KINDS = frozenset({"if_test", "elif_test", "ifexp"})


def _mark_conditional_branches(self: Trace) -> None:
    """Step 5: Classify bools, materialize events, and attribute conditional edges.

    The public Step 5 entry point delegates to six internal phases:

    1. Build AST file indexes for all files touched by terminal scalar bools.
    2. Classify terminal bools and collect observed structural conditional keys.
    3. Materialize dense ``ConditionalEvent`` records from those keys.
    4. Backward-flood IF edges from branch-participating bools only.
    5. Attribute executed ops and forward edges to conditional branch arms.
    6. Rebuild compatibility views derived from primary structures.

    Performance fast-path: when no terminal scalar bools were captured, the
    model has no conditional branches the pipeline can attribute, so we skip
    the AST file indexing, per-bool classification, and per-op
    ``attribute_op()`` work. All Trace-level conditional collections are
    already initialized empty in :meth:`Trace.__init__`, and per-layer
    conditional fields are initialized to their empty defaults during
    capture (see ``capture/output_tensors.py``). The fast-path is verified
    against the slow-path defaults via ``tests/test_perf_bundle.py``.
    """

    _seed_proven_bool_consumers(self)
    if _can_fast_skip_step5(self):
        return

    file_indexes = _build_file_indexes(self)
    conditional_keys, bool_classifications = _classify_bool_layers(self)
    # Defensive guard: if no terminal bool produced a structural conditional
    # key, attribution will produce zero edges, matching the fast-skip output.
    # This invariant makes bool-detector drift fail explicitly instead of
    # silently making the fast-path miss work.
    if not bool_classifications and conditional_keys:
        # A real raise, not an assert: the whole point of this guard is to make
        # bool-detector drift fail EXPLICITLY, and `python -O` strips asserts --
        # which would restore exactly the silent fast-path miss it exists to
        # prevent.
        raise RuntimeError(
            "Internally-terminated bool layers were absent but "
            f"{len(conditional_keys)} conditional key(s) were produced; the "
            "fast-skip precondition is stale. This means the bool detector and "
            "the conditional-key builder disagree, so conditional attribution "
            "would silently diverge from the fast-path output."
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


def _seed_proven_bool_consumers(self: Trace) -> None:
    """Add captured tensor-to-host bool consumers to Step 5's candidate list.

    Parameters
    ----------
    self:
        Trace being postprocessed.

    Notes
    -----
    A predicate returned by the model has a synthetic output child, so it is not
    an internal graph sink. The capture-time ``__bool__`` observer independently
    proves that the tensor was consumed on the host; that proof, rather than
    childlessness, makes it a terminal conditional candidate.

    Seeding stays gated on ``is_scalar_bool`` (0-dim ``torch.bool``): the
    runnable witness-obligation registry can only witness scalar-bool
    predicates, so seeding a proven non-bool truthiness consumer (or a
    one-element bool VECTOR) would materialize conditional arm edges that
    every level="runnable" save refuses at producer preflight. Recording
    those classes is deferred until the runnable contract gains a matching
    predicate witness family.
    """

    from ..backends.torch.completeness_witness import host_escape_bool_source_labels

    proven_labels = host_escape_bool_source_labels(self)
    # Shadow set over the list[str] ledger: the per-label list scan was
    # O(k^2) in terminated-bool count (hunt-6 R52-2 sibling site).
    seen_terminated_bool_labels = set(self.internally_terminated_bool_ops)
    for label in self._raw_graph_ws.raw_layer_labels_list:
        if label not in proven_labels:
            continue
        layer = self[label]
        if not layer.is_scalar_bool or getattr(layer, "is_orphan", False):
            continue
        if label not in seen_terminated_bool_labels:
            seen_terminated_bool_labels.add(label)
            self.internally_terminated_bool_ops.append(label)
        layer.is_terminal_bool = True


def _can_fast_skip_step5(self: Trace) -> bool:
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
    return not self.conditional_edge_call_indices


def _build_file_indexes(
    self: Trace,
) -> dict[str, ast_branches.FileIndex | None]:
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

    from ..backends.torch.completeness_witness import host_escape_bool_consumer_locations

    file_indexes: dict[str, ast_branches.FileIndex | None] = {}
    consumer_locations = host_escape_bool_consumer_locations(self)
    for bool_label in _iter_terminal_scalar_bool_labels(self):
        bool_layer = self[bool_label]
        for filename, _line_number in consumer_locations.get(bool_label, ()):
            if filename not in file_indexes:
                file_indexes[filename] = ast_branches.get_file_index(filename)
        for frame in bool_layer.code_context:
            if frame.file in file_indexes:
                continue
            file_indexes[frame.file] = ast_branches.get_file_index(frame.file)
    return file_indexes


def _classify_bool_layers(
    self: Trace,
) -> tuple[list[ast_branches.ConditionalKey], dict[str, list[ast_branches.BoolClassification]]]:
    """Phase 5b: Classify terminal scalar bools and collect observed conditionals.

    Every witnessed consumer location of a bool is classified — not just the
    first non-``"unknown"`` one. A bool consumed by ``assert``/``while`` and
    LATER by an ``if`` test is still a conditional bool (order independence),
    and one bool gating several ``if`` statements yields one classification
    per gated conditional (1:N predicate reuse). The runtime frame fallback
    (column-precise) runs only when no consumer location produced a
    branch-participating classification.

    Parameters
    ----------
    self:
        Model log being postprocessed.

    Returns
    -------
    Tuple[List[ast_branches.ConditionalKey], Dict[str, List[ast_branches.BoolClassification]]]
        First-seen ordered conditional keys plus, per raw bool layer label, the
        deduplicated list of branch-participating classifications (empty when
        the bool participates in no conditional).
    """

    from ..backends.torch.completeness_witness import host_escape_bool_consumer_locations

    bool_classifications: dict[str, list[ast_branches.BoolClassification]] = {}
    ordered_conditional_keys: dict[ast_branches.ConditionalKey, None] = {}
    consumer_locations = host_escape_bool_consumer_locations(self)

    for bool_label in _iter_terminal_scalar_bool_labels(self):
        bool_layer = self[bool_label]
        observed: list[ast_branches.BoolClassification] = []
        for filename, line_number in consumer_locations.get(bool_label, ()):
            location_classification = ast_branches.classify_bool(filename, line_number, None)
            if location_classification.kind != "unknown":
                observed.append(location_classification)

        branch_classifications = _dedup_branch_classifications(observed)

        frame_classification: ast_branches.BoolClassification | None = None
        for frame in reversed(bool_layer.code_context):
            frame_candidate = ast_branches.classify_bool(
                frame.file,
                frame.line_number,
                frame.col_offset,
            )
            if frame_candidate.kind == "unknown":
                continue
            frame_classification = frame_candidate
            break

        if not branch_classifications:
            if frame_classification is not None:
                observed.append(frame_classification)
                branch_classifications = _dedup_branch_classifications([frame_classification])
        elif (
            frame_classification is not None
            and frame_classification.kind in _BRANCH_CONTEXT_KINDS
            and frame_classification.conditional_key is not None
            and frame_classification.conditional_key
            not in {c.conditional_key for c in branch_classifications}
        ):
            # Creation-site conflict: the bool op was created inside the test
            # span of one conditional while the witnessed consumer line
            # attributes it to a DIFFERENT conditional. Line-only runtime
            # attribution is misreporting one of the two (e.g. a formatter-
            # wrapped multi-line nested ternary, where the interpreter
            # reports the inner ternary's line for the outer test's
            # ``__bool__``). Linking either key could cross-wire a foreign
            # bool into a conditional's public record, so fail closed for
            # this bool instead of guessing.
            observed = []
            branch_classifications = []

        if branch_classifications:
            primary = branch_classifications[0]
        elif observed:
            primary = observed[0]
        else:
            primary = ast_branches.BoolClassification("unknown", None, None, None)

        bool_layer.conditional_context_kind = primary.kind
        bool_layer.conditional_wrapper_kind = primary.wrapper_kind
        bool_layer.is_terminal_conditional_bool = bool(branch_classifications)
        bool_layer.terminal_conditional_id = None
        bool_classifications[bool_label] = branch_classifications

        for classification in branch_classifications:
            assert classification.conditional_key is not None  # mypy narrowing
            ordered_conditional_keys.setdefault(classification.conditional_key, None)

    return list(ordered_conditional_keys.keys()), bool_classifications


def _dedup_branch_classifications(
    classifications: list[ast_branches.BoolClassification],
) -> list[ast_branches.BoolClassification]:
    """Return the branch-participating classifications, deduplicated in order.

    Parameters
    ----------
    classifications:
        Classification results from consumer locations or runtime frames.

    Returns
    -------
    List[ast_branches.BoolClassification]
        Classifications whose kind is branch-participating and whose
        conditional key is present, deduplicated by
        ``(conditional_key, branch_test_kind)`` preserving first-seen order.
    """

    deduplicated: list[ast_branches.BoolClassification] = []
    seen: set[tuple[ast_branches.ConditionalKey, str | None]] = set()
    for classification in classifications:
        if (
            classification.kind not in _BRANCH_CONTEXT_KINDS
            or classification.conditional_key is None
        ):
            continue
        identity_key = (classification.conditional_key, classification.branch_test_kind)
        if identity_key in seen:
            continue
        seen.add(identity_key)
        deduplicated.append(classification)
    return deduplicated


def _materialize_conditional_records(
    self: Trace,
    file_indexes: dict[str, ast_branches.FileIndex | None],
    conditional_keys: list[ast_branches.ConditionalKey],
    bool_classifications: dict[str, list[ast_branches.BoolClassification]],
) -> dict[ast_branches.ConditionalKey, ConditionalEvent]:
    """Phase 5c: Materialize dense conditional events and translate bool keys.

    One bool may gate several conditionals (predicate reuse), so every
    branch-participating classification links its bool to the matching event.
    The scalar ``terminal_conditional_id`` keeps its historical 1:1 shape by
    pointing at the FIRST linked event; the complete 1:N linkage lives on each
    event's ``bool_layers``. Three postprocess-internal annotations are stashed
    on each event for finalization's public record builder:
    ``_arm_bool_indices`` (branch kind -> indices into ``bool_layers`` whose
    runtime consumption evaluated THAT arm's test; indices survive the
    raw-to-final label rename that rewrites ``bool_layers`` in place),
    ``_arm_test_structures`` (branch kind -> ``"bare"``/``"negated"``/
    ``"compound"`` bool-value semantics of the arm's test expression), and
    ``_bool_layers_raw`` (index-aligned RAW labels so finalization resolves
    the exact per-pass evaluating op of rolled multi-pass bools).

    Parameters
    ----------
    self:
        Model log being postprocessed.
    file_indexes:
        Cached AST file indexes from phase 5a.
    conditional_keys:
        Ordered structural conditional keys observed in phase 5b.
    bool_classifications:
        Per-bool branch-participating classifications keyed by raw layer label.

    Returns
    -------
    Dict[ast_branches.ConditionalKey, ConditionalEvent]
        Mapping from structural conditional key to the dense event object.
    """

    from ..data_classes.trace import ConditionalEvent

    record_lookup = _build_conditional_record_lookup(file_indexes)
    self.conditional_records = []

    events_by_key: dict[ast_branches.ConditionalKey, ConditionalEvent] = {}
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
            # Copies, never aliases: the record lives in ast_branches' process-
            # global file cache, so handing its dicts to the event verbatim let
            # any caller mutating trace metadata poison every later capture of
            # the same file (fw3settle: invariant-7's elif-key edit surfaced as
            # a clean-capture invariant failure in a later test). Span values
            # are immutable tuples, so a shallow copy is a full fence.
            branch_ranges=dict(record.branch_ranges),
            branch_test_spans=dict(record.branch_test_spans),
            call_depth=record.call_depth,
            parent_conditional_id=None,
            parent_branch_kind=record.parent_branch_kind,
        )
        # Postprocess-internal annotations consumed by finalization's
        # ``_build_conditional_records``; instance attributes (not dataclass
        # fields) so the portable/public ConditionalEvent schema is unchanged.
        # ``_bool_layers_raw`` mirrors ``bool_layers`` with RAW labels: raw
        # labels are unique per pass and stay valid ``layer_dict_all_keys``
        # lookup keys, so finalization can resolve the EXACT evaluating op of
        # a rolled multi-pass bool layer. The renamed public ``bool_layers``
        # collapses rolled passes onto one base label, and an unqualified
        # lookup resolves last-writer-wins to an arbitrary pass (round-24
        # condbranch seal, S3).
        setattr(event, "_arm_bool_indices", {})
        setattr(event, "_arm_test_structures", dict(record.branch_test_structures))
        setattr(event, "_bool_layers_raw", [])
        events_by_key[conditional_key] = event
        self.conditional_records.append(event)

    for conditional_key in conditional_keys:
        record, _function_qualname = record_lookup[conditional_key]
        event = events_by_key[conditional_key]
        parent_conditional_key = record.parent_conditional_key
        if parent_conditional_key is not None and parent_conditional_key in events_by_key:
            event.parent_conditional_id = events_by_key[parent_conditional_key].id

    # Position/membership shadows (r8 R60-6): the list `not in` / `.index()`
    # trio made this loop O(k^2) per conditional over its terminal bool ops
    # (a bool inside a hot unrolled loop shares one structural key). The
    # shadows keep `bool_layers`/`_arm_bool_indices` byte-identical --
    # first-occurrence index, insertion order preserved.
    bool_positions_by_event: dict[int, dict[str, int]] = {}
    arm_seen_by_event: dict[int, dict[str, set[int]]] = {}
    for bool_label, classifications in bool_classifications.items():
        bool_layer = self[bool_label]
        bool_layer.terminal_conditional_id = None
        for classification in classifications:
            bool_conditional_key: ast_branches.ConditionalKey | None = (
                classification.conditional_key
            )
            if bool_conditional_key is None or bool_conditional_key not in events_by_key:
                continue
            event = events_by_key[bool_conditional_key]
            if bool_layer.terminal_conditional_id is None:
                bool_layer.terminal_conditional_id = event.id
            positions = bool_positions_by_event.get(event.id)
            if positions is None:
                positions = {}
                for index, label in enumerate(event.bool_layers):
                    positions.setdefault(label, index)
                bool_positions_by_event[event.id] = positions
            bool_index = positions.get(bool_label)
            if bool_index is None:
                bool_index = len(event.bool_layers)
                positions[bool_label] = bool_index
                event.bool_layers.append(bool_label)
                getattr(event, "_bool_layers_raw").append(bool_label)
            arm_kind = classification.branch_test_kind or "then"
            arm_bool_indices: dict[str, list[int]] = getattr(event, "_arm_bool_indices")
            arm_indices = arm_bool_indices.setdefault(arm_kind, [])
            arm_seen = arm_seen_by_event.setdefault(event.id, {}).get(arm_kind)
            if arm_seen is None:
                arm_seen = set(arm_indices)
                arm_seen_by_event[event.id][arm_kind] = arm_seen
            if bool_index not in arm_seen:
                arm_seen.add(bool_index)
                arm_indices.append(bool_index)

    for bool_label in _iter_terminal_scalar_bool_labels(self):
        assert not hasattr(self[bool_label], "_bool_conditional_key")

    return events_by_key


def _mark_conditional_branches_if_backward_flood(
    self: Trace,
    bool_classifications: dict[str, list[ast_branches.BoolClassification]],
) -> None:
    """Phase 5d: Backward-flood IF edges from branch-participating bools only.

    Parameters
    ----------
    self:
        Model log being postprocessed.
    bool_classifications:
        Per-bool branch-participating classifications keyed by raw layer label.
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
        if bool_classifications[bool_label] and self[bool_label].is_terminal_conditional_bool
    ]

    nodes_seen: set[str] = set()
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
    self: Trace,
    events_by_key: dict[ast_branches.ConditionalKey, ConditionalEvent],
) -> None:
    """Phase 5e: Attribute executed ops and forward edges to conditional arms.

    Parameters
    ----------
    self:
        Model log being postprocessed.
    events_by_key:
        Structural-to-dense conditional event lookup created in phase 5c.
    """

    conditional_arm_entry_edges: dict[tuple[int, str], list[tuple[str, str]]] = defaultdict(list)
    conditional_edge_call_indices: dict[tuple[str, str, int, str], list[int]] = defaultdict(list)

    for layer_label in self._raw_graph_ws.raw_layer_labels_list:
        layer = self[layer_label]
        if getattr(layer, "is_orphan", False):
            continue
        layer.conditional_branch_stack = _translate_conditional_stack(
            layer.code_context,
            events_by_key,
        )
        layer.conditional_branch_depth = len(layer.conditional_branch_stack)
        layer.conditional_arm_children = {}

    for parent_label in self._raw_graph_ws.raw_layer_labels_list:
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
                ].append(child_layer.pass_index)

    self.conditional_arm_entry_edges = dict(conditional_arm_entry_edges)
    self.conditional_edge_call_indices = dict(conditional_edge_call_indices)


def _materialize_derived_views(self: Trace) -> None:
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

    for layer_label in self._raw_graph_ws.raw_layer_labels_list:
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

        elif_children: dict[int, set[str]] = defaultdict(set)
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


def _iter_terminal_scalar_bool_labels(self: Trace) -> list[str]:
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
        for layer_label in self._raw_graph_ws.raw_layer_labels_list
        if layer_label in terminal_bool_labels
        and self[layer_label].is_scalar_bool
        and not getattr(self[layer_label], "is_orphan", False)
    ]


def _build_conditional_record_lookup(
    file_indexes: dict[str, ast_branches.FileIndex | None],
) -> dict[ast_branches.ConditionalKey, tuple[ast_branches.ConditionalRecord, str]]:
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

    record_lookup: dict[
        ast_branches.ConditionalKey, tuple[ast_branches.ConditionalRecord, str]
    ] = {}
    for file_index in file_indexes.values():
        if file_index is None:
            continue
        for scope in file_index.scopes:
            for conditional_record in scope.conditionals:
                record_lookup[conditional_record.key] = (conditional_record, scope.qualname)
    return record_lookup


def _translate_conditional_stack(
    code_context: list[FuncCallLocation],
    events_by_key: dict[ast_branches.ConditionalKey, ConditionalEvent],
) -> list[tuple[int, str]]:
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

    translated_stack: list[tuple[int, str]] = []
    for conditional_key, branch_kind in _attribute_op_with_scope_fallback(code_context):
        if conditional_key not in events_by_key:
            continue
        translated_stack.append((events_by_key[conditional_key].id, branch_kind))
    return translated_stack


def _attribute_op_with_scope_fallback(
    code_context: list[FuncCallLocation],
) -> list[tuple[ast_branches.ConditionalKey, str]]:
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

    fallback_branch_stack: list[tuple[ast_branches.ConditionalKey, str]] = []
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
    frame: FuncCallLocation,
) -> ast_branches.ScopeEntry | None:
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
        func_qualname=frame.func_qualname,
    )
    if resolved_scope is not None:
        return resolved_scope

    candidate_firstlineno = frame.code_firstlineno + 1
    if frame.func_qualname is not None:
        qualname_matches = [
            scope
            for scope in file_index.scopes
            if scope.qualname == frame.func_qualname
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
    parent_stack: list[tuple[int, str]],
    child_stack: list[tuple[int, str]],
) -> list[tuple[int, str]]:
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


def _buffer_value_fingerprint(value: torch.Tensor) -> tuple[Any, ...]:
    """Return a cheap equality-compatible fingerprint for a buffer value.

    ``torch.equal`` tensors always share a fingerprint (shape, dtype, device,
    and the first/last elements), so bucketing dedup candidates by this key
    never separates a pair the pairwise sweep would have merged. Collisions
    are fine -- the caller still confirms with ``torch.equal``. NaN sample
    elements compare unequal to themselves, which matches ``torch.equal``
    refusing to equate NaN-bearing tensors.

    Parameters
    ----------
    value:
        Captured buffer tensor.

    Returns
    -------
    tuple[Any, ...]
        Hashable bucket key.
    """

    numel = value.numel()
    head: Any
    tail: Any
    if numel == 0:
        head = tail = None
    else:
        try:
            if value.dim() == 0:
                head = tail = value.item()
            else:
                head = value[(0,) * value.dim()].item()
                tail = value[tuple(size - 1 for size in value.shape)].item()
        except (RuntimeError, ValueError):
            # Exotic dtypes without .item() support degrade to a
            # metadata-only bucket -- correct, just coarser.
            head = tail = "unsampled"
    return (str(value.dtype), str(value.device), tuple(value.shape), numel, head, tail)


def _fix_buffer_layers(self: Trace) -> None:
    """Step 6: Connect buffer sources, merge duplicates, and assign pass numbers.

    Buffer tensors (nn.Module registered buffers) are logged as source tensors
    during the forward pass but may lack proper parent connections. This function:

    1. Connects each buffer to its buffer_source (the tensor that produced the
       buffer's value), updating parent/child links and ancestry.
    2. Deduplicates buffers: buffers with the same containing module, same parent,
       same address, AND same tensor value are merged into a single node.
       The dedup hash is (modules + buffer_source + address).
    3. Assigns sequential buffer_pass numbers per address.

    Note: Buffer deduplication is scoped by containing module, source, address, and value.
    """
    buffer_counter: dict[str, int] = defaultdict(lambda: 1)
    buffer_hash_groups: dict[str, list[str]] = defaultdict(list)
    # Buffer rows whose edges this step changes; their descendant cones need ancestry
    # re-derived (see _repropagate_ancestry_after_buffer_wiring).
    rewired_buffers: list[str] = []

    for layer_label in self.buffer_layers:
        layer = self[layer_label]
        if layer.buffer_source is not None:
            rewired_buffers.append(layer._label_raw)
            if layer.buffer_source not in layer.parents:
                layer.parents.append(layer.buffer_source)
            if layer_label not in self[layer.buffer_source].children:
                self[layer.buffer_source].children.append(layer_label)
            self[layer.buffer_source].has_children = True
            source_matches_buffer = _buffer_source_value_matches(
                self[layer.buffer_source],
                layer,
                structure_only=bool(getattr(self, "structure_only", False)),
            )
            if source_matches_buffer:
                layer.func = identity
                layer.func_name = "identity"
            else:
                layer.buffer_replay_validated = False
            layer.has_input_ancestor = bool(self[layer.buffer_source].has_input_ancestor)
            layer.input_ancestors.update(self[layer.buffer_source].input_ancestors)
            layer.root_ancestors.discard(layer._label_raw)
            layer.root_ancestors.update(self[layer.buffer_source].root_ancestors)
            layer.parent_arg_positions["args"][0] = layer.buffer_source
            if (self[layer.buffer_source].out is not None) and (layer.saved_args is not None):
                layer.saved_args.append(
                    safe_copy(self[layer.buffer_source].out, detach_tensor=True)
                )

        if layer.address is None:
            equivalence_class = str(getattr(layer, "equivalence_class", ""))
            if equivalence_class.startswith("buffer_"):
                recovered_address = equivalence_class.removeprefix("buffer_")
                if recovered_address and recovered_address != "None":
                    layer.address = recovered_address
        if layer.address is None:
            layer.address = f"anonymous_buffer_{layer._label_raw}"
        buffer_hash = str(layer.modules) + str(layer.buffer_source) + layer.address
        buffer_hash_groups[buffer_hash].append(layer_label)

    # Merge buffers with the same hash AND the same tensor value.
    # Buffers sharing the same hash but different values are kept as separate
    # unique buffers (the for/else clause registers unmatched buffers as new uniques).
    # torch.equal candidates are narrowed by a cheap value fingerprint first:
    # the former sweep compared every new buffer against EVERY prior unique in
    # its hash group, Theta(G^2) whole-tensor compares when the values all
    # differ (a training-mode recurrent BatchNorm's running stats, hunt-6
    # R52-1). Equal tensors always share a fingerprint, so bucketing never
    # changes which unique a buffer merges into.
    deferred_buffer_removals: dict[str, tuple[Op, Op]] = {}
    # Per-survivor membership sets shared across the whole sweep: the
    # membership guards in _merge_buffer_entries otherwise rescan the
    # survivor's growing edge lists once per merged duplicate (the second
    # half of the R52-1 quadratic).
    survivor_edge_shadows: dict[str, dict[str, set[str]]] = {}
    for _, buffers_orig in buffer_hash_groups.items():
        unique_labels_by_fingerprint: dict[tuple[Any, ...], list[str]] = defaultdict(list)
        first_out = self[buffers_orig[0]].out if buffers_orig else None
        if first_out is not None:
            unique_labels_by_fingerprint[_buffer_value_fingerprint(first_out)].append(
                buffers_orig[0]
            )
        for buffer_label in buffers_orig[1:]:
            buffer = self[buffer_label]
            candidate_labels = (
                ()
                if buffer.out is None
                else unique_labels_by_fingerprint.get(_buffer_value_fingerprint(buffer.out), ())
            )
            for unique_buffer_label in candidate_labels:
                unique_buffer = self[unique_buffer_label]
                # W1-FAB guard: value-equality dedup is unobservable on meta
                # payloads (no aten::equal meta kernel); unknown never merges.
                if (
                    buffer.out.is_meta
                    or unique_buffer.out is not None
                    and unique_buffer.out.is_meta
                ):
                    continue
                if (unique_buffer.out is not None) and torch.equal(buffer.out, unique_buffer.out):
                    _merge_buffer_entries(
                        self,
                        unique_buffer,
                        buffer,
                        deferred_removals=deferred_buffer_removals,
                        survivor_edge_shadows=survivor_edge_shadows,
                    )
                    rewired_buffers.append(unique_buffer._label_raw)
                    break
            else:
                if buffer.out is not None:
                    unique_labels_by_fingerprint[_buffer_value_fingerprint(buffer.out)].append(
                        buffer_label
                    )

    _finish_deferred_buffer_removals(self, deferred_buffer_removals)

    _repropagate_ancestry_after_buffer_wiring(self, rewired_buffers)
    _repropagate_descendants_after_buffer_wiring(self, rewired_buffers)

    # And relabel the buffer ops.

    for layer_label in self.buffer_layers:
        layer = self[layer_label]
        address = layer.address
        layer.buffer_pass = buffer_counter[address]
        self.buffer_num_calls[address] = buffer_counter[address]
        buffer_counter[address] += 1


def _repropagate_ancestry_after_buffer_wiring(self: Trace, rewired: list[str]) -> None:
    """Re-derive ancestry over the DESCENDANT CONE of every buffer rewired at step 6.

    Step 6 inserts ``buffer -> buffer_source`` edges AFTER capture-time ancestry
    propagation and after steps 2/4, and it only ever updated the buffer row's OWN
    ``input_ancestors``/``root_ancestors``. Nothing revisited the descendants, so on a
    write-then-reread buffer every op downstream of the buffer kept its pre-edge sets:
    ``self.b[:2].copy_(x); return self.b.sum()`` produced a ``sum`` op whose parent
    carries ``input_ancestors={'input_1'}`` while the op itself carried ``set()`` -- an
    op that demonstrably depends on the model input reporting no input ancestry at all,
    on the default (depths-off) path where step 4's flood does not run to paper over it.
    Public ``op.input_ancestors`` / ``root_ancestors`` reads, reachability queries, and
    ``receptive_field`` all consume these sets, and they are portable state that survives
    save/load.

    The re-derivation is exactly the closure the ``ancestry_closure`` invariant checks,
    applied to the affected cone only (raw-label space, topological order):

    * ``input_ancestors``            = own-if-input, else the union over parents;
    * ``internal_source_ancestors``  = ``{self}`` if an internal source, else the union;
    * ``internal_source_parents``    = the parents carrying internal-source ancestry;
    * ``root_ancestors``             = ``input_ancestors | internal_source_ancestors``.

    Internal-source rows keep the ``root_ancestors`` value step 6 assigned them: the
    source-minting producers disagree about self-inclusion there (a parentless factory
    records the empty set, a buffer source records ``{self}``), which is a field-naming
    question, not something to silently change here.
    """

    if not rewired:
        return
    raw_dict = self._raw_graph_ws.raw_layer_dict
    cone: set[str] = set()
    frontier = [label for label in rewired if label in raw_dict]
    while frontier:
        current = frontier.pop()
        if current in cone:
            continue
        cone.add(current)
        frontier.extend(child for child in raw_dict[current].children if child in raw_dict)

    for raw_label in self._raw_graph_ws.raw_layer_labels_list:
        if raw_label not in cone:
            continue
        layer = raw_dict[raw_label]
        parents = [raw_dict[parent] for parent in layer.parents if parent in raw_dict]
        input_ancestors: set[str] = {raw_label} if layer.is_input else set()
        for parent in parents:
            input_ancestors.update(parent.input_ancestors)
        if layer.is_internal_source:
            internal_source_ancestors = {raw_label}
        else:
            internal_source_ancestors = set()
            for parent in parents:
                internal_source_ancestors.update(parent.internal_source_ancestors)
        layer.input_ancestors = input_ancestors
        layer.has_input_ancestor = bool(input_ancestors)
        layer.internal_source_ancestors = internal_source_ancestors
        layer.has_internal_source_ancestor = bool(internal_source_ancestors)
        if not layer.is_internal_source:
            layer.internal_source_parents = [
                parent._label_raw for parent in parents if parent.has_internal_source_ancestor
            ]
            layer.root_ancestors = input_ancestors | internal_source_ancestors


def _repropagate_descendants_after_buffer_wiring(self: Trace, rewired: list[str]) -> None:
    """Re-derive output reach over the ANCESTOR CONE of every buffer rewired at step 6.

    The child-direction mirror of :func:`_repropagate_ancestry_after_buffer_wiring`
    (which repairs only the four parent-direction sets). ``output_descendants`` /
    ``has_output_descendant`` are computed once at step 2 from the PRE-MERGE edges,
    and the step-6 duplicate merge transfers the removed duplicate's children onto
    the survivor without ever reconciling the survivor's child-direction reach.
    Merged duplicates reaching DIFFERENT output sets (a multi-output model whose
    value-identical buffer reads feed different outputs, or a dead-ending survivor
    merged with an output-reaching duplicate) therefore shipped stale
    ``output_descendants`` on the survivor and on every ancestor of it -- which the
    ``ancestry_closure`` invariant (a genuine recompute from the final edges)
    correctly FAILS on an honest capture.

    The re-derivation is exactly the closure the invariant checks, applied to the
    affected cone only (raw-label space, reverse topological order):

    * ``output_descendants`` = ``{self}`` if an output, else the union over children;
    * ``has_output_descendant`` mirrors the set's emptiness.

    The cone walks PARENT edges from every rewired/survivor buffer: only ancestors
    of a node whose child edges changed can have gained (or lost) output reach.
    Children outside the cone kept their step-2 values, which are still the closure
    of their (unchanged) child edges, so reading them is sound.

    Distance fields are deliberately untouched, matching the parent-direction
    repair's scope: step 4 populates them only under the non-default
    ``mark_layer_depths`` and the distance closure check skips ``None`` values.
    """

    if not rewired:
        return
    raw_dict = self._raw_graph_ws.raw_layer_dict
    cone: set[str] = set()
    frontier = [label for label in rewired if label in raw_dict]
    while frontier:
        current = frontier.pop()
        if current in cone:
            continue
        cone.add(current)
        frontier.extend(parent for parent in raw_dict[current].parents if parent in raw_dict)

    for raw_label in reversed(self._raw_graph_ws.raw_layer_labels_list):
        if raw_label not in cone:
            continue
        layer = raw_dict[raw_label]
        children = [raw_dict[child] for child in layer.children if child in raw_dict]
        output_descendants: set[str] = {raw_label} if layer.is_output else set()
        for child in children:
            output_descendants.update(child.output_descendants)
        layer.output_descendants = output_descendants
        layer.has_output_descendant = bool(output_descendants)


def _buffer_source_value_matches(
    source: Op, buffer_layer: Op, *, structure_only: bool = False
) -> bool:
    """Return whether a buffer-version source op output equals the full buffer value.

    Structure-only captures retain no payloads (W3), so the value comparison
    cannot run; the verdict falls back to the DECLARED geometry (shape +
    dtype agreement between the producer op and the buffer's write record) —
    a hypothesis claim exactly like the shapes it rides on, corroborated or
    refuted by discharge against a real capture (weightsfree memo D7: the
    parity gate is the acceptance authority, the graph must not diverge from
    the real capture's purely because values were unavailable).
    """

    if structure_only and (source.out is None or buffer_layer.out is None):
        source_shape = getattr(source, "shape", None)
        buffer_shape = getattr(buffer_layer, "shape", None)
        return (
            source_shape is not None
            and buffer_shape is not None
            and tuple(source_shape) == tuple(buffer_shape)
            and str(getattr(source, "dtype", None)) == str(getattr(buffer_layer, "dtype", None))
        )
    if source.out is None or buffer_layer.out is None:
        return False
    if not isinstance(source.out, torch.Tensor) or not isinstance(buffer_layer.out, torch.Tensor):
        return False
    if tuple(source.out.shape) != tuple(buffer_layer.out.shape):
        return False
    with pause_logging():
        try:
            return bool(torch.equal(source.out, buffer_layer.out))
        except Exception:
            return False


def _merge_buffer_entries(
    self: Trace,
    source_buffer: Op,
    buffer_to_remove: Op,
    *,
    deferred_removals: dict[str, tuple[Op, Op]] | None = None,
    survivor_edge_shadows: dict[str, dict[str, set[str]]] | None = None,
) -> None:
    """Merge a duplicate buffer into a source buffer, rewiring all edges.

    Transfers all child and parent connections from ``buffer_to_remove`` to
    ``source_buffer``, updates parent_arg_positions in children to point to
    the source buffer, fixes internal_source_parents/ancestors references
    across the graph, and removes the duplicate from the layer dict.

    ``survivor_edge_shadows`` (keyed by survivor raw label) carries the
    survivor's edge-list membership sets across repeated merges into the same
    survivor, so the guards below stay O(1) instead of rescanning lists that
    grow with every merged duplicate (hunt-6 R52-1). Any entry for a node
    whose lists this call mutates as a NEIGHBOUR is invalidated, keeping the
    shadows exact.
    """
    if survivor_edge_shadows is None:
        survivor_edge_shadows = {}
    shadow = survivor_edge_shadows.get(source_buffer._label_raw)
    if shadow is None:
        shadow = {
            "children": set(source_buffer.children),
            "parents": set(source_buffer.parents),
            "internal_source_parents": set(source_buffer.internal_source_parents),
            "conditional_entry_children": set(source_buffer.conditional_entry_children),
        }
        survivor_edge_shadows[source_buffer._label_raw] = shadow
    # The removed duplicate can never be a survivor again.
    survivor_edge_shadows.pop(buffer_to_remove._label_raw, None)
    for child_layer in buffer_to_remove.children:
        if child_layer not in shadow["children"]:
            shadow["children"].add(child_layer)
            source_buffer.children.append(child_layer)
        # This call rewrites the child's own parent lists below; drop any
        # survivor shadow it may hold so a later merge rebuilds it fresh.
        if child_layer != source_buffer._label_raw:
            survivor_edge_shadows.pop(child_layer, None)
        # Preserve edge MULTIPLICITY: ``parents`` is an edge-OCCURRENCE list (one entry
        # per argument slot), so a child consuming the removed buffer at two slots must
        # end with two entries naming the survivor. ``list.remove`` strips only the FIRST
        # occurrence, so repointing one-for-one is the multiplicity-faithful move; the
        # closing ``_remove_log_entry(..., remove_references=True)`` scrub would otherwise
        # strip the leftovers and drop the count to 1 (DISPUTED D1 -- safe hardening,
        # not an adjudication of reachability).
        child_parents = self[child_layer].parents
        repointed = 0
        while buffer_to_remove._label_raw in child_parents:
            child_parents.remove(buffer_to_remove._label_raw)
            repointed += 1
        child_parents.extend([source_buffer._label_raw] * max(1, repointed))
        if buffer_to_remove._label_raw in self[child_layer].internal_source_parents:
            self[child_layer].internal_source_parents.remove(buffer_to_remove._label_raw)
            self[child_layer].internal_source_parents.append(source_buffer._label_raw)

        for arg_type in ["args", "kwargs"]:
            for arg_label, arg_val in self[child_layer].parent_arg_positions[arg_type].items():
                if arg_val == buffer_to_remove._label_raw:
                    self[child_layer].parent_arg_positions[arg_type][arg_label] = (
                        source_buffer._label_raw
                    )

    # The survivor now owns the removed duplicate's child edges, so it reaches
    # every output the duplicate reached: merge the child-direction ancestry
    # WITH the edges (symmetric with the parent-direction ancestry handling).
    # The survivor's own ANCESTORS are reconciled afterwards by
    # _repropagate_descendants_after_buffer_wiring's cone re-derivation.
    if buffer_to_remove.has_output_descendant:
        source_buffer.output_descendants.update(buffer_to_remove.output_descendants)
        source_buffer.has_output_descendant = True

    for parent_layer in buffer_to_remove.parents:
        if parent_layer not in shadow["parents"]:
            shadow["parents"].add(parent_layer)
            source_buffer.parents.append(parent_layer)
        if parent_layer != source_buffer._label_raw:
            survivor_edge_shadows.pop(parent_layer, None)
        parent_children = self[parent_layer].children
        if buffer_to_remove._label_raw in parent_children:
            parent_children.remove(buffer_to_remove._label_raw)
        # Membership-guard the NEIGHBOUR side too (DISPUTED D1 -- safe hardening either
        # way, NOT an adjudication of reachability). The survivor's own appends above are
        # guarded, but this one was unconditional: both merged duplicates share their
        # parent BY CONSTRUCTION (the dedup hash at the call site includes
        # ``buffer_source``), so on any non-None-source merge the shared parent's
        # ``children`` got the survivor appended a SECOND time -- a duplicated child edge,
        # a shape no honest capture produces (parents may legitimately duplicate for
        # multi-slot reuse; children never do).
        if source_buffer._label_raw not in parent_children:
            parent_children.append(source_buffer._label_raw)

    for parent_layer in buffer_to_remove.internal_source_parents:
        if parent_layer not in shadow["internal_source_parents"]:
            shadow["internal_source_parents"].add(parent_layer)
            source_buffer.internal_source_parents.append(parent_layer)

    # Step 5 ran BEFORE this merge: transfer the removed duplicate's
    # conditional-parent annotations so the survivor keeps parenting the
    # branch bools / arm-entry children the duplicate parented. The matching
    # trace-level edges (conditional_branch_edges, conditional_arm_entry_edges,
    # conditional_edge_call_indices) repoint via ``replacement_labels`` in the
    # closing reference scrub (deep-hunt C3).
    for entry_child in buffer_to_remove.conditional_entry_children:
        if entry_child not in shadow["conditional_entry_children"]:
            shadow["conditional_entry_children"].add(entry_child)
            source_buffer.conditional_entry_children.append(entry_child)
    for cond_id, branch_children in buffer_to_remove.conditional_arm_children.items():
        survivor_branches = source_buffer.conditional_arm_children.setdefault(cond_id, {})
        for branch_kind, child_labels in branch_children.items():
            survivor_children = survivor_branches.setdefault(branch_kind, [])
            for child_label in child_labels:
                if child_label not in survivor_children:
                    survivor_children.append(child_label)

    if deferred_removals is not None:
        deferred_removals[buffer_to_remove._label_raw] = (source_buffer, buffer_to_remove)
        return

    self._raw_graph_ws.raw_layer_labels_list.remove(buffer_to_remove._label_raw)
    self._raw_graph_ws.raw_layer_dict.pop(buffer_to_remove._label_raw)

    for layer in self:
        if buffer_to_remove._label_raw in layer.root_ancestors:
            layer.root_ancestors.remove(buffer_to_remove._label_raw)
            layer.root_ancestors.add(source_buffer._label_raw)
        if buffer_to_remove._label_raw in layer.internal_source_ancestors:
            layer.internal_source_ancestors.remove(buffer_to_remove._label_raw)
            layer.internal_source_ancestors.add(source_buffer._label_raw)
        # Repoint any op whose scalar ``buffer_source`` still names the removed
        # buffer to the value-identical survivor. Unlike parents/children, the
        # ``buffer_source`` field (and its arg-0 mirror in ``parent_arg_positions``)
        # is NOT rewritten above and is absent from the raw->final rename + scrub
        # lists, so without this it dangles -- the buffer-merge analogue of the
        # campaign's "scrub removed buffer graph references" fix (e.g. speechbrain
        # CRDNN LiGRU per-forward ``drop_mask_te.to(device)`` reassign buffers).
        if layer.buffer_source == buffer_to_remove._label_raw:
            layer.buffer_source = source_buffer._label_raw
            arg_positions = layer.parent_arg_positions.get("args")
            if arg_positions is not None and arg_positions.get(0) == buffer_to_remove._label_raw:
                arg_positions[0] = source_buffer._label_raw

    self._remove_log_entry(
        buffer_to_remove,
        remove_references=True,
        replacement_labels={buffer_to_remove._label_raw: source_buffer._label_raw},
    )


def _finish_deferred_buffer_removals(
    self: Trace,
    removals: dict[str, tuple[Op, Op]],
) -> None:
    """Apply all trace-wide buffer substitutions in one graph scan.

    Parameters
    ----------
    self:
        Trace whose duplicate buffers were locally rewired.
    removals:
        Removed raw label to ``(survivor, removed op)`` mapping.
    """

    if not removals:
        return
    replacement_labels = {
        removed_label: source._label_raw for removed_label, (source, _removed) in removals.items()
    }
    removed_labels = set(removals)
    for layer in self:
        root_hits = layer.root_ancestors & removed_labels
        if root_hits:
            layer.root_ancestors.difference_update(root_hits)
            layer.root_ancestors.update(replacement_labels[label] for label in root_hits)
        source_hits = layer.internal_source_ancestors & removed_labels
        if source_hits:
            layer.internal_source_ancestors.difference_update(source_hits)
            layer.internal_source_ancestors.update(
                replacement_labels[label] for label in source_hits
            )
        replacement = replacement_labels.get(layer.buffer_source)
        if replacement is not None:
            old_source = layer.buffer_source
            layer.buffer_source = replacement
            arg_positions = layer.parent_arg_positions.get("args")
            if arg_positions is not None and arg_positions.get(0) == old_source:
                arg_positions[0] = replacement

    self._raw_graph_ws.raw_layer_labels_list[:] = [
        label for label in self._raw_graph_ws.raw_layer_labels_list if label not in removed_labels
    ]
    for removed_label in removed_labels:
        self._raw_graph_ws.raw_layer_dict.pop(removed_label, None)
    self._batch_remove_log_entries(
        (removed for _source, removed in removals.values()),
        remove_references=True,
        replacement_labels=replacement_labels,
    )
