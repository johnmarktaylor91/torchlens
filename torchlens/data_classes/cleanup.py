"""Trace cleanup helpers and post-session teardown.

This module provides the helper stack behind Trace cleanup operations:

1. **cleanup()** — full teardown: deletes all Op attributes, then
   deletes all Trace attributes (both FIELD_ORDER and internal containers).
   Breaks circular references (Trace <-> Op.source_trace,
   Module <-> _source_trace).
   Also frees GPU memory via ``torch.cuda.empty_cache()`` when CUDA is
   available (gated to avoid CUDA driver probe cost on CPU-only runs).

2. **_remove_log_entry_references()** — removes a single layer label from all
   Trace list/dict fields that hold graph references.

3. **_scrub_conditional_fields_after_removal()** — repairs conditional metadata
   after one or more labels are removed.

4. **_LIST_FIELDS_TO_CLEAN** — canonical Trace list fields filtered by both
   single-entry and batch removal helpers.
"""

from collections.abc import Iterable
from dataclasses import fields, is_dataclass, replace
from typing import TYPE_CHECKING, Any, cast

import torch

from .._trace_core.groups import GroupRef
from ..constants import MODEL_LOG_FIELD_ORDER
from ..intervention.types import ParentRef, Unsupported
from ..utils.collections import remove_entry_from_list
from ..utils.display import cleanup_trace_visualizer_dir
from ..utils.tensor_utils import _is_cuda_available, capture_touched_cuda
from ._state_adapter import state_items
from .op import Op

if TYPE_CHECKING:
    from .trace import Trace


def cleanup(self: "Trace") -> None:
    """Delete all log entries, break circular references, and free GPU memory.

    Called explicitly by the user or automatically at the end of a logging
    session. After cleanup, the Trace is effectively empty and should
    not be used further. No long-lived safetensors handles need to be
    closed here because lazy materialization opens and closes files per call.

    Raises
    ------
    CaptureContextError
        If ``self`` is the trace a live capture window is currently writing
        into (code ``cleanup_during_active_capture``). Husking the active
        trace mid-window (reachable single-threaded from a forward hook or
        ``activation_transform``) previously let the capture die on a raw
        ``AttributeError`` deep inside the commit path — sibling of the
        ``unwrap_torch`` / ``release_model`` mid-capture guards, refused
        typed here instead. Cleaning up a *different*, finished trace during
        a capture stays supported.
    """
    from .. import _state
    from .._errors import CaptureContextError
    from .._fast_run import close_fast_run_session
    from ..backends.torch.backward import _purge_trace_from_backward_registry
    from ..backends.torch.tensor_tracking import remove_owned_state_grad_hooks
    from ..captured_run import forget_event_stream

    if self.__dict__.get("_tl_cleaned_up", False):
        # Idempotent: a second cleanup() has nothing left to husk and must
        # not crash on its own output (b6-opus R25).
        return
    if _state._active_trace is self:
        raise CaptureContextError(
            "Trace.cleanup() was called on the trace a live TorchLens capture "
            "window is still writing into",
            code="cleanup_during_active_capture",
            remedy=(
                "let the capture (or backward projection) finish before "
                "cleaning up its trace — husking it mid-window kills the "
                "capture with an unrelated internal error"
            ),
        )

    close_fast_run_session(self)
    _purge_trace_from_backward_registry(self)
    # Gradient hooks on model state (a mutated Parameter's history) outlive the
    # pass; remove them so later backward passes on the model never reach them.
    remove_owned_state_grad_hooks(self)
    forget_event_stream(self)
    cleanup_trace_visualizer_dir(self)
    # Snapshot the CUDA gate BEFORE the attribute deletions below drop
    # ``forward_memory_backend`` (after which the fact reads as unknown and
    # fails toward the historical flush).
    touched_cuda = capture_touched_cuda(self)
    # GC-1: Release parameter references to allow model GC.
    if hasattr(self, "param_logs"):
        for pl in self.param_logs:
            pl.release_param_ref()
    # Materialize the M8 Layer mirror fields BEFORE husking the ops that back
    # them: a user-held Layer keeps exactly the readable metadata the dict-era
    # per-layer copies kept after cleanup (the copies existed at this point in
    # the dict era, so post-cleanup memory is unchanged).
    from .layer import materialize_layer_mirrors

    for layer_log in (self.__dict__.get("layer_logs") or {}).values():
        materialize_layer_mirrors(layer_log)
    # First, clear all attributes from each Op entry.
    # This breaks the Op -> Trace circular reference
    # (via source_trace) without needing per-entry reference removal.
    for tensor_log_entry in self:
        _clear_entry_attributes(tensor_log_entry)
    # Then delete all Trace attributes listed in the canonical FIELD_ORDER.
    for attr in MODEL_LOG_FIELD_ORDER:
        if hasattr(self, attr):
            delattr(self, attr)
    # GC-5/GC-12: Also clear internal containers not in MODEL_LOG_FIELD_ORDER.
    # These hold back-references (e.g. _module_logs -> Module -> _source_trace)
    # and large data structures (layer_logs, layer_dict_all_keys).
    for attr in [
        "_capture_events",
        "_raw_graph_ws",
        "_module_capture_ws",
        "_wrapper_runtime_ws",
        "_trace_core",
        "_saved_grad_labels",
        "_module_logs",
        "_buffer_accessor",
        "_param_logs_by_module",
        "layer_logs",
        "layer_dict_all_keys",
        "layer_dict_main_keys",
        "_orphan_labels",
        "_loaded_from_bundle",
        "_source_bundle_manifest_sha256",
        "_source_bundle_path",
        "_source_bundle_created_at",
        "_fast_run_session",
        "_validation_replay_status",
        # R33: the receptive-field solution cache (~54 MB on a resnet18
        # trace) lives outside MODEL_LOG_FIELD_ORDER, so it survived the
        # husking above with no eviction path at all.
        "_receptive_field_solution",
        # r8 R50 (sol): the save-budget accountant lives outside
        # MODEL_LOG_FIELD_ORDER too (session-time FieldPolicy.DROP), so it
        # survived cleanup with its per-device ledgers and weak payload
        # watches -- dead weight on a husked trace the :42 contract says is
        # effectively empty. (The sibling claim on
        # ``_predicate_lookback_candidates`` was verified already-clean.)
        "_save_budget_accountant",
    ]:
        if hasattr(self, attr):
            delattr(self, attr)
    # A cleaned trace no longer owns an event stream, but tensor/grad-fn hooks
    # already registered on the user's still-live graph cannot be removed and
    # will keep firing inside the user's own later backward(). Disarm them so
    # they no-op instead of raising the typed stream-refusal error from within
    # the autograd engine (which would kill an unrelated user backward).
    self._tl_backward_triggers_disarmed = True
    # Gated behind cached cuda.is_available() so CPU-only runs don't pay the
    # CUDA driver / NVML probe cost (per profiling audit 2026-04-27 finding #4),
    # AND on the capture having touched CUDA (R36-3): a CPU-only trace cleaned
    # up inside a GPU training loop must not flush the caller's allocator.
    # This was the third empty_cache site; the backend teardown and
    # postprocess step-13 sites were already gated.
    if _is_cuda_available() and touched_cuda:
        torch.cuda.empty_cache()
    # Husked-trace sentinel: every later public read funnels through
    # ``_raise_missing_trace_attribute`` and refuses with ONE typed code
    # (``trace_cleaned_up``) instead of leaking whichever private field the
    # reader touches first; ``Trace.outcome`` settles to UNKNOWN.
    self._tl_cleaned_up = True
    # R33: freed capture-sized allocations land in the glibc arena, not the
    # OS -- gc.collect() returned ~2.7% of a discarded capture's footprint
    # and RSS never receded in a long-lived analysis process. cleanup() is
    # the explicit give-the-memory-back API, so best-effort trim here
    # (glibc-only; a no-op elsewhere).
    _trim_host_allocator()


_MALLOC_TRIM: Any = False  # False = not probed yet; None = unavailable


def _trim_host_allocator() -> None:
    """Return freed glibc arena memory to the OS, best-effort.

    ``malloc_trim(0)`` only releases memory the allocator already considers
    free, so it is correctness-neutral; non-glibc hosts probe once to None
    and no-op forever after.
    """

    global _MALLOC_TRIM
    if _MALLOC_TRIM is False:
        try:
            import ctypes

            _MALLOC_TRIM = ctypes.CDLL("libc.so.6").malloc_trim
        except Exception:
            _MALLOC_TRIM = None
    if _MALLOC_TRIM is not None:
        try:
            _MALLOC_TRIM(0)
        except Exception:
            _MALLOC_TRIM = None


def _clear_entry_attributes(log_entry: Op) -> None:
    """Clear all instance attributes from a Op entry."""
    from .._trace_core.op_store import mark_op_row_released
    from .op import _detach_op_husk

    # Whole-row release: tell any active step audit the per-cell reads and
    # deletes below are the op's removal husking (a row-lifecycle event
    # checked against the step's 'deletes' row_effects sanction), not
    # column accesses.
    try:
        row_store = object.__getattribute__(log_entry, "_core")
        row = object.__getattribute__(log_entry, "_row")
    except AttributeError:
        row_store = None
    if row_store is not None:
        mark_op_row_released(row_store, row)
    for attr, _ in list(state_items(log_entry)):
        delattr(log_entry, attr)
    # Rebind the emptied facade to a detached row so a user-held husk cannot
    # pin the trace's shared columnar store (slot-era husks pinned nothing).
    _detach_op_husk(log_entry)


def _strip_pass_suffix(layer_label: str) -> str:
    """Remove any ``:call_index`` suffix from a layer label.

    Args:
        layer_label: Layer label, optionally pass-qualified.

    Returns:
        The pass-stripped label.
    """
    return layer_label.split(":", 1)[0]


def _label_for_reference_removal(log_entry: Op, pass_finished: bool) -> str:
    """Return the label namespace currently used by graph-level references.

    Parameters
    ----------
    log_entry:
        Entry being removed.
    pass_finished:
        Whether postprocessing has fully completed.

    Returns
    -------
    str
        Final layer label when available, otherwise the raw tensor label.
    """
    if pass_finished:
        return cast(str, log_entry.layer_label)
    if getattr(log_entry, "layer_label", None):
        return cast(str, log_entry.layer_label)
    return cast(str, log_entry._label_raw)


def _map_removed_label(
    label: str,
    labels_to_remove: set[str],
    replacements: dict[str, str] | None,
) -> str | None:
    """Return the surviving label for one reference, or ``None`` to drop it.

    Args:
        label: Referenced label.
        labels_to_remove: Labels removed in this pass.
        replacements: Optional removed-label -> survivor substitutions (merge
            removals repoint references; plain removals drop them).

    Returns:
        ``label`` itself when it survives, the substituted survivor when a
        replacement is known, else ``None``.
    """
    if label not in labels_to_remove:
        return label
    if replacements is not None:
        return replacements.get(label)
    return None


def _filter_conditional_arm_children(
    conditional_arm_children: dict[int, dict[str, list[str]]],
    labels_to_remove: set[str],
    replacements: dict[str, str] | None = None,
) -> dict[int, dict[str, list[str]]]:
    """Drop or substitute removed labels in ``conditional_arm_children``.

    Args:
        conditional_arm_children: ``cond_id -> branch_kind -> child labels``.
        labels_to_remove: Labels that should be removed.
        replacements: Optional removed-label -> survivor substitutions.

    Returns:
        A new nested dict with removed labels substituted or pruned, empty
        containers pruned, and substituted duplicates deduplicated in order.
    """
    filtered_children_by_cond: dict[int, dict[str, list[str]]] = {}
    for cond_id, branch_children in conditional_arm_children.items():
        filtered_branch_children: dict[str, list[str]] = {}
        for branch_kind, child_labels in branch_children.items():
            kept_children: list[str] = []
            kept_seen: set[str] = set()
            for child_label in child_labels:
                mapped = _map_removed_label(child_label, labels_to_remove, replacements)
                if mapped is not None and mapped not in kept_seen:
                    kept_seen.add(mapped)
                    kept_children.append(mapped)
            if kept_children:
                filtered_branch_children[branch_kind] = kept_children
        if filtered_branch_children:
            filtered_children_by_cond[cond_id] = filtered_branch_children
    return filtered_children_by_cond


def _filter_conditional_arm_entry_edges(
    conditional_arm_entry_edges: dict[tuple[int, str], list[tuple[str, str]]],
    labels_to_remove: set[str],
    replacements: dict[str, str] | None = None,
) -> dict[tuple[int, str], list[tuple[str, str]]]:
    """Drop or substitute removed labels in ``conditional_arm_entry_edges``.

    Args:
        conditional_arm_entry_edges: ``(cond_id, branch_kind) -> [(parent, child)]``.
        labels_to_remove: Labels that should be removed.
        replacements: Optional removed-label -> survivor substitutions.

    Returns:
        A new dict with empty edge lists pruned and substituted duplicate
        edges deduplicated in order.
    """
    filtered_arm_edges: dict[tuple[int, str], list[tuple[str, str]]] = {}
    for key, edge_list in conditional_arm_entry_edges.items():
        filtered_edges: list[tuple[str, str]] = []
        filtered_seen: set[tuple[str, str]] = set()
        for parent, child in edge_list:
            mapped_parent = _map_removed_label(parent, labels_to_remove, replacements)
            mapped_child = _map_removed_label(child, labels_to_remove, replacements)
            if mapped_parent is None or mapped_child is None:
                continue
            mapped_edge = (mapped_parent, mapped_child)
            if mapped_edge not in filtered_seen:
                filtered_seen.add(mapped_edge)
                filtered_edges.append(mapped_edge)
        if filtered_edges:
            filtered_arm_edges[key] = filtered_edges
    return filtered_arm_edges


def _filter_conditional_edge_call_indices(
    conditional_edge_call_indices: dict[tuple[str, str, int, str], list[int]],
    labels_to_remove_no_pass: set[str],
    replacements_no_pass: dict[str, str] | None = None,
) -> dict[tuple[str, str, int, str], list[int]]:
    """Drop or substitute removed labels in ``conditional_edge_call_indices`` keys.

    Args:
        conditional_edge_call_indices: ``(parent_no_pass, child_no_pass, cond_id, branch_kind) -> pass list``.
        labels_to_remove_no_pass: Pass-stripped labels that should be removed.
        replacements_no_pass: Optional pass-stripped removed-label -> survivor
            substitutions. A substituted key colliding with an existing key
            merges the two pass lists (sorted union).

    Returns:
        A new dict with removed-key entries substituted or pruned.
    """
    filtered_indices: dict[tuple[str, str, int, str], list[int]] = {}
    for key, call_indexs in conditional_edge_call_indices.items():
        mapped_parent = _map_removed_label(key[0], labels_to_remove_no_pass, replacements_no_pass)
        mapped_child = _map_removed_label(key[1], labels_to_remove_no_pass, replacements_no_pass)
        if mapped_parent is None or mapped_child is None:
            continue
        mapped_key = (mapped_parent, mapped_child, key[2], key[3])
        existing = filtered_indices.get(mapped_key)
        if existing is None:
            filtered_indices[mapped_key] = list(call_indexs)
        else:
            filtered_indices[mapped_key] = sorted(set(existing) | set(call_indexs))
    return filtered_indices


def _project_conditional_child_views(
    conditional_arm_children: dict[int, dict[str, list[str]]],
) -> tuple[list[str], dict[int, list[str]], list[str]]:
    """Project pass-level child views from ``conditional_arm_children``.

    Parameters
    ----------
    conditional_arm_children:
        Primary ``cond_id -> branch_kind -> child labels`` mapping for a
        concrete ``Op`` record.

    Returns
    -------
    tuple[list[str], dict[int, list[str]], list[str]]
        THEN, ELIF, and ELSE child views normalized to the invariant
        contract: unique labels sorted lexicographically.
    """

    then_children = sorted(
        {
            child_label
            for branch_children in conditional_arm_children.values()
            for child_label in branch_children.get("then", [])
        }
    )
    elif_children: dict[int, list[str]] = {}
    for branch_children in conditional_arm_children.values():
        for branch_kind, child_labels in branch_children.items():
            if not branch_kind.startswith("elif_"):
                continue
            elif_index = int(branch_kind.split("_", 1)[1])
            existing_children = set(elif_children.get(elif_index, []))
            existing_children.update(child_labels)
            elif_children[elif_index] = sorted(existing_children)

    else_children = sorted(
        {
            child_label
            for branch_children in conditional_arm_children.values()
            for child_label in branch_children.get("else", [])
        }
    )
    return then_children, elif_children, else_children


def _append_unique_child_label(
    child_labels: list[str], seen_labels: set[str], child_label: str
) -> None:
    """Append ``child_label`` to ``child_labels`` if it is not already present.

    Parameters
    ----------
    child_labels:
        Ordered child-label list being built.
    seen_labels:
        Membership set mirroring ``child_labels`` (the bare list scan made
        each aggregate projection O(k^2) in its child count -- R52).
    child_label:
        Candidate label to append.
    """

    if child_label not in seen_labels:
        seen_labels.add(child_label)
        child_labels.append(child_label)


def _project_aggregate_conditional_child_views(
    conditional_arm_children: dict[int, dict[str, list[str]]],
) -> tuple[list[str], dict[int, list[str]], list[str]]:
    """Project aggregate child views from ``conditional_arm_children``.

    Parameters
    ----------
    conditional_arm_children:
        Primary ``cond_id -> branch_kind -> child labels`` mapping for an
        aggregate ``Layer`` record.

    Returns
    -------
    tuple[list[str], dict[int, list[str]], list[str]]
        THEN, ELIF, and ELSE child views preserving first-seen order.
    """

    then_children: list[str] = []
    then_seen: set[str] = set()
    elif_children: dict[int, list[str]] = {}
    elif_seen: dict[int, set[str]] = {}
    else_children: list[str] = []
    else_seen: set[str] = set()
    for branch_children in conditional_arm_children.values():
        for child_label in branch_children.get("then", []):
            _append_unique_child_label(then_children, then_seen, child_label)
        for branch_kind, child_labels in branch_children.items():
            if not branch_kind.startswith("elif_"):
                continue
            elif_index = int(branch_kind.split("_", 1)[1])
            aggregate_children = elif_children.setdefault(elif_index, [])
            aggregate_seen = elif_seen.setdefault(elif_index, set())
            for child_label in child_labels:
                _append_unique_child_label(aggregate_children, aggregate_seen, child_label)
        for child_label in branch_children.get("else", []):
            _append_unique_child_label(else_children, else_seen, child_label)
    return then_children, elif_children, else_children


def _scrub_layer_entry_conditional_fields(
    layer_entry: Op,
    labels_to_remove: set[str],
    replacements: dict[str, str] | None = None,
) -> None:
    """Remove or repoint deleted labels in conditional fields on a surviving Op.

    Args:
        layer_entry: Surviving layer entry to scrub.
        labels_to_remove: Labels that were removed elsewhere in the log.
        replacements: Optional removed-label -> survivor substitutions.
    """
    entry_children: list[str] = []
    entry_seen: set[str] = set()
    for child_label in layer_entry.conditional_entry_children:
        mapped = _map_removed_label(child_label, labels_to_remove, replacements)
        if mapped is not None and mapped not in entry_seen:
            entry_seen.add(mapped)
            entry_children.append(mapped)
    layer_entry.conditional_entry_children = entry_children
    layer_entry.conditional_arm_children = _filter_conditional_arm_children(
        layer_entry.conditional_arm_children,
        labels_to_remove,
        replacements,
    )
    (
        layer_entry.conditional_then_children,
        layer_entry.conditional_elif_children,
        layer_entry.conditional_else_children,
    ) = _project_conditional_child_views(layer_entry.conditional_arm_children)


def _scrub_layer_log_conditional_fields(
    self: "Trace",
    labels_to_remove_no_pass: set[str],
    replacements_no_pass: dict[str, str] | None = None,
) -> None:
    """Remove or repoint deleted labels in aggregate Layer conditional fields.

    Args:
        self: Trace owning the LayerLogs.
        labels_to_remove_no_pass: Pass-stripped labels that were removed.
        replacements_no_pass: Optional pass-stripped removed-label -> survivor
            substitutions.
    """
    for layer_log in getattr(self, "layer_logs", {}).values():
        if "conditional_entry_children" not in getattr(layer_log, "__dict__", {}):
            continue
        # Layer is dict-backed (no normalizing descriptors until M8), so the
        # scrub itself preserves the finished-trace immutable relation
        # surface: tuple views in, tuple views out.
        aggregate_entry_children: list[str] = []
        aggregate_seen: set[str] = set()
        for child_label in layer_log.conditional_entry_children:
            mapped = _map_removed_label(child_label, labels_to_remove_no_pass, replacements_no_pass)
            if mapped is not None and mapped not in aggregate_seen:
                aggregate_seen.add(mapped)
                aggregate_entry_children.append(mapped)
        layer_log.conditional_entry_children = tuple(aggregate_entry_children)
        layer_log.conditional_arm_children = _filter_conditional_arm_children(
            layer_log.conditional_arm_children,
            labels_to_remove_no_pass,
            replacements_no_pass,
        )
        (
            then_children,
            elif_children,
            else_children,
        ) = _project_aggregate_conditional_child_views(layer_log.conditional_arm_children)
        layer_log.conditional_then_children = tuple(then_children)
        layer_log.conditional_elif_children = elif_children
        layer_log.conditional_else_children = tuple(else_children)


def _scrub_conditional_fields_after_removal(
    self: "Trace",
    labels_to_remove: set[str],
    surviving_entries: Iterable[Op],
    replacement_labels: dict[str, str] | None = None,
) -> None:
    """Scrub conditional references after one or more layer labels are removed.

    Args:
        self: Trace being updated.
        labels_to_remove: Removed layer labels using the same qualification as the
            current removal pass.
        surviving_entries: Surviving Op entries to scrub in-place.
        replacement_labels: Optional removed-label -> survivor substitutions.
            Merge-style removals (step-6 buffer dedup) REPOINT conditional
            references to the value-identical survivor instead of dropping
            them; plain removals keep the historical drop behavior.
    """
    labels_to_remove_no_pass = {_strip_pass_suffix(layer_label) for layer_label in labels_to_remove}
    replacements_no_pass = _strip_pass_suffix_replacements(replacement_labels)

    for layer_entry in surviving_entries:
        _scrub_layer_entry_conditional_fields(layer_entry, labels_to_remove, replacement_labels)

    _scrub_layer_log_conditional_fields(self, labels_to_remove_no_pass, replacements_no_pass)

    self.conditional_arm_entry_edges = _filter_conditional_arm_entry_edges(
        self.conditional_arm_entry_edges,
        labels_to_remove,
        replacement_labels,
    )
    self.conditional_edge_call_indices = _filter_conditional_edge_call_indices(
        self.conditional_edge_call_indices,
        labels_to_remove_no_pass,
        replacements_no_pass,
    )
    for conditional_event in self.conditional_records:
        if replacement_labels is not None:
            # Substitute IN PLACE (never drop): ``_arm_bool_indices`` and
            # ``_bool_layers_raw`` are index-aligned with ``bool_layers``, so
            # positional substitution keeps them coherent where a drop-filter
            # would silently shift every later index.
            conditional_event.bool_layers = [
                replacement_labels.get(layer_label, layer_label)
                if layer_label in labels_to_remove
                else layer_label
                for layer_label in conditional_event.bool_layers
                if layer_label not in labels_to_remove
                or replacement_labels.get(layer_label) is not None
            ]
            bool_layers_raw = getattr(conditional_event, "_bool_layers_raw", None)
            if bool_layers_raw is not None:
                setattr(
                    conditional_event,
                    "_bool_layers_raw",
                    [
                        replacement_labels.get(layer_label, layer_label)
                        if layer_label in labels_to_remove
                        else layer_label
                        for layer_label in bool_layers_raw
                        if layer_label not in labels_to_remove
                        or replacement_labels.get(layer_label) is not None
                    ],
                )
        else:
            conditional_event.bool_layers = [
                layer_label
                for layer_label in conditional_event.bool_layers
                if layer_label not in labels_to_remove
            ]

    _scrub_intervention_fields_after_removal(self, labels_to_remove, surviving_entries)


def _strip_pass_suffix_replacements(
    replacement_labels: dict[str, str] | None,
) -> dict[str, str] | None:
    """Project a replacement map into the pass-stripped label namespace.

    Args:
        replacement_labels: Removed-label -> survivor substitutions, or ``None``.

    Returns:
        The pass-stripped projection, dropping keys whose stripped forms
        collide with conflicting survivors (those fall back to the historical
        drop behavior), or ``None`` when no map was given.
    """
    if replacement_labels is None:
        return None
    stripped: dict[str, str] = {}
    conflicting: set[str] = set()
    for removed_label, survivor_label in replacement_labels.items():
        removed_no_pass = _strip_pass_suffix(removed_label)
        survivor_no_pass = _strip_pass_suffix(survivor_label)
        # Identity mappings (survivor shares the base label) are recorded so
        # the no-pass key is KEPT rather than dropped as removed.
        existing = stripped.get(removed_no_pass)
        if existing is not None and existing != survivor_no_pass:
            conflicting.add(removed_no_pass)
            continue
        stripped[removed_no_pass] = survivor_no_pass
    for removed_no_pass in conflicting:
        stripped.pop(removed_no_pass, None)
    return stripped


def _scrub_intervention_fields_after_removal(
    self: Any,
    labels_to_remove: set[str],
    surviving_entries: Iterable[Op],
) -> None:
    """Scrub replay/intervention metadata that carries layer labels.

    Args:
        self: Trace being updated.
        labels_to_remove: Removed labels in the active label namespace.
        surviving_entries: Surviving entries to scrub in-place.
    """

    for layer_entry in surviving_entries:
        layer_entry._edge_uses = [
            edge
            for edge in getattr(layer_entry, "_edge_uses", [])
            if edge.parent_label not in labels_to_remove
            and edge.child_label not in labels_to_remove
        ]
        layer_entry.args_template = _replace_removed_parent_refs(
            getattr(layer_entry, "args_template", None), labels_to_remove
        )
        layer_entry.kwargs_template = _replace_removed_parent_refs(
            getattr(layer_entry, "kwargs_template", None), labels_to_remove
        )
        interventions = [
            record
            for record in getattr(layer_entry, "interventions", [])
            if not _record_mentions_removed_label(record, labels_to_remove)
        ]
        if hasattr(layer_entry, "_internal_set"):
            layer_entry._internal_set("interventions", interventions)
        else:
            layer_entry.interventions = interventions

    self.state_history = [
        record
        for record in getattr(self, "state_history", [])
        if not _record_mentions_removed_label(record, labels_to_remove)
    ]
    _scrub_intervention_spec_after_removal(self, labels_to_remove)


def _scrub_intervention_spec_after_removal(self: Any, labels_to_remove: set[str]) -> None:
    """Remove deleted-label entries from mutable intervention-spec collections.

    Parameters
    ----------
    self:
        Trace-like owner of the mutable intervention spec.
    labels_to_remove:
        Removed labels in the active label namespace.
    """

    intervention_spec = getattr(self, "_intervention_spec", None)
    if intervention_spec is None:
        return

    spec_mutated = False
    for spec_field in fields(intervention_spec):
        field_value = getattr(intervention_spec, spec_field.name)
        if not isinstance(field_value, list):
            continue
        filtered_value = [
            record
            for record in field_value
            if not _record_mentions_removed_label(record, labels_to_remove)
        ]
        if len(filtered_value) == len(field_value):
            continue
        setattr(intervention_spec, spec_field.name, filtered_value)
        spec_mutated = True

    if spec_mutated and hasattr(self, "_mark_intervention_spec_mutated"):
        self._mark_intervention_spec_mutated()


def _replace_removed_parent_refs(value: Any, labels_to_remove: set[str]) -> Any:
    """Replace template parent refs to removed labels with unsupported leaves.

    Args:
        value: Template or nested component.
        labels_to_remove: Removed layer labels.

    Returns:
        Template value with stale parent refs replaced.
    """

    if isinstance(value, ParentRef) and value.parent_label in labels_to_remove:
        return Unsupported(reason="removed_parent_ref", value_type="ParentRef")
    if isinstance(value, tuple):
        return tuple(_replace_removed_parent_refs(item, labels_to_remove) for item in value)
    if isinstance(value, list):
        return [_replace_removed_parent_refs(item, labels_to_remove) for item in value]
    if isinstance(value, dict):
        return {
            key: _replace_removed_parent_refs(item, labels_to_remove) for key, item in value.items()
        }
    if is_dataclass(value) and not isinstance(value, type):
        updates = {
            field.name: _replace_removed_parent_refs(getattr(value, field.name), labels_to_remove)
            for field in fields(value)
            if hasattr(value, field.name)
        }
        return replace(value, **updates)
    return value


def _record_mentions_removed_label(record: Any, labels_to_remove: set[str]) -> bool:
    """Return whether a record contains a removed label-bearing field.

    Args:
        record: Dataclass record or nested object.
        labels_to_remove: Removed labels.

    Returns:
        Whether the record references a removed label.
    """

    label_fields = {"parent_label", "child_label", "target_label", "call_label", "site_label"}
    if isinstance(record, str):
        return record in labels_to_remove
    selector_kind = getattr(record, "selector_kind", None)
    selector_value = getattr(record, "selector_value", None)
    if selector_kind == "label" and isinstance(selector_value, str):
        return selector_value in labels_to_remove
    if isinstance(record, (list, tuple)):
        return any(_record_mentions_removed_label(item, labels_to_remove) for item in record)
    if isinstance(record, dict):
        return any(
            _record_mentions_removed_label(key, labels_to_remove)
            or _record_mentions_removed_label(value, labels_to_remove)
            for key, value in record.items()
        )
    if is_dataclass(record) and not isinstance(record, type):
        for field in fields(record):
            if not hasattr(record, field.name):
                continue
            field_value = getattr(record, field.name)
            if field.name in label_fields and field_value in labels_to_remove:
                return True
            if field.name not in label_fields and _record_mentions_removed_label(
                field_value, labels_to_remove
            ):
                return True
    return False


# List fields on Trace that hold tensor labels and need filtering during entry
# removal. Both single-entry and batch removal iterate this list.
_LIST_FIELDS_TO_CLEAN = [
    "input_layers",
    "output_layers",
    "buffer_layers",
    "internal_source_ops",
    "internal_sink_ops",
    "internally_terminated_bool_ops",
]

_OP_LABEL_FIELDS_TO_CLEAN = (
    "parents",
    "root_ancestors",
    "children",
    "input_ancestors",
    "output_descendants",
    "internal_source_parents",
    "internal_source_ancestors",
    "conditional_entry_children",
    "conditional_then_children",
    "conditional_else_children",
    "equivalent_ops",
    "recurrent_ops",
)

#: The M7 group-membership fields: scrubbed via their shared group row.
_OP_GROUP_FIELDS = frozenset({"equivalent_ops", "recurrent_ops"})


def _remove_log_entry_references(
    self: "Trace",
    layer_to_remove: str,
    replacement_labels: dict[str, str] | None = None,
) -> None:
    """Removes all references to a single Op from the Trace's list/dict fields.

    This is the single-entry counterpart to the reference-cleaning logic in
    ``_batch_remove_log_entries``. Both iterate ``_LIST_FIELDS_TO_CLEAN`` for
    Trace list fields.

    Args:
        layer_to_remove: The label of the log entry to remove.
        replacement_labels: Optional removed-label -> survivor substitutions
            for merge-style removals (conditional references repoint instead
            of dropping).
    """
    # Clear any fields in Trace referring to the entry.

    for field_name in _LIST_FIELDS_TO_CLEAN:
        remove_entry_from_list(getattr(self, field_name), layer_to_remove)

    _scrub_conditional_fields_after_removal(self, {layer_to_remove}, self, replacement_labels)

    self.conditional_branch_edges = _substitute_conditional_branch_edges(
        self.conditional_branch_edges, {layer_to_remove}, replacement_labels
    )
    # Now any nested fields.

    for _param_group, tensor_labels in self.layers_with_params.items():
        if layer_to_remove in tensor_labels:
            tensor_labels.remove(layer_to_remove)
    self.layers_with_params = {
        param_group: tensor_labels
        for param_group, tensor_labels in self.layers_with_params.items()
        if len(tensor_labels) > 0
    }

    for _equiv_group, equiv_tensor_labels in self.op_equivalence_classes.items():
        if layer_to_remove in equiv_tensor_labels:
            equiv_tensor_labels.remove(layer_to_remove)
    self.op_equivalence_classes = {
        equiv_group: tensor_labels
        for equiv_group, tensor_labels in self.op_equivalence_classes.items()
        if len(tensor_labels) > 0
    }

    _scrub_per_op_equivalence_lists(self, {layer_to_remove})


def _substitute_conditional_branch_edges(
    conditional_branch_edges: list[tuple[str, str]],
    labels_to_remove: set[str],
    replacement_labels: dict[str, str] | None = None,
) -> list[tuple[str, str]]:
    """Drop or repoint removed labels in ``conditional_branch_edges``.

    Args:
        conditional_branch_edges: Trace-level ``(parent, child)`` IF edges.
        labels_to_remove: Removed labels in the active label namespace.
        replacement_labels: Optional removed-label -> survivor substitutions.

    Returns:
        A new edge list with removed labels substituted (merge removals) or
        dropped, substituted duplicates deduplicated in order.
    """
    filtered_edges: list[tuple[str, str]] = []
    # Set-backed dedup (round-7 R52, wave-introduced 1e211f81): the list
    # membership test made this O(E^2) PER REMOVED LAYER (probe-measured
    # exponent ~2.3 on trace-level IF edges); the arm-edge filter above
    # already uses the seen-set idiom.
    filtered_seen: set[tuple[str, str]] = set()
    for parent, child in conditional_branch_edges:
        mapped_parent = _map_removed_label(parent, labels_to_remove, replacement_labels)
        mapped_child = _map_removed_label(child, labels_to_remove, replacement_labels)
        if mapped_parent is None or mapped_child is None:
            continue
        mapped_edge = (mapped_parent, mapped_child)
        if mapped_edge not in filtered_seen:
            filtered_seen.add(mapped_edge)
            filtered_edges.append(mapped_edge)
    return filtered_edges


def _scrub_per_op_equivalence_lists(ops: Iterable["Op"], labels_to_remove: set[str]) -> None:
    """Remove dead labels from per-op graph-reference fields.

    ``equivalent_ops`` and other raw-label fields are stored per op
    (``FieldPolicy.KEEP``), so removing an Op from the Trace does not by itself
    clear references to it held by other ops. Later postprocess phases rename
    these fields by raw-label lookup; stale labels therefore mean the graph is
    internally inconsistent and can raise ``KeyError`` before validation gets a
    chance to replay the model. Keep every raw-label-bearing per-op field
    consistent with the surviving graph.
    """

    for op in ops:
        _scrub_op_label_collections(op, labels_to_remove)
        _scrub_parent_arg_positions(op, labels_to_remove)
        _scrub_out_versions_by_child(op, labels_to_remove)
        _scrub_conditional_child_maps(op, labels_to_remove)


def _scrub_op_label_collections(op: "Op", labels_to_remove: set[str]) -> None:
    """Remove dead labels from direct list/set fields on one op.

    Parameters
    ----------
    op:
        Operation record to repair.
    labels_to_remove:
        Raw labels that no longer have a materialized operation record.
    """

    core = getattr(op, "_core", None)
    for field_name in _OP_LABEL_FIELDS_TO_CLEAN:
        # M7 group fields: the raw cell holds ONE GroupRef shared by every
        # member. Scrub the GROUP ROW once — every member's next read
        # reflects the filtered membership (live views), sharing intact.
        # Idempotent: the second member sees a disjoint view and skips.
        if field_name in _OP_GROUP_FIELDS and core is not None:
            fid = core.layout.fid_by_name.get(field_name)
            raw = core.cell_get(op._row, fid) if fid is not None else None
            if raw.__class__ is GroupRef:
                view = raw.view()
                if not labels_to_remove.isdisjoint(view):
                    raw.groups.replace(
                        raw.group_id,
                        [label for label in view if label not in labels_to_remove],
                    )
                continue
        value = getattr(op, field_name, None)
        if not value:
            continue
        if isinstance(value, (list, tuple)):
            # Only rebind when a dead label is actually present, for the same
            # reason as the set branch below: ``recurrent_ops`` shares ONE
            # canonical list across every Op of a recurrence group, and an
            # unconditional rebind would hand every Op its own equal-but-
            # distinct copy for nothing. Finished traces store immutable
            # tuple views here; reading one materializes any CSR-backed cell
            # into an explicit view, and the rebind below writes the filtered
            # view back, so removal can never resurrect through the edge
            # table.
            if not labels_to_remove.isdisjoint(value):
                setattr(op, field_name, [label for label in value if label not in labels_to_remove])
        elif isinstance(value, (set, frozenset)):
            # Only rebind when a dead label is actually present. ``equivalent_ops``
            # shares ONE set object across every Op of an equivalence class, and
            # the Trace-level group behind it is scrubbed in place by the caller,
            # so an unconditional ``value - labels_to_remove`` would hand every Op
            # its own equal-but-distinct copy and undo that sharing for nothing.
            if not value.isdisjoint(labels_to_remove):
                setattr(op, field_name, value - labels_to_remove)


def _scrub_parent_arg_positions(op: "Op", labels_to_remove: set[str]) -> None:
    """Remove parent-argument references to deleted raw labels.

    Parameters
    ----------
    op:
        Operation record to repair.
    labels_to_remove:
        Raw labels that no longer have a materialized operation record.
    """

    parent_arg_positions = getattr(op, "parent_arg_positions", None)
    if not parent_arg_positions:
        return
    for arg_type in ("args", "kwargs"):
        positions = parent_arg_positions.get(arg_type, {})
        for key, value in list(positions.items()):
            if value in labels_to_remove:
                del positions[key]


def _scrub_out_versions_by_child(op: "Op", labels_to_remove: set[str]) -> None:
    """Remove output-version records keyed by deleted child raw labels.

    Parameters
    ----------
    op:
        Operation record to repair.
    labels_to_remove:
        Raw labels that no longer have a materialized operation record.
    """

    out_versions_by_child = getattr(op, "out_versions_by_child", None)
    if not out_versions_by_child:
        return
    op.out_versions_by_child = {
        child_label: tensor_version
        for child_label, tensor_version in out_versions_by_child.items()
        if child_label not in labels_to_remove
    }


def _scrub_conditional_child_maps(op: "Op", labels_to_remove: set[str]) -> None:
    """Remove deleted raw labels from nested conditional child maps.

    Parameters
    ----------
    op:
        Operation record to repair.
    labels_to_remove:
        Raw labels that no longer have a materialized operation record.
    """

    conditional_arm_children = getattr(op, "conditional_arm_children", None)
    if conditional_arm_children:
        op.conditional_arm_children = {
            cond_id: {
                branch_kind: [label for label in child_labels if label not in labels_to_remove]
                for branch_kind, child_labels in branch_children.items()
            }
            for cond_id, branch_children in conditional_arm_children.items()
        }
        (
            op.conditional_then_children,
            op.conditional_elif_children,
            op.conditional_else_children,
        ) = _project_conditional_child_views(op.conditional_arm_children)
        return

    conditional_elif_children = getattr(op, "conditional_elif_children", None)
    if conditional_elif_children:
        op.conditional_elif_children = {
            elif_ix: [label for label in child_labels if label not in labels_to_remove]
            for elif_ix, child_labels in conditional_elif_children.items()
        }
