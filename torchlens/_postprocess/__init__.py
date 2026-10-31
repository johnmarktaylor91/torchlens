"""Postprocessing pipeline for cleaning up the model log after the forward pass.

After the forward pass captures raw tensor metadata into a Trace, this pipeline
transforms the raw graph into its user-facing form. The full pipeline runs
stable contract steps 0-20, split into thematic submodules:

- graph_traversal (Steps 1-4): Add output nodes, trace ancestry, remove orphans,
  compute input/output distances.
- control_flow (Steps 5-6): Mark conditional branches and deduplicate/merge buffer layers.
- loop_detection (Step 7): Identify repeated operations (loops/recurrence), assign
  same-layer groupings via BFS isomorphic subgraph expansion.
- labeling (Steps 8-11): Generate final human-readable labels, rename all internal
  references, trim/reorder fields, build lookup keys, and finalize retained layer lists.
- finalization (Steps 12-20): Undecorate saved tensors, log timing, finalize
  ParamLogs, build Layer/Module aggregates, mark pass as finished, then
  finalize any streamed bundle, optionally evict in-memory outs, and release
  live parameter references.

Step order is load-bearing: ``_executor.STEP_REGISTRY`` lists the steps in the
order they run, and a reorder is verified by the byte-identity oracles. Warnings
and first-exception order are pinned by that order.

Out-of-model, disclosed: deferred gradient streaming re-runs step-18/19-equivalent
code after backward, outside the pipeline.
"""

import time
import warnings
from typing import TYPE_CHECKING, Any

import torch

from .._data_classes._compaction import compact_op_metadata as _compact_op_metadata
from .._ir.capture_events import _clone_op_event_for_replay
from .._trace_core.relation_views import freeze_trace_relation_views as _freeze_relation_views
from .._utils.hashing import (
    compute_graph_shape_hash,
    compute_raw_event_shape_hash,
    populate_normalized_layer_addresses,
)
from .._utils.tensor_utils import _is_cuda_available
from ..backends.torch.ops import _compact_ancestor_sets
from ..capture.session import capture_session_for_events
from . import ast_branches
from ._executor import StepContext, run_pipeline
from ._materialize import materialize_from_events
from ._primitive_profile import _finalize_forward_primitive_profile
from .ast_branches import resolve_var_names
from .control_flow import (
    _fix_buffer_layers,
    _mark_conditional_branches,
)
from .finalization import (
    _build_layer_logs,
    _build_module_logs,
    _evict_streamed_outs,
    _finalize_param_logs,
    _finalize_streamed_bundle,
    _log_time_elapsed,
    _set_tracing_finished,
    _undecorate_all_saved_tensors,
)
from .graph_traversal import (
    _add_output_layers,
    _find_output_ancestors,
    _mark_layer_depths,
    _remove_orphan_nodes,
    _resolve_output_parent_labels,
)
from .labeling import (
    _build_lookup_keys_and_finalize_retained_layers,
    _log_final_info_for_layers,
    _map_raw_labels_to_final_labels,
    _rename_model_history_layer_names,
)
from .loop_detection import _detect_and_label_loops, _group_by_shared_params
from .loop_grouping_adapter import (
    RecurrenceAssignment,
    RecurrenceGroupingGraph,
    RecurrenceNode,
    group_recurrent_nodes,
)
from .saved_summary import refresh_pinned_autograd_memory, refresh_saved_module_call_count

if TYPE_CHECKING:
    from .._data_classes.trace import Trace

from .._quantities import Bytes

__all__ = [
    "RecurrenceAssignment",
    "RecurrenceGroupingGraph",
    "RecurrenceNode",
    "group_recurrent_nodes",
    "postprocess",
]
from .._utils.display import _vprint, _vtimed, user_stacklevel

#: The executor resolves every step callable through THIS module namespace
#: at call time (late binding, design-ppdag-v3 §5.2) — the names below are
#: consumed via getattr, not textual reference, and several are load-bearing
#: monkeypatch seams (tests patch torchlens._postprocess._add_output_layers,
#: _rename_model_history_layer_names, _find_output_ancestors, ...).
_EXECUTOR_STEP_NAMESPACE: tuple[object, ...] = (
    _add_output_layers,
    _find_output_ancestors,
    _remove_orphan_nodes,
    _mark_layer_depths,
    _mark_conditional_branches,
    _fix_buffer_layers,
    _detect_and_label_loops,
    _group_by_shared_params,
    _map_raw_labels_to_final_labels,
    _log_final_info_for_layers,
    _rename_model_history_layer_names,
    _build_lookup_keys_and_finalize_retained_layers,
    _undecorate_all_saved_tensors,
    _is_cuda_available,
    _log_time_elapsed,
    _finalize_param_logs,
    _build_layer_logs,
    _build_module_logs,
    refresh_saved_module_call_count,
    populate_normalized_layer_addresses,
    compute_graph_shape_hash,
    _finalize_streamed_bundle,
    _evict_streamed_outs,
)


def _unattributed_op_offenders(ops: Any) -> list[str]:
    """Return ``"label (positions)"`` rows for non-output ops with source-less args."""

    offenders: list[str] = []
    for op in ops:
        if getattr(op, "type", None) == "output":
            continue
        positions = tuple(getattr(op, "unattributed_tensor_args", ()) or ())
        if not positions:
            continue
        label = getattr(op, "label", None) or getattr(op, "layer_label", None) or op._label_raw
        offenders.append(f"{label} ({', '.join(positions)})")
    return offenders


def _pop_boundary_adoptions(self: "Trace", field_name: str, suffix: str) -> list[str]:
    """Pop one module-boundary adoption queue into human-readable offender rows."""

    return [
        f"{label} (adopted at module {boundary} {module_address}{suffix})"
        for label, boundary, module_address in (self.__dict__.pop(field_name, None) or ())
    ]


def _warn_unattributed_tensor_args(self: "Trace") -> None:
    """Warn once for tensor arguments without graph/source provenance.

    Parameters
    ----------
    self:
        Trace being postprocessed.

    Returns
    -------
    None
        Emits at most one aggregate warning.
    """

    from .._capture_honesty import (
        ADVISORY_MODULE_BOUNDARY_ADOPTION,
        ADVISORY_ORPHAN_UNATTRIBUTED_ARGS,
        ADVISORY_UNRECORDED_OPERATOR_MUTATION,
        append_capture_advisory,
    )

    offenders = _unattributed_op_offenders(getattr(self, "layer_list", ()) or ())
    # Orphan pruning (step 3) runs BEFORE this step and strips the pruned ops, so a
    # source-less tensor whose only consumer was pruned (``G.sum().item()``, a
    # control-flow predicate on a global) would lose its witness with the op;
    # ``_remove_orphan_nodes`` reads the witness before removal.
    orphans = list(self.__dict__.pop("_orphan_unattributed_tensor_args", None) or ())
    self.__dict__.pop("_plain_scalar_escape_labels", None)
    # A direct data read (``G.tolist()``, ``float(G)``) of a source-less tensor has
    # no op at all to carry the witness; the plain escape belt queued it.
    orphans += list(self.__dict__.pop("_plain_direct_escape_gaps", None) or ())
    # R16: module-entry adoptions of untagged tensors (outside disclosed
    # transform/dynamo regions) are the module-consumed twin of the
    # unattributed-args case; without this fold, a stale-ref escape whose
    # output was first consumed by a MODULE was laundered into a clean
    # ``internalsource`` node -- no warning, no rescue,
    # consumption-order-dependent disclosure. Module-EXIT adoptions are the
    # module-returned twin (a stale ref whose output a module returns, e.g.
    # transformers' ``GELUActivation``): the boundary op tagged it before any
    # consumer could flag it.
    adoptions = _pop_boundary_adoptions(self, "_module_boundary_adoptions", "")
    # Closure/forward-global tensors a module consumes or returns directly: no
    # escape (they predate the forward), but no source either.
    outside = _pop_boundary_adoptions(
        self, "_module_boundary_outside_sources", ", closure or forward-global tensor"
    )
    # Direct ``torch.ops`` calls that wrote an argument the recorder could not record
    # in place (``_torch_ops_calls._disclose_unrecorded_mutation``).
    mutations = list(self.__dict__.pop("_unrecorded_operator_mutations", None) or ())
    if not (offenders or orphans or adoptions or outside or mutations):
        return
    # Adopted tensors leave a functionless internal-source node that replays and
    # validates, and pruned ops leave nothing at all; persist both so forward
    # validation fails them (retained per-op positions already persist as
    # ``Op.unattributed_tensor_args``). Rows are written before the warning,
    # which a warning filter may raise.
    append_capture_advisory(self, ADVISORY_MODULE_BOUNDARY_ADOPTION, adoptions + outside)
    append_capture_advisory(self, ADVISORY_ORPHAN_UNATTRIBUTED_ARGS, orphans)
    append_capture_advisory(self, ADVISORY_UNRECORDED_OPERATOR_MUTATION, mutations)
    if offenders or orphans or adoptions:
        # Session-time escape signal: the capture entry reads this flag to decide
        # whether a rescue re-run (TorchFunctionMode net) should be attempted.
        self._had_unattributed_tensor_args = True
    warnings.warn(
        "TorchLens found tensor arguments with no graph/source provenance. "
        "These are usually tensors captured from outside the traced model; "
        "module tensor attributes, inputs, parameters, and buffers are known sources. "
        "Offending ops/arg positions: "
        + "; ".join(offenders + orphans + adoptions + outside + mutations),
        UserWarning,
        stacklevel=2,
    )


def _populate_var_names(self: "Trace") -> None:
    """Populate source assignment names for captured operation call sites.

    Parameters
    ----------
    self:
        Trace being postprocessed.
    """

    if not getattr(self, "save_code_context", False):
        return
    for op in getattr(self, "layer_list", ()):
        if getattr(op, "type", None) == "output":
            op.var_names = []
            continue
        op.var_names = resolve_var_names(
            getattr(op, "code_context", []) or [],
            getattr(op, "func_name", None),
        )


def _drop_transient_capture_state(self: "Trace") -> None:
    """Remove capture/session scratch that must not survive on final traces.

    Args:
        self: Trace whose postprocess-local state should be discarded.

    Returns:
        None. Mutates ``self.__dict__``.
    """

    keep_deferred_streaming = bool(
        self.__dict__.get("_defer_streaming_bundle_finalization", False)
        and self.__dict__.get("_out_writer") is not None
    )
    # WT1 A-IV item 18 (lane A08): a step-18 STAGED streamed bundle publishes
    # at the settlement seam (capture/trace.py), which runs after this scrub;
    # popping the writer here would strand the staged bundle in its temp dir.
    keep_staged_publish = bool(
        getattr(self.__dict__.get("_out_writer"), "staged_for_settlement", False)
    )
    keep_selective_sink = self.__dict__.get("_out_sink") is not None
    wrapper_ws = self.__dict__.get("_wrapper_runtime_ws")
    if wrapper_ws is not None:
        registry = getattr(wrapper_ws, "container_registry", None)
        if registry is not None:
            registry.clear_live_state()
    field_names = [
        "_raw_graph_ws",
        "_module_capture_ws",
        "_wrapper_runtime_ws",
        "capture_events",
        "_output_container_specs_by_raw_label",
        # B1-02: session-time semantic-output scratch. Two of the four pin live
        # user objects (an HF tokenizer, a model-derived metadata key), so this
        # seam is the postprocess-side belt for the capture-boundary drop in
        # ``capture/trace.py`` -- including the halted postprocess, which
        # reaches here through ``_finalize_halted_trace``.
        "_output_style",
        "_output_head",
        "_output_tokenizer",
        "_semantic_output_metadata",
        # B1-17 companion (INTERVENED axis): capture-time dedup caches for the
        # predicate-intervention spec/target mirrors. `..._target_keys` holds a
        # STRONG reference to the live intervention spec plus a frozen-target
        # set, so leaving it on the finished product retains capture-time
        # objects for no reason. Cleaned centrally here rather than in
        # `backends/torch/_ops_interventions.py` (another lane's file).
        "_tl_predicate_intervention_spec_keys",
        "_tl_predicate_intervention_target_keys",
    ]
    keep_writer = keep_deferred_streaming or keep_staged_publish
    if not keep_writer and not keep_selective_sink:
        field_names.extend(
            [
                "_out_writer",
                "_out_sink",
                "_keep_outs_in_memory",
                "_keep_grads_in_memory",
                "_grad_stream_retain_in_memory",
                "_defer_streaming_bundle_finalization",
            ]
        )
    elif not keep_writer:
        field_names.extend(
            [
                "_out_writer",
                "_keep_outs_in_memory",
                "_keep_grads_in_memory",
                "_grad_stream_retain_in_memory",
                "_defer_streaming_bundle_finalization",
            ]
        )
    for field_name in field_names:
        self.__dict__.pop(field_name, None)


def _refresh_fast_saved_summary(self: "Trace") -> None:
    """Refresh saved-output counters after retained layers are finalized.

    Args:
        self: Trace whose final retained layer entries were updated.

    Returns:
        None. Mutates aggregate saved-output fields on ``self``.
    """

    saved_layers = [
        layer_entry
        for layer_entry in self.layer_list
        if getattr(layer_entry, "has_saved_activation", False)
        and not getattr(layer_entry, "is_orphan", False)
    ]
    self.num_saved_ops = len(saved_layers)
    # F20 (brainpipe D-17): read the ONE byte model -- physically retained
    # alias-aware bytes -- never a sum of raw activation_memory fields. The
    # explorer P4 memory truth (reduce-only captures count the retained
    # TRANSFORMED bytes, raw-retaining captures the raw bytes) is served by
    # this same aggregate: it counts the payloads capture actually kept.
    from .._save_budget import retained_activation_bytes

    self.saved_activation_memory = Bytes(retained_activation_bytes(saved_layers))
    self.num_saved_layers = len({layer_entry.layer_label for layer_entry in saved_layers})
    refresh_saved_module_call_count(self, {layer_entry.label for layer_entry in saved_layers})
    refresh_pinned_autograd_memory(self, saved_layers)


def postprocess(
    self: "Trace", output_tensors: list[torch.Tensor], output_tensor_addresses: list[str]
) -> None:
    """Run the full postprocessing pipeline, with a TOTAL retention epilogue.

    Transforms the raw Trace captured during the forward pass into its
    final user-facing form.

    B1-18: the H2 hot-AST release used to be a success-path epilogue
    statement, so a postprocess failure -- and the zero-layer early return --
    left the process-wide file cache holding parsed ASTs, parent maps, and line
    splits for whole torch-library files. Distinct failing files accumulate up
    to the 256-file cache bound, which resurrects exactly the multi-megabyte
    retention class the seal exists to close. The release is a pure
    process-wide cache drop (spans and retained source survive; an unprojected
    scope re-parses), so it is safe on every exit and belongs in ``finally``.

    Parameters
    ----------
    output_tensors:
        Actual output tensors returned by the model's forward call.
    output_tensor_addresses:
        Hierarchical address strings for each output, for example ``"0.1"``
        for nested tuple outputs.
    """

    try:
        _postprocess_body(self, output_tensors, output_tensor_addresses)
    finally:
        ast_branches.release_parsed_asts()


def _postprocess_body(
    self: "Trace", output_tensors: list[torch.Tensor], output_tensor_addresses: list[str]
) -> None:
    """Run steps 0-20 and the core freeze (see :func:`postprocess`).

    Parameters
    ----------
    output_tensors:
        Actual output tensors returned by the model's forward call.
    output_tensor_addresses:
        Hierarchical address strings for each output.
    """

    capture_events = getattr(self, "capture_events", None)
    capture_session = None
    # Resolve each output tensor's graph parent BEFORE materializing events:
    # a registered buffer returned directly from forward() without ever being
    # used by a traced op has no graph node yet, and is logged here as a late
    # buffer source event so it materializes with everything else.
    output_parent_labels = _resolve_output_parent_labels(self, output_tensors)
    if capture_events is not None:
        self._raw_event_shape_hash = compute_raw_event_shape_hash(capture_events)
        capture_session = capture_session_for_events(capture_events)
        # Both branches read the AMENDED fold (never the raw list): the seal
        # folds via the reducer, and a session-detached journal (cooked
        # fastlog projection) folds directly. Seeding the working copy through
        # ``projected_op_events`` also filters the carried amendment lane to
        # seq > the seal watermark, so seal-folded knowledge cannot apply
        # twice while genuinely-new pre-0 amendments keep riding the copy.
        sealed_op_events = (
            [_clone_op_event_for_replay(event) for event in capture_session.seal().events]
            if capture_session is not None
            else list(capture_events.amended_op_records())
        )
        working_events = capture_events._copy_for_replay(projected_op_events=sealed_op_events)
        self._capture_events = working_events
        with _vtimed(self, "  Step 0: Materialize capture events"):
            materialize_from_events(self, working_events)
        working_events.release_working_projection()
        delattr(self, "capture_events")

    # Guard: if the model produced no logged layers, skip postprocessing (#153)
    if len(self._raw_graph_ws.raw_layer_labels_list) == 0:
        import warnings

        # This is about the user's model, so blame the user's line, not this one.
        warnings.warn(
            "No layers were logged during the forward pass; skipping postprocessing.",
            stacklevel=user_stacklevel(),
        )
        _set_tracing_finished(self)
        _drop_transient_capture_state(self)
        if capture_events is not None:
            capture_events.release_runtime_sidecars()
            # The trace is the sole strong owner of its (sidecar-released)
            # event stream; the former _EVENT_STREAMS weak registry is gone.
            self.__dict__["_capture_events"] = capture_events
        return

    _vprint(
        self,
        f"Postprocessing {len(self._raw_graph_ws.raw_layer_labels_list):,} layers "
        f"({len(self.buffer_layers):,} buffers)...",
    )
    _post_t0 = time.time() if getattr(self, "verbose", False) else 0

    # Steps 1-20 run through the executor (_executor.py): each step body
    # resolves its callable through this module's namespace at call time.
    run_pipeline(
        StepContext(
            trace=self,
            output_tensors=list(output_tensors),
            output_tensor_addresses=list(output_tensor_addresses),
            output_parent_labels=output_parent_labels,
            capture_session=capture_session,
        )
    )

    # The compaction passes belong to the freeze (M11 fold): ancestor
    # closures intern into shared bitmaps and repeated immutable Op metadata
    # pools onto shared instances, right before the physical seal.
    _compact_ancestor_sets(self)
    _compact_op_metadata(self)
    _finalize_forward_primitive_profile(self)

    # The core freeze point (trace_core_design.md section 3.3): forward
    # topology froze logically at step 17, the payload plane settled through
    # step 20, and the ancestor closures were just interned. The M6 relation
    # conversion runs here — parents/children project into the core's
    # canonical dataflow edge table (differentially verified before the
    # staging cells die) and the remaining relation families become interned
    # immutable views — the M7 group and shared-fact conversions follow
    # (equivalence/recurrence GroupRefs, FunctionCall/ParamAlias fact
    # blocks) — then the Op row store seals (columnar transpose on large
    # traces). Later public writes land in the store's sparse overlay;
    # facade behavior is otherwise unchanged.
    _core = self.__dict__.get("_trace_core")
    if _core is not None and _core.ops is not None:
        _freeze_relation_views(self)
        _core.ops.freeze()
        # Payload-lifetime owner (fix/fork F4): when the LAST owning core
        # (this one plus any fork cores viewing the sealed base) is
        # garbage-collected, the store evicts tensor payload cells so a
        # retained Op facade stops pinning every captured activation.
        _core.ops.adopt_payload_owner(_core)
        # Adopt the trace-scoped FuncCallLocation records (cached per call
        # site in _code_context_cache) into their kind table before sealing.
        # Cache entries mix FuncCallLocation records with plain metadata, so
        # filter by type.
        from .._data_classes.func_call_location import FuncCallLocation as _FCL

        _fcl_seen: dict[int, object] = {}
        for _fcl_group in (self.__dict__.get("_code_context_cache") or {}).values():
            for _fcl in _fcl_group or ():
                if isinstance(_fcl, _FCL):
                    _fcl_seen.setdefault(id(_fcl), _fcl)
        if _fcl_seen:
            from .._trace_core.record_rows import adopt_records

            adopt_records(_core, "func_call_location", _fcl_seen.values())
        # The M8 non-Op kind tables (param/module/module_call/buffer/
        # func_call_location) seal with the same lifecycle: appends stop,
        # later writes keep landing in row cells via the sealed-store path.
        for _kind_store in _core.kind_rows.values():
            _kind_store.freeze()

    _vprint(self, f"Postprocessing complete ({time.time() - _post_t0:.2f}s)")
    _drop_transient_capture_state(self)
    if capture_events is not None:
        capture_events.release_runtime_sidecars()
        # The trace is the sole strong owner of its (sidecar-released) event
        # stream; the former _EVENT_STREAMS weak registry is gone. Sidecar
        # release already stripped payloads, native handles, and the
        # source_trace backrefs, so this strong edge closes no new cycle.
        self.__dict__["_capture_events"] = capture_events
    # H2 retention seal: steps 5/11.5 are done with the parsed ASTs. The drop
    # itself now lives in `postprocess`'s `finally` (B1-18) so a failing
    # pipeline and the zero-layer early return release it too; span data and
    # projected calls survive, so lazy post-capture queries
    # (Op.arg_expressions) stay re-parse-free in the common case and re-parse
    # from retained source otherwise.
