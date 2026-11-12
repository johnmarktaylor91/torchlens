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
  ParamLogs, build Layer/Module aggregates, seal the Op store (step
  ``seal_op_store``, body ``_seal_op_store`` here), mark pass as finished, then
  finalize any streamed bundle, optionally evict in-memory outs, and release
  live parameter references.

Step order is load-bearing: ``_pipeline.PIPELINE`` lists the named steps in the
order they run (tuple position is the order; no step carries a rank), and a
reorder is verified by the byte-identity oracles. Warnings and first-exception
order are pinned by that order.

Out-of-model, disclosed: deferred gradient streaming re-runs step-18/19-equivalent
code after backward, outside the pipeline.
"""

import gc
import time
from typing import TYPE_CHECKING

import torch

from .._capture.session import capture_session_for_events
from .._data_classes._compaction import compact_op_metadata as _compact_op_metadata
from .._data_classes._param_pin import pin_params
from .._ir.capture_events import _clone_op_event_for_replay
from .._trace_core.relation_views import freeze_trace_relation_views as _freeze_relation_views
from .._trace_core.view_tables import release_views
from .._utils.hashing import (
    compute_raw_event_shape_hash,
)
from ..backends.torch._ops_capture_records import _compact_ancestor_sets
from . import ast_branches
from ._executor import StepContext
from ._materialize import materialize_from_events
from ._op_seal import fold_object_overlay
from ._pipeline import run_pipeline
from ._primitive_profile import _finalize_forward_primitive_profile
from ._row_ingest import hand_off_capture_rows
from .finalization import (
    _set_tracing_finished,
)
from .graph_traversal import (
    _resolve_output_parent_labels,
)
from .loop_grouping_adapter import (
    RecurrenceAssignment,
    RecurrenceGroupingGraph,
    RecurrenceNode,
    group_recurrent_nodes,
)

if TYPE_CHECKING:
    from .._data_classes.trace import Trace

__all__ = [
    "RecurrenceAssignment",
    "RecurrenceGroupingGraph",
    "RecurrenceNode",
    "group_recurrent_nodes",
    "postprocess",
]
from .._utils.display import _vprint, _vtimed, user_stacklevel


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
    # A step-18 STAGED streamed bundle publishes
    # at the settlement seam (_capture/trace.py), which runs after this scrub;
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
        # session-time semantic-output scratch. Both pin live
        # user objects (an HF tokenizer, a model-derived metadata key), so this
        # seam is the postprocess-side belt for the capture-boundary drop in
        # ``_capture/trace.py`` -- including the halted postprocess, which
        # reaches here through ``_finalize_halted_trace``.
        "_output_tokenizer",
        "_semantic_output_metadata",
        # Capture-time dedup caches for the
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


def _seal_op_store(self: "Trace") -> None:
    """Pipeline step ``seal_op_store``: compact, then seal the Op row store.

    Runs after ``adopt_containers_and_drop_workspaces`` (compaction pools the
    container tokens that step hands to the core) and before
    ``mark_tracing_finished``. Ancestor closures intern into shared bitmaps,
    repeated immutable Op metadata pools onto shared instances, and primitive
    rows adopt into the core. Then the relation conversion runs (trace_core_design.md
    section 3.3): parents/children project into the core's canonical dataflow
    edge table and the other relation families become interned immutable views,
    the group and shared-fact conversions follow, and the Op row store seals
    (columnar transpose on large traces). Later public writes land in the
    store's sparse overlay; facade behavior is otherwise unchanged. The body
    resolves each pass through this package's globals at call time.
    """

    _compact_ancestor_sets(self)
    _compact_op_metadata(self)
    _finalize_forward_primitive_profile(self)
    core = self.__dict__.get("_trace_core")
    if core is None or core.ops is None:
        return
    _freeze_relation_views(self)
    core.ops.freeze()
    # Payload-lifetime owner: when the LAST owning core (this one plus any
    # fork cores viewing the sealed base) is garbage-collected, the store
    # evicts tensor payload cells so a retained Op facade stops pinning every
    # captured activation.
    core.ops.adopt_payload_owner(core)


def postprocess(
    self: "Trace", output_tensors: list[torch.Tensor], output_tensor_addresses: list[str]
) -> None:
    """Run the full postprocessing pipeline, with a TOTAL retention epilogue.

    Transforms the raw Trace captured during the forward pass into its
    final user-facing form.

    the H2 hot-AST release used to be a success-path epilogue
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

    # Automatic cyclic collection is paused for the call and restored to the
    # exact prior state on every exit (a caller who disabled it keeps it off;
    # no forced collect afterwards). Postprocess allocates hundreds of
    # thousands of short-lived containers that reference counting frees, and
    # each automatic collection the allocations trigger rescans the whole
    # heap (the model plus the trace being built): on a GPT-2 generate
    # trace that was about a quarter of postprocess wall time spent finding
    # almost nothing. Cyclic garbage made here waits for the next automatic
    # collection after the call. Same pattern as the fork-time cell isolation.
    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()
    try:
        _postprocess_body(self, output_tensors, output_tensor_addresses)
    finally:
        try:
            ast_branches.release_parsed_asts()
        finally:
            # The build is over on every exit: records are handed out weakly
            # from here on, and each one handed out holds this trace.
            release_views(self)
            pin_params(self)
            if gc_was_enabled:
                gc.enable()


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
        sealed_op_events, capture_rows = hand_off_capture_rows(
            capture_session, capture_events, _clone_op_event_for_replay
        )
        working_events = capture_events._copy_for_replay(projected_op_events=sealed_op_events)
        self._capture_events = working_events
        with _vtimed(self, "  Step 0: Materialize capture events"):
            materialize_from_events(self, working_events, capture_rows)
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
    _post_t0 = time.time() if self._options_verbose else 0

    # The named steps run through _pipeline.run_pipeline over PIPELINE: each
    # step body resolves its callable through its home module at call time.
    run_pipeline(
        StepContext(
            trace=self,
            output_tensors=list(output_tensors),
            output_tensor_addresses=list(output_tensor_addresses),
            output_parent_labels=output_parent_labels,
            capture_session=capture_session,
        )
    )

    # The op store sealed inside the pipeline (step ``seal_op_store``, ahead of
    # the finished mark). Post-seal writes of the steps after it (streamed
    # refs, evicted outs) fold back into the store's object columns, so an
    # evicted payload is not kept alive under its overlay entry.
    _core = self.__dict__.get("_trace_core")
    if _core is not None and _core.ops is not None:
        fold_object_overlay(_core.ops)
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
        # The non-Op kind tables (param/module/module_call/buffer/
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
    # itself now lives in `postprocess`'s `finally` so a failing
    # pipeline and the zero-layer early return release it too; span data and
    # projected calls survive, so lazy post-capture queries
    # (Op.arg_expressions) stay re-parse-free in the common case and re-parse
    # from retained source otherwise.
