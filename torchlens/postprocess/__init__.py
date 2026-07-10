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

Step ordering invariants:
- Steps 1-3 MUST precede Step 5 (conditional branch detection needs orphan-free graph).
- Module suffixes must be present on equivalence_class before Step 7 loop detection.
- Step 7 MUST precede Step 8 (label generation needs recurrent_ops).
- Step 8 MUST precede Step 9 (final info logging uses finalized labels).
- Step 9 MUST precede Step 11 (lookup key generation needs module hierarchy data
  populated in Step 9).
- Step 10 (rename) MUST precede Step 11 (lookup keys use renamed labels).
- Step 15.5 (_build_layer_logs) MUST precede Step 16 (_build_module_logs) because
  Module.layers references Layer keys.

Fast mode skips Steps 1-9 and Step 16, reusing the exhaustive pass's metadata.
It only copies outs from parent to output nodes, trims, renames, builds
LayerLogs, and marks the pass as finished. See postprocess_fast() below.
"""

from dataclasses import dataclass
import os
from typing import TYPE_CHECKING, List, cast

import time
import torch
import warnings

from ..capture.session import capture_session_for_events
from ..utils.tensor_utils import _is_cuda_available, safe_copy
from ..utils.hashing import (
    compute_graph_shape_hash,
    compute_raw_event_shape_hash,
    populate_normalized_layer_addresses,
)

from .control_flow import (
    _fix_buffer_layers,
    _mark_conditional_branches,
)
from .finalization import (
    _build_layer_logs,
    _build_module_logs,
    _evict_streamed_outs,
    _finalize_streamed_bundle,
    _finalize_param_logs,
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
    _log_final_info_for_layers,
    _map_raw_labels_to_final_labels,
    _build_lookup_keys_and_finalize_retained_layers,
    _rename_model_history_layer_names,
)
from .loop_detection import _detect_and_label_loops, _group_by_shared_params
from .loop_grouping_adapter import (
    RecurrenceAssignment,
    RecurrenceGroupingGraph,
    RecurrenceNode,
    group_recurrent_nodes,
)
from .saved_summary import refresh_saved_module_call_count
from ._materialize import materialize_from_events
from .ast_branches import resolve_var_names

if TYPE_CHECKING:
    from ..data_classes.op import Op
    from ..data_classes.trace import Trace

from ..quantities import Bytes

__all__ = [
    "RecurrenceAssignment",
    "RecurrenceGroupingGraph",
    "RecurrenceNode",
    "group_recurrent_nodes",
    "postprocess",
    "postprocess_fast",
]
from ..utils.display import _vprint, _vtimed
from ..captured_run import remember_event_stream


_POSTPROCESS_ASSERT_ENV = "TORCHLENS_POSTPROCESS_ASSERTIONS"


@dataclass(frozen=True)
class PostprocessStepContract:
    """Declared contract for one postprocess pipeline step.

    Parameters
    ----------
    step:
        Stable step identifier used by the pipeline.
    name:
        Human-readable step name.
    contract:
        Short consumes/produces/mutation contract.
    """

    step: str
    name: str
    contract: str


POSTPROCESS_STEP_CONTRACTS: dict[str, PostprocessStepContract] = {
    "0": PostprocessStepContract(
        "0",
        "Materialize capture events",
        "Consumes capture events; rebuilds raw Op state; mutates Trace in place.",
    ),
    "1": PostprocessStepContract(
        "1",
        "Add output layers",
        "Consumes model outputs and parent labels; produces output Ops in raw state.",
    ),
    "2": PostprocessStepContract(
        "2",
        "Trace output ancestors",
        "Consumes raw graph links; mutates output-descendant ancestry flags in place.",
    ),
    "3": PostprocessStepContract(
        "3",
        "Remove orphan nodes",
        "Consumes ancestry flags; removes or records orphan raw Ops in place.",
    ),
    "4": PostprocessStepContract(
        "4",
        "Input/output distances",
        "Consumes orphan-free graph; mutates distance fields in place.",
    ),
    "5": PostprocessStepContract(
        "5",
        "Mark conditional branches",
        "Consumes orphan-free graph; mutates conditional metadata in place.",
    ),
    "6": PostprocessStepContract(
        "6",
        "Fix buffer layers",
        "Consumes buffer events and graph links; mutates buffer metadata in place.",
    ),
    "7": PostprocessStepContract(
        "7",
        "Loop detection",
        "Consumes final raw graph structure; mutates recurrence/equivalence metadata.",
    ),
    "8": PostprocessStepContract(
        "8",
        "Map labels",
        "Consumes raw labels and recurrence metadata; produces raw-to-final maps.",
    ),
    "9": PostprocessStepContract(
        "9",
        "Log final info",
        "Consumes mapped labels; mutates final Op metadata and module build data.",
    ),
    "10": PostprocessStepContract(
        "10",
        "Rename labels",
        "Consumes raw-to-final maps; mutates graph references to final labels.",
    ),
    "11": PostprocessStepContract(
        "11",
        "Build lookup keys",
        "Consumes final labels; rebuilds final lookup containers in place.",
    ),
    "11.5": PostprocessStepContract(
        "11.5",
        "Populate source var names",
        "Consumes code context; mutates Op var_names in place.",
    ),
    "12": PostprocessStepContract(
        "12",
        "Undecorate tensors",
        "Consumes saved tensors; mutates payload wrappers in place.",
    ),
    "13": PostprocessStepContract(
        "13",
        "Clear CUDA cache",
        "Runs optional CUDA allocator cleanup; leaves Trace metadata unchanged.",
    ),
    "14": PostprocessStepContract(
        "14",
        "Log timing",
        "Consumes capture timestamps; mutates duration fields in place.",
    ),
    "15": PostprocessStepContract(
        "15",
        "Finalize params",
        "Consumes Op param references; mutates Param reverse mappings.",
    ),
    "15.5": PostprocessStepContract(
        "15.5",
        "Build layer logs",
        "Consumes final Op list; rebuilds aggregate Layer logs and pass index.",
    ),
    "16": PostprocessStepContract(
        "16",
        "Build module logs",
        "Consumes module build data and layer logs; rebuilds Module/ModuleCall logs.",
    ),
    "16.5": PostprocessStepContract(
        "16.5",
        "Graph shape hash",
        "Consumes final graph; mutates normalized addresses and graph hash.",
    ),
    "17": PostprocessStepContract(
        "17",
        "Mark pass finished",
        "Consumes finalized containers; mutates Trace to user-facing finished state.",
    ),
    "18": PostprocessStepContract(
        "18",
        "Finalize streamed bundle",
        "Consumes stream writer state; finalizes bundle metadata in place.",
    ),
    "19": PostprocessStepContract(
        "19",
        "Evict streamed outs",
        "Consumes finalized stream state; drops in-memory output payloads.",
    ),
    "20": PostprocessStepContract(
        "20",
        "Release param refs",
        "Consumes finalized Param logs; drops live parameter references in place.",
    ),
}


def _postprocess_assertions_enabled() -> bool:
    """Return whether postprocess boundary assertions are enabled.

    Returns
    -------
    bool
        ``True`` when ``TORCHLENS_POSTPROCESS_ASSERTIONS`` is set to a truthy value.
    """

    return os.environ.get(_POSTPROCESS_ASSERT_ENV, "").lower() in {"1", "true", "yes", "on"}


def _assert_postprocess_contract(self: "Trace", step: str) -> None:
    """Assert cheap postconditions for one completed postprocess step.

    Parameters
    ----------
    self:
        Trace being postprocessed.
    step:
        Step identifier from ``POSTPROCESS_STEP_CONTRACTS``.
    """

    if not _postprocess_assertions_enabled():
        return
    if step == "1":
        assert self.output_layers, "Step 1 must register output layers"
    elif step == "8":
        assert self._raw_to_final_layer_labels, "Step 8 must build raw-to-final layer labels"
        assert self._raw_to_final_op_labels, "Step 8 must build raw-to-final op labels"
    elif step == "11":
        for op in self.layer_list:
            assert op.label, f"Step 11 left {op!r} without a final op label"
            assert op.layer_label, f"Step 11 left {op!r} without a final layer label"
            assert op.lookup_keys, f"Step 11 left {op.label} without lookup keys"
            assert self.layer_dict_all_keys[op.label] is op
            assert self.layer_dict_all_keys[op.layer_label] is op
    elif step == "15.5":
        assert self.layer_logs, "Step 15.5 must build aggregate layer logs"
        assert len(self.layer_logs) == len(self.layer_labels)
        assert isinstance(self.by_pass, dict)
    elif step == "16":
        assert getattr(self, "_module_logs", None) is not None, "Step 16 must build module logs"
    elif step == "16.5":
        assert self.graph_shape_hash is not None, "Step 16.5 must compute graph_shape_hash"
    elif step == "17":
        assert self._tracing_finished is True, "Step 17 must mark tracing finished"


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

    offenders: list[str] = []
    for op in getattr(self, "layer_list", ()):
        if getattr(op, "type", None) == "output":
            continue
        positions = tuple(getattr(op, "unattributed_tensor_args", ()) or ())
        if not positions:
            continue
        label = getattr(op, "label", None) or getattr(op, "layer_label", None) or op._label_raw
        offenders.append(f"{label} ({', '.join(positions)})")
    if not offenders:
        return
    warnings.warn(
        "TorchLens found tensor arguments with no graph/source provenance. "
        "These are usually tensors captured from outside the traced model; "
        "module tensor attributes, inputs, parameters, and buffers are known sources. "
        "Offending ops/arg positions: " + "; ".join(offenders),
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
    keep_selective_sink = self.__dict__.get("_out_sink") is not None
    build_state = self.__dict__.get("_build_state")
    if build_state is not None:
        registry = getattr(build_state, "container_registry", None)
        if registry is not None:
            registry.clear_live_state()
    field_names = [
        "_build_state",
        "capture_events",
        "_output_container_specs_by_raw_label",
    ]
    if not keep_deferred_streaming and not keep_selective_sink:
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
    elif not keep_deferred_streaming:
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
    """Refresh saved-output counters after a fast replay pass.

    Args:
        self: Trace whose final layer entries were updated in fast mode.

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
    self.saved_activation_memory = Bytes(
        sum(int(getattr(layer_entry, "activation_memory", 0) or 0) for layer_entry in saved_layers)
    )
    self.num_saved_layers = len({layer_entry.layer_label for layer_entry in saved_layers})
    refresh_saved_module_call_count(self, {layer_entry.label for layer_entry in saved_layers})


def postprocess(
    self: "Trace", output_tensors: List[torch.Tensor], output_tensor_addresses: List[str]
) -> None:
    """Run the full postprocessing pipeline in exhaustive mode.

    Transforms the raw Trace captured during the forward pass into its
    final user-facing form. In fast mode, delegates to ``postprocess_fast``
    which skips most steps (graph traversal, loop detection, module annotation)
    and only copies outs, trims fields, and builds LayerLogs.

    Parameters
    ----------
    output_tensors:
        Actual output tensors returned by the model's forward call.
    output_tensor_addresses:
        Hierarchical address strings for each output, for example ``"0.1"``
        for nested tuple outputs.
    """
    if self.capture_mode == "fast":
        postprocess_fast(self)
        return

    capture_events = getattr(self, "capture_events", None)
    # Resolve each output tensor's graph parent BEFORE materializing events:
    # a registered buffer returned directly from forward() without ever being
    # used by a traced op has no graph node yet, and is logged here as a late
    # buffer source event so it materializes with everything else.
    output_parent_labels = _resolve_output_parent_labels(self, output_tensors)
    if capture_events is not None:
        self._raw_event_shape_hash = compute_raw_event_shape_hash(capture_events)
        remember_event_stream(self, capture_events)
        capture_session = capture_session_for_events(capture_events)
        sealed_op_events = (
            [fact.event for fact in capture_session.event_journal.facts]
            if capture_session is not None
            else list(capture_events.op_events)
        )
        working_events = capture_events.copy_for_replay()
        working_events.op_events = sealed_op_events
        working_events.op_event_by_label_raw = {
            event.label_raw: event for event in sealed_op_events
        }
        working_events.op_event_index_by_label_raw = {
            event.label_raw: index for index, event in enumerate(sealed_op_events)
        }
        self._capture_events = working_events
        with _vtimed(self, "  Step 0: Materialize capture events"):
            materialize_from_events(self, working_events)
        _assert_postprocess_contract(self, "0")
        delattr(self, "capture_events")

    # Guard: if the model produced no logged layers, skip postprocessing (#153)
    if len(self._raw_layer_labels_list) == 0:
        import warnings

        warnings.warn("No layers were logged during the forward pass; skipping postprocessing.")
        _set_tracing_finished(self)
        _drop_transient_capture_state(self)
        return

    _vprint(
        self,
        f"Postprocessing {len(self._raw_layer_labels_list):,} layers "
        f"({len(self.buffer_layers):,} buffers)...",
    )
    _post_t0 = time.time() if getattr(self, "verbose", False) else 0

    # Step 1: Add dedicated output nodes
    with _vtimed(self, "  Step 1: Add output layers"):
        _add_output_layers(self, output_tensors, output_tensor_addresses, output_parent_labels)
    _assert_postprocess_contract(self, "1")

    # Step 2: Trace which nodes are ancestors of output nodes
    with _vtimed(self, "  Step 2: Trace output ancestors"):
        _find_output_ancestors(self)
    _assert_postprocess_contract(self, "2")

    # Step 3: Remove orphan nodes, find nodes that don't terminate in output node
    with _vtimed(self, "  Step 3: Remove orphan nodes"):
        _remove_orphan_nodes(self)
    _assert_postprocess_contract(self, "3")

    # Step 4: Find min/max distance from input and output nodes.
    # Conditional: only runs when the user requested distance metadata.
    if self.mark_layer_depths:
        with _vtimed(self, "  Step 4: Input/output distances"):
            _mark_layer_depths(self)
        _assert_postprocess_contract(self, "4")

    # Step 5: Starting from terminal single boolean tensors, mark the conditional branches.
    with _vtimed(self, "  Step 5: Mark conditional branches"):
        _mark_conditional_branches(self)
    _assert_postprocess_contract(self, "5")

    # Step 6: Fix the buffer ops and parent information.
    with _vtimed(self, "  Step 6: Fix buffer layers"):
        _fix_buffer_layers(self)
    _assert_postprocess_contract(self, "6")

    # Step 7: Identify all loops, mark repeated layers.
    loop_desc = (
        "  Step 7: Loop detection (full)"
        if self.recurrence_detection
        else "  Step 7: Loop detection (params only)"
    )
    with _vtimed(self, loop_desc):
        if self.recurrence_detection:
            _detect_and_label_loops(self)
        else:
            _group_by_shared_params(self)
    _assert_postprocess_contract(self, "7")

    # Step 8: Go down tensor list, get the mapping from raw tensor names to final tensor names.
    with _vtimed(self, "  Step 8: Map labels"):
        _map_raw_labels_to_final_labels(self)
    _assert_postprocess_contract(self, "8")

    # Step 9: Log final info for all layers
    with _vtimed(self, "  Step 9: Log final info"):
        _log_final_info_for_layers(self)
    _assert_postprocess_contract(self, "9")

    # Step 10: Rename all raw labels to final labels
    with _vtimed(self, "  Step 10: Rename labels"):
        _rename_model_history_layer_names(self)
    _assert_postprocess_contract(self, "10")

    # Step 11: Build lookup keys and finalize retained layer lists
    with _vtimed(self, "  Step 11: Build lookup keys"):
        _build_lookup_keys_and_finalize_retained_layers(self)
        _warn_unattributed_tensor_args(self)
    _assert_postprocess_contract(self, "11")

    # Step 11.5: Populate source assignment names from full-file AST context.
    with _vtimed(self, "  Step 11.5: Populate source var names"):
        _populate_var_names(self)
    _assert_postprocess_contract(self, "11.5")

    # Step 12: Undecorate all saved tensors and remove saved grad_fns.
    with _vtimed(self, "  Step 12: Undecorate tensors"):
        _undecorate_all_saved_tensors(self)
    _assert_postprocess_contract(self, "12")

    # Step 13: Clear the cache after any tensor deletions for garbage collection purposes.
    # Gated behind cached cuda.is_available() so CPU-only runs don't pay the
    # CUDA driver / NVML probe cost (per profiling audit 2026-04-27 finding #4).
    if _is_cuda_available():
        torch.cuda.empty_cache()
    _assert_postprocess_contract(self, "13")

    # Step 14: Log time elapsed.
    with _vtimed(self, "  Step 14: Log timing"):
        _log_time_elapsed(self)
    _assert_postprocess_contract(self, "14")

    # Step 15: Populate Param reverse mappings, linked params, num_calls, and grad metadata.
    with _vtimed(self, "  Step 15: Finalize params"):
        _finalize_param_logs(self)
    _assert_postprocess_contract(self, "15")

    # Step 15.5: Build aggregate Layer objects from per-pass Op entries.
    with _vtimed(self, "  Step 15.5: Build layer logs"):
        _build_layer_logs(self)
        self.by_pass = {}
        for index, op in enumerate(self.layer_list):
            pass_index = getattr(op, "pass_index", None)
            if pass_index is not None:
                self.by_pass.setdefault(pass_index, []).append(index)
    _assert_postprocess_contract(self, "15.5")

    # Step 16: Build structured Module objects from raw module_* dicts.
    with _vtimed(self, "  Step 16: Build module logs"):
        _build_module_logs(self)
        refresh_saved_module_call_count(self)
    _assert_postprocess_contract(self, "16")

    # Step 16.5: Compute graph shape hash before _set_tracing_finished changes access behavior.
    with _vtimed(self, "  Step 16.5: Graph shape hash"):
        populate_normalized_layer_addresses(self)
        self.graph_shape_hash = compute_graph_shape_hash(self)
    _assert_postprocess_contract(self, "16.5")

    # Step 17: log the pass as finished, changing the Trace behavior to its user-facing version.
    with _vtimed(self, "  Step 17: Mark pass finished"):
        _set_tracing_finished(self)
    _assert_postprocess_contract(self, "17")

    build_state = self.__dict__.get("_build_state")
    if build_state is not None:
        registry = getattr(build_state, "container_registry", None)
        if registry is not None:
            if registry.records:
                self.__dict__["_containers"] = dict(registry.records)
            registry.clear_live_state()

    for field_name in ("_build_state", "capture_events", "_output_container_specs_by_raw_label"):
        self.__dict__.pop(field_name, None)

    should_finalize_streaming = getattr(self, "_out_writer", None) is not None and not getattr(
        self, "_defer_streaming_bundle_finalization", False
    )
    if should_finalize_streaming:
        with _vtimed(self, "  Step 18: Finalize streamed bundle"):
            _finalize_streamed_bundle(self)
        _assert_postprocess_contract(self, "18")

    if should_finalize_streaming and not self._keep_outs_in_memory:
        with _vtimed(self, "  Step 19: Evict streamed outs"):
            _evict_streamed_outs(self)
        _assert_postprocess_contract(self, "19")

    with _vtimed(self, "  Step 20: Release param refs"):
        self.release_param_refs(allow_iter_rehydrate=True)
    _assert_postprocess_contract(self, "20")

    if getattr(self, "verbose", False):
        print(f"[torchlens] Postprocessing complete ({time.time() - _post_t0:.2f}s)")
    _drop_transient_capture_state(self)


def postprocess_fast(self: "Trace") -> None:
    """Lightweight postprocessing for fast (second-pass) logging mode.

    The fast pass reuses the exhaustive pass's graph structure, loop groupings,
    and labels. It only needs to:
    1. Copy out data from each output's parent tensor into the output node.
    2. Trim and reorder fields.
    3. Refresh saved-output summaries.
    4. Undecorate saved tensors.
    5. Build Layer aggregates.
    6. Mark the pass as finished.

    Skipped steps (already computed by the exhaustive pass):
    - Steps 1-4: Graph structure (output nodes, ancestry, orphans, distances)
    - Steps 5-6: Conditional branches and buffer fixing
    - Step 7: Loop detection
    - Steps 8-9: Label mapping and final info logging
    - Step 16: _build_module_logs — module structure doesn't change between
      ops and _module_build_data isn't repopulated in fast mode (#108).
    """
    _vprint(self, "Fast-pass postprocessing...")
    # Use layer_dict_main_keys to get Op directly (not Layer)
    for output_layer_label in self.output_layers:
        output_layer = cast("Op", self.layer_dict_all_keys[output_layer_label])
        if not output_layer.parents:
            continue  # Guard for parentless output layers (#152)
        parent_layer = cast("Op", self.layer_dict_all_keys[output_layer.parents[0]])
        parent_contents = parent_layer.out
        parent_transformed = parent_layer.transformed_out
        output_layer._internal_set(
            "out",
            safe_copy(parent_contents, detach_tensor=self.detach_saved_activations)
            if parent_contents is not None
            else None,
        )
        output_layer._internal_set(
            "transformed_out",
            safe_copy(parent_transformed, detach_tensor=self.detach_saved_activations)
            if isinstance(parent_transformed, torch.Tensor)
            else parent_transformed,
        )
        output_layer.activation_memory = parent_layer.activation_memory
        output_layer.transformed_out_shape = parent_layer.transformed_out_shape
        output_layer.transformed_out_dtype = parent_layer.transformed_out_dtype
        output_layer.transformed_activation_memory = parent_layer.transformed_activation_memory
        output_layer.has_saved_activation = parent_layer.has_saved_activation
        output_layer.has_grad = parent_layer.has_grad
        output_layer._internal_set("grad", parent_layer.grad)
        output_layer._internal_set("transformed_grad", parent_layer.transformed_grad)
        output_layer.transformed_grad_shape = parent_layer.transformed_grad_shape
        output_layer.transformed_grad_dtype = parent_layer.transformed_grad_dtype
        output_layer.transformed_gradient_memory = parent_layer.transformed_gradient_memory
    _refresh_fast_saved_summary(self)
    _undecorate_all_saved_tensors(self)

    # Gated behind cached cuda.is_available() so CPU-only fast-pass runs don't
    # pay the CUDA driver / NVML probe cost.
    if _is_cuda_available():
        torch.cuda.empty_cache()
    _log_time_elapsed(self)
    _build_layer_logs(self)
    # Note: _build_module_logs is NOT called here because module structure
    # doesn't change between ops and _module_build_data isn't repopulated
    # in fast mode (Step 9 is skipped). Existing module logs remain valid. (#108)
    if self.intervention_ready and not getattr(self, "save_arg_templates", False):
        self.intervention_ready = False
    populate_normalized_layer_addresses(self)
    self.graph_shape_hash = compute_graph_shape_hash(self)
    _set_tracing_finished(self)

    should_finalize_streaming = getattr(self, "_out_writer", None) is not None and not getattr(
        self, "_defer_streaming_bundle_finalization", False
    )
    if should_finalize_streaming:
        _finalize_streamed_bundle(self)
    if should_finalize_streaming and not self._keep_outs_in_memory:
        _evict_streamed_outs(self)
    _drop_transient_capture_state(self)
