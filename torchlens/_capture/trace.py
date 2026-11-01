"""Forward-pass orchestration: runs the model, manages logging state, and saves outs.

This module implements the forward-pass architecture that TorchLens uses to extract
model outs:

1. **Exhaustive pass** (``capture_mode="exhaustive"``): Runs the model once,
   capturing every tensor operation's full metadata (shapes, dtypes, FLOPs,
   parent-child relationships, module context, etc.) into Op entries.
   This builds the complete computational graph.

2. **Predicate pass** (``capture_mode="predicate"``): Captures selectively
   while preserving the shared event journal and fixed-order kernel.

Key ordering constraint:
    RNG state must be captured/restored BEFORE ``active_logging()`` is entered,
    because the logging context manager itself may trigger decorated operations
    that consume RNG state.  See ``_pre_forward_rng_states`` handling.

Key functions:
    - ``normalize_input_args``: resolves tuple-vs-multi-arg ambiguity
    - ``safe_copy_args``: clones tensors to protect user inputs from in-place mutation
    - ``run_and_log_inputs_through_model``: the main entry point that orchestrates
      input setup, logging toggle, forward pass, output marking, and postprocessing
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass
from torchlens._capture._forward_helpers import (
    _backend_name_for_trace as _backend_name_for_trace,
    _capture_backend_from_registry as _capture_backend_from_registry,
    _cleanup_forward_memory_once as _cleanup_forward_memory_once,
    _clear_saved_activation_dedup_caches as _clear_saved_activation_dedup_caches,
    _forward_peak_memory_bracket as _forward_peak_memory_bracket,
    _run_predicate_forward_with_root_frame as _run_predicate_forward_with_root_frame,
    _structure_only_forward_boundary as _structure_only_forward_boundary,
    _weightsfree_and_belt_boundary as _weightsfree_and_belt_boundary,
)
from torchlens._capture._input_records import (
    _OPAQUE_INPUT_LEAF as _OPAQUE_INPUT_LEAF,
    _get_op_nums_from_user_labels as _get_op_nums_from_user_labels,
    _record_runnable_input_literal_leaves as _record_runnable_input_literal_leaves,
    _record_runnable_input_structure as _record_runnable_input_structure,
    _record_runnable_input_tensor_sites as _record_runnable_input_tensor_sites,
    _record_runnable_module_training_modes as _record_runnable_module_training_modes,
    _register_model_input_container_snapshots as _register_model_input_container_snapshots,
    _select_raw_indexes_by_retention_predicate as _select_raw_indexes_by_retention_predicate,
)
from torchlens._capture.finalize.settle import (
    _SEMANTIC_OUTPUT_TRANSIENT_FIELDS as _SEMANTIC_OUTPUT_TRANSIENT_FIELDS,
    _drop_semantic_output_transients as _drop_semantic_output_transients,
    _extract_and_mark_outputs as _extract_and_mark_outputs,
    _finalize_halted_trace as _finalize_halted_trace,
    _publish_streamed_bundle_at_settlement as _publish_streamed_bundle_at_settlement,
    _scrub_failed_capture_transients as _scrub_failed_capture_transients,
    _settle_interrupted_halted_arm as _settle_interrupted_halted_arm,
    save_new_outs as save_new_outs,
)
from torchlens._capture.run import (
    _ACTIVE_CAPTURE_BACKEND as _ACTIVE_CAPTURE_BACKEND,
    _AUTO_SEED_ENTROPY as _AUTO_SEED_ENTROPY,
    _fetch_label_move_input_tensors as _fetch_label_move_input_tensors,
    run_and_log_inputs_through_model as run_and_log_inputs_through_model,
)
