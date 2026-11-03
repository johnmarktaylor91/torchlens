"""The grouped ``CaptureOptions`` each per-chunk recursive capture receives (PREPARE stage).

Moved verbatim from ``prepare.chunked`` (where the capture seam had moved it from
``prepare.door``) when the two batch J lanes that appended to ``chunked.py`` put it over
its file cap; the import graph is unchanged (same imports, same one caller in ``door``).
"""

from __future__ import annotations

from typing import Any

from ..._deprecations import MISSING
from ...options import CaptureOptions, merge_capture_options
from .request import _DoorState


def _explicit_or_missing(capture_options: CaptureOptions, field_name: str) -> Any:
    """The field's value when the user set it explicitly, else ``MISSING``."""
    if capture_options._is_field_explicit(field_name):
        return getattr(capture_options, field_name)
    return MISSING


# Session knobs the chunk path historically DROPPED (list-A row 26,
# second clause): the recursive options object was rebuilt from a
# hand-enumerated field list, so a chunked capture silently reset
# save_budget (and the other session-time knobs) to defaults. The
# recursive constructor must cover EVERY CaptureOptions field;
# tests/test_capopts_truth_chunk_options.py pins the full-coverage
# set difference so a future field cannot silently drop here.
def _recursive_capture_options(state: _DoorState) -> CaptureOptions:
    """The grouped options each per-chunk recursive capture receives."""
    opts = state.capture_options
    rebuilt = CaptureOptions(
        **_copied_capture_fields(opts), **_routed_capture_fields(state, opts, state.save_options)
    )
    # The resolved save= selection rides the private slot, never a field.
    return merge_capture_options(capture=rebuilt, _layers_to_save=state.layers_to_save)


def _copied_capture_fields(opts: CaptureOptions) -> dict[str, Any]:
    """The ``CaptureOptions`` fields the chunk path copies unchanged from the caller's options.

    Together with :func:`_routed_capture_fields` this names every field exactly once;
    tests/test_capopts_truth_chunk_options.py pins that.
    """
    return {
        "batch_render": opts.batch_render,
        "output_transform": opts.output_transform,
        "layer_visualizers": opts.layer_visualizers,
        "save_visualizations": opts.save_visualizations,
        "keep_orphans": opts.keep_orphans,
        "output_device": opts.output_device,
        "save_args": opts.save_args,
        "capture_tensor_grad_hooks": opts.capture_tensor_grad_hooks,
        "save_source_context": opts.save_source_context,
        "save_rng": opts.save_rng,
        "seed": opts.seed,
        "source_context_lines": opts.source_context_lines,
        "optimizer": opts.optimizer,
        "compute_input_output_distances": opts.compute_input_output_distances,
        "detach_saved_activations": opts.detach_saved_activations,
        "recurrence_detection": opts.recurrence_detection,
        "replayable": opts.replayable,
        "capture_container_structure": opts.capture_container_structure,
        "verbose": opts.verbose,
        "cache_dir": opts.cache_dir,
        "module_identity_mode": opts.module_identity_mode,
        "payload_policy": opts.payload_policy,
        "emit_nvtx": opts.emit_nvtx,
        "measure_python_peak_memory": opts.measure_python_peak_memory,
        "distributed_witness": opts.distributed_witness,
        "save_budget": opts.save_budget,
        "on_nonfinite": opts.on_nonfinite,
        "track_device_memory": opts.track_device_memory,
        "structure_only": opts.structure_only,
        "log_injections": opts.log_injections,
        "carry_state": opts.carry_state,
        "measure_host_peak_memory": opts.measure_host_peak_memory,
        "aten_ops": opts.aten_ops,
        "reconstruction_ready": opts.reconstruction_ready,
        "grouping": opts.grouping,
        "jax_static_argnums": opts.jax_static_argnums,
        "grad_options": opts.grad_options,
    }


def _routed_capture_fields(state: _DoorState, opts: CaptureOptions, save: Any) -> dict[str, Any]:
    """The fields the chunk path resolves, resets or reads from the door state instead of copying."""
    return {
        "input_transform": None,
        "save_raw_input": state.save_raw_input_policy,
        "save_transformed_output": state.save_transformed_output_policy,
        "save_grads": state.save_grads_policy,
        "hooks": MISSING,
        "unwrap_when_done": False,
        "backward_ready": state.train_mode_value,
        "inference_only": state.inference_only_value,
        "name": state.log_name,
        "cache": False,
        "jax_control_flow": _explicit_or_missing(opts, "jax_control_flow"),
        "jax_max_control_flow_unroll": _explicit_or_missing(opts, "jax_max_control_flow_unroll"),
        "activation_transform": save.activation_transform,
        "grad_transform": save.grad_transform,
        "save_raw_activations": save.save_raw_activations,
        "save_raw_grads": save.save_raw_gradients,
        "save_mode": state.save_mode_value,
        "lookback": state.args.lookback,
        "lookback_payload_policy": state.args.lookback_payload_policy,
        "chunk_size": None,
    }
