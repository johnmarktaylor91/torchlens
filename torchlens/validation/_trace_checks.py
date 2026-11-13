"""Trace check functions reached from the records through ``_data_classes._doors.validation``.

``Trace.validate_forward_pass``, ``Trace.check_metadata_invariants``, ``Trace._save_new_outs`` and the
``Trace.validation_replay_status`` property are the thin methods. Validation is LOCKED: the bodies
moved verbatim and no check changed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from .._capture.finalize.settle import save_new_outs as _save_new_outs_impl
from .._capture.outcome import require_capture_capability
from .._capture.structure_only import require_structure_only_capability
from .._capture_state_helpers import unwrap_compiled_model
from .._io._replay_recipe_adapter import loaded_replay_recipe_status
from .._runnable import refuse_poisoned_trace
from .._utils.rng import restores_rng_streams
from ..backends.registry import get_backend_spec
from ..backends.torch._trace_backward import _refuse_stats_capture
from . import _degenerate_forward
from ._invariants import check_metadata_invariants as _check_metadata_invariants_impl
from ._status import ValidationStatus

if TYPE_CHECKING:
    from .._data_classes.trace import Trace

__all__ = [
    "check_metadata_invariants",
    "save_new_outs",
    "validate_forward_pass",
    "validation_replay_status",
]

_JAX_VALIDATION_REPLAY_BACKEND = "jax"
_MLX_VALIDATION_REPLAY_BACKEND = "mlx"
_TINYGRAD_VALIDATION_REPLAY_BACKEND = "tinygrad"


def _loaded_non_torch_validation_replay_unavailable(trace: Any) -> bool:
    """Return whether loaded non-torch replay validation cannot run.

    Parameters
    ----------
    trace:
        Trace-like object being checked.

    Returns
    -------
    bool
        True for loaded non-torch traces whose backend runtime replay
        capture lists were stripped by save and not rebuilt from replay recipes.
    """
    try:
        loaded_from_bundle = bool(trace._loaded_from_bundle)
    except AttributeError:
        loaded_from_bundle = False
    if not loaded_from_bundle:
        return False
    backend = str(trace._provenance_backend)
    if backend == _MLX_VALIDATION_REPLAY_BACKEND:
        return not bool(getattr(trace, "_mlx_op_captures", ()))
    if backend == _JAX_VALIDATION_REPLAY_BACKEND:
        return not bool(getattr(trace, "jax_equation_captures", ()))
    if backend == _TINYGRAD_VALIDATION_REPLAY_BACKEND:
        return not bool(getattr(trace, "tinygrad_uop_captures", ()))
    return False


def save_new_outs(  # noqa: PLR0913  (the thin Trace._save_new_outs shell keeps this documented call shape; the fold removes the duplicate site)
    trace: Trace,
    model: torch.nn.Module,
    inputs: torch.Tensor | list[Any],
    input_kwargs: dict[Any, Any] | None = None,
    save: str | list[str] = "all",
    save_grads: str | list[str] | None = "all",
    seed: int | None = None,
    backward_ready: bool | None = None,
    _run_until_plan: Any | None = None,
    _staged_plan: Any | None = None,
) -> None:
    """Body of ``Trace._save_new_outs``; the method carries the documented contract."""
    # N3/N5: a live refresh re-drives the FULL native forward against the
    # recorded graph, which a halted/failed/unproven capture cannot honor.
    require_capture_capability(trace, "live_replay")
    require_structure_only_capability(trace, "live_replay")
    _refuse_stats_capture(trace, "live_replay")
    model = unwrap_compiled_model(model)

    return _save_new_outs_impl(
        trace,
        model=model,
        inputs=inputs,
        input_kwargs=input_kwargs,
        save=save,
        save_grads=save_grads,
        seed=seed,
        backward_ready=backward_ready,
        _run_until_plan=_run_until_plan,
        _staged_plan=_staged_plan,
    )


@restores_rng_streams
def validate_forward_pass(
    trace: Trace,
    ground_truth_output_tensors: list[torch.Tensor] | torch.Tensor,
    verbose: bool = False,
    validate_metadata: bool = True,
) -> bool | ValidationStatus:
    """Body of ``Trace.validate_forward_pass``; the method carries the documented contract."""
    if isinstance(ground_truth_output_tensors, torch.Tensor):
        ground_truth_output_tensors = [ground_truth_output_tensors]

    refuse_poisoned_trace(trace, "validation")
    # N2: refusing ENTRY for failed/unproven captures is not a check
    # exemption -- the tripwire bodies stay byte-untouched, the halted
    # exemption neither widens nor narrows, and legacy UNATTESTED
    # artifacts deliberately keep entry OPEN (the tripwire stays armed).
    require_capture_capability(trace, "validation_entry")
    require_structure_only_capability(trace, "validation_entry")
    _refuse_stats_capture(trace, "validation_entry")
    status = trace.validation_replay_status
    try:
        loaded_from_bundle = bool(trace._loaded_from_bundle)
    except AttributeError:
        loaded_from_bundle = False
    if loaded_from_bundle and not status.is_available:
        trace._validation_replay_status = status
        return status
    spec = get_backend_spec(trace._provenance_backend)
    validation_result = spec.validate_trace(
        trace,
        ground_truth_output_tensors=ground_truth_output_tensors,
        verbose=verbose,
        validate_metadata=validate_metadata,
    )
    if spec.name == "torch":  # noqa: SIM102  (verbatim check body; the deferred import that sat here was hoisted)
        if isinstance(validation_result, ValidationStatus):
            trace._validation_replay_status = validation_result
            if validation_result.state in {"passed", "failed"}:
                # A pass on entirely non-finite final outputs proves nothing:
                # fail it as a degenerate forward (tightening only).
                if (
                    validation_result.passed
                    and _degenerate_forward.fail_if_final_outputs_non_finite(
                        trace, ground_truth_output_tensors
                    )
                ):
                    return False
                return validation_result.passed
    return validation_result


def validation_replay_status(trace: Trace) -> ValidationStatus:
    """Body of ``Trace.validation_replay_status``; the method carries the documented contract."""
    try:
        cached_status = trace._validation_replay_status
    except AttributeError:
        cached_status = None
    if isinstance(cached_status, ValidationStatus):
        return cached_status
    backend = str(trace._provenance_backend)
    # A loaded trace saved with replay recipes binds them here (or names why it cannot).
    recipe_status = loaded_replay_recipe_status(trace)
    if recipe_status is not None:
        return recipe_status
    if _loaded_non_torch_validation_replay_unavailable(trace):
        return ValidationStatus.unavailable_loaded_runtime_stripped(
            backend=backend,
            payload_load_status=getattr(trace, "payload_load_status", None),
        )
    spec = get_backend_spec(backend)
    if not spec.capabilities.has_validation_replay:
        return ValidationStatus.unavailable_unsupported(backend=backend)
    return ValidationStatus.available_live(backend=backend)


def check_metadata_invariants(trace: Trace) -> bool:
    """Body of ``Trace.check_metadata_invariants``; the method carries the documented contract."""
    # N2: refusing ENTRY for failed/unproven captures is not a check
    # exemption -- the tripwire bodies stay byte-untouched, and legacy
    # UNATTESTED artifacts deliberately keep entry OPEN.
    require_capture_capability(trace, "validation_entry")
    require_structure_only_capability(trace, "validation_entry")
    _refuse_stats_capture(trace, "validation_entry")
    return _check_metadata_invariants_impl(trace)
