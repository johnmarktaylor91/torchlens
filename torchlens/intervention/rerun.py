"""Full-forward rerun engine for TorchLens interventions."""

from __future__ import annotations

import time
import warnings
from typing import TYPE_CHECKING, Any

from torch import nn

from .._run_state import RunState
from .errors import (
    ControlFlowDivergenceError,
    ControlFlowDivergenceWarning,
    DirectActivationWriteWarning,
)
from .hooks import normalize_hooks_from_spec
from .runtime import active_intervention_context

if TYPE_CHECKING:
    from ..data_classes.model_log import ModelLog
    from .hooks import NormalizedHookEntry


def rerun(
    log: "ModelLog",
    model: nn.Module,
    x: Any = None,
    *,
    append: bool = False,
    strict: bool = False,
) -> "ModelLog":
    """Full-forward rerun with the active intervention spec from ``log``.

    Re-executes ``model`` through TorchLens decorated wrappers with the current
    intervention spec installed in runtime context. A fresh ``ModelLog`` is
    built off to the side, validated, then atomically swapped into ``log``.
    Concurrent reads during rerun are unsupported; no lock is taken.

    Parameters
    ----------
    log:
        ModelLog to update in place after the fresh run validates.
    model:
        Model to execute through the rerun engine.
    x:
        Forward input. Phase 7 does not retain strong references to original
        inputs, so ``None`` raises and callers must pass the input explicitly.
    append:
        Reserved for Phase 12 append reruns. ``True`` raises in Phase 7.
    strict:
        If true, graph-shape divergence raises ``ControlFlowDivergenceError``.
        If false, divergence emits ``ControlFlowDivergenceWarning`` and the
        atomic swap proceeds.

    Returns
    -------
    ModelLog
        The same ``log`` object after atomic run-state replacement.
    """

    if append:
        raise NotImplementedError("rerun(..., append=True) is deferred until Phase 12.")
    _preflight(log, model, x)
    _warn_if_direct_writes_will_be_overlaid(log)

    spec = getattr(log, "_intervention_spec", None)
    hook_plan = normalize_hooks_from_spec(spec)
    started_at = time.monotonic()
    old_hash = getattr(log, "graph_shape_hash", None)

    with active_intervention_context(intervention_spec=spec, hook_plan=hook_plan):
        new_log = _capture_with_active_spec(
            log,
            model,
            x,
            intervention_spec=spec,
            hook_plan=hook_plan,
        )

    divergence_count = _validate_rerun_result(new_log, log, strict=strict)
    log.replace_run_state_from(new_log)

    history_record = _build_operation_history_record(
        log,
        started_at=started_at,
        old_hash=old_hash,
        new_hash=getattr(new_log, "graph_shape_hash", None),
        hook_plan=hook_plan,
        strict=strict,
        divergence_count=divergence_count,
    )
    log.run_state = RunState.RERUN_PROPAGATED
    log.last_run_ctx = {
        "engine": "rerun",
        "timestamp": time.monotonic(),
        "started_at": started_at,
        "duration_s": time.monotonic() - started_at,
        "spec_revision": getattr(log, "_spec_revision", 0),
        "strict": strict,
        "append": False,
        "hooks": len(hook_plan),
        "divergence_count": divergence_count,
        "old_graph_shape_hash": old_hash,
        "new_graph_shape_hash": getattr(log, "graph_shape_hash", None),
    }
    log._record_operation(**history_record)
    log._has_direct_writes = False
    log._activation_recipe_revision = getattr(log, "_spec_revision", 0)
    return log


def _warn_if_direct_writes_will_be_overlaid(log: "ModelLog") -> None:
    """Warn once that rerun propagation overlays direct writes.

    Parameters
    ----------
    log:
        Model log about to be propagated.
    """

    if not getattr(log, "_has_direct_writes", False):
        return
    if getattr(log, "_warned_direct_write_propagation", False):
        return
    warnings.warn(
        "DirectActivationWriteWarning: replay/rerun propagation uses the intervention "
        "recipe and may overlay direct LayerPassLog activation writes.",
        DirectActivationWriteWarning,
        stacklevel=3,
    )
    setattr(log, "_warned_direct_write_propagation", True)


def _preflight(log: "ModelLog", model: nn.Module, x: Any) -> None:
    """Validate rerun preconditions before any fresh capture starts.

    Parameters
    ----------
    log:
        ModelLog that will be updated.
    model:
        Model to execute.
    x:
        Forward input supplied by the caller.

    Returns
    -------
    None
        Raises if a precondition fails.
    """

    if x is None:
        raise ValueError(
            "rerun(..., x=None) cannot recover the original input in Phase 7. "
            "Pass the forward input explicitly as log.rerun(model, x)."
        )
    from ..user_funcs import _reject_opaque_wrappers

    _reject_opaque_wrappers(model)


def _capture_with_active_spec(
    log: "ModelLog",
    model: nn.Module,
    x: Any,
    *,
    intervention_spec: Any | None,
    hook_plan: list["NormalizedHookEntry"],
) -> "ModelLog":
    """Build a fresh rerun ``ModelLog`` with active hooks installed.

    Parameters
    ----------
    log:
        Existing log whose capture settings should be mirrored.
    model:
        Model to execute.
    x:
        Forward input supplied by the caller.
    intervention_spec:
        Active intervention spec exposed through runtime state.
    hook_plan:
        Normalized live hook plan derived from the spec.

    Returns
    -------
    ModelLog
        Fresh log built off to the side.
    """

    from ..user_funcs import (
        _run_model_and_save_specified_activations,
        _unwrap_data_parallel,
        check_model_and_input_variants,
    )

    model = _unwrap_data_parallel(model)
    check_model_and_input_variants(model, x, {})
    return _run_model_and_save_specified_activations(
        model=model,
        input_args=x,
        input_kwargs={},
        layers_to_save="all",
        keep_unsaved_layers=True,
        output_device=getattr(log, "output_device", "same"),
        activation_postfunc=getattr(log, "activation_postfunc", None),
        gradient_postfunc=getattr(log, "gradient_postfunc", None),
        save_raw_activation=getattr(log, "save_raw_activation", True),
        save_raw_gradient=getattr(log, "save_raw_gradient", True),
        mark_input_output_distances=getattr(log, "mark_input_output_distances", False),
        detach_saved_tensors=getattr(log, "detach_saved_tensors", False),
        save_function_args=getattr(log, "save_function_args", False),
        save_gradients=getattr(log, "save_gradients", False),
        gradients_to_save=getattr(log, "gradients_to_save", "all"),
        random_seed=getattr(log, "random_seed_used", None),
        num_context_lines=getattr(log, "num_context_lines", 7),
        optimizer=getattr(log, "_optimizer", None),
        save_source_context=getattr(log, "save_source_context", False),
        save_rng_states=getattr(log, "save_rng_states", False),
        detect_loops=getattr(log, "detect_loops", True),
        intervention_ready=True,
        intervention_spec=intervention_spec,
        normalized_hook_plan=hook_plan,
        verbose=getattr(log, "verbose", False),
        train_mode=getattr(log, "train_mode", False),
    )


def _validate_rerun_result(new_log: "ModelLog", old_log: "ModelLog", *, strict: bool) -> int:
    """Validate a fresh rerun log before atomic state replacement.

    Parameters
    ----------
    new_log:
        Freshly captured candidate log.
    old_log:
        Existing log being replaced.
    strict:
        Whether divergence should raise.

    Returns
    -------
    int
        Number of graph-shape divergence events detected by the Phase 7 MVP.
    """

    old_hash = getattr(old_log, "graph_shape_hash", None)
    new_hash = getattr(new_log, "graph_shape_hash", None)
    if old_hash == new_hash:
        return 0

    message = (
        "rerun graph_shape_hash diverged from the captured graph "
        f"(old={old_hash!r}, new={new_hash!r}). Phase 7 uses graph-shape hash "
        "comparison as the conservative control-flow divergence detector."
    )
    if strict:
        raise ControlFlowDivergenceError(message)
    warnings.warn(message, ControlFlowDivergenceWarning, stacklevel=3)
    return 1


def _build_operation_history_record(
    log: "ModelLog",
    *,
    started_at: float,
    old_hash: str | None,
    new_hash: str | None,
    hook_plan: list["NormalizedHookEntry"],
    strict: bool,
    divergence_count: int,
) -> dict[str, Any]:
    """Create the append-only operation history record for a rerun.

    Parameters
    ----------
    log:
        ModelLog after state replacement.
    started_at:
        Monotonic start time for the rerun.
    old_hash:
        Graph-shape hash before rerun.
    new_hash:
        Candidate graph-shape hash from the fresh capture.
    hook_plan:
        Normalized hook entries active during rerun.
    strict:
        Whether strict divergence handling was requested.
    divergence_count:
        Number of divergence events detected.

    Returns
    -------
    dict[str, Any]
        Operation-history entry.
    """

    return {
        "op": "rerun",
        "engine": "rerun",
        "started_at": started_at,
        "strict": strict,
        "append": False,
        "hook_count": len(hook_plan),
        "divergence_count": divergence_count,
        "old_graph_shape_hash": old_hash,
        "new_graph_shape_hash": new_hash,
    }


__all__ = ["rerun"]
