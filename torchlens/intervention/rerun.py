"""Full-forward rerun engine for TorchLens interventions."""

from __future__ import annotations

import time
import warnings
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from .._deprecations import MISSING, MissingType
from .._run_state import RunState
from ..options import ReplayOptions, merge_replay_options
from .errors import (
    AppendBatchDependenceError,
    AppendMismatchError,
    BatchNormTrainModeWarning,
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
    append: bool | MissingType = MISSING,
    strict: bool | MissingType = MISSING,
    replay: ReplayOptions | None = None,
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
        If true, capture ``x`` as a compatible chunk and append saved tensors
        along batch dimension 0 instead of replacing the run state.
    strict:
        If true, graph-shape divergence raises ``ControlFlowDivergenceError``.
        If false, divergence emits ``ControlFlowDivergenceWarning`` and the
        atomic swap proceeds.

    Returns
    -------
    ModelLog
        The same ``log`` object after atomic run-state replacement.
    """

    replay_options = merge_replay_options(replay=replay, append=append, strict=strict)
    if replay_options.append:
        return _append_rerun(log, model, x, strict=replay_options.strict)
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

    divergence_count = _validate_rerun_result(new_log, log, strict=replay_options.strict)
    log.replace_run_state_from(new_log)

    history_record = _build_operation_history_record(
        log,
        started_at=started_at,
        old_hash=old_hash,
        new_hash=getattr(new_log, "graph_shape_hash", None),
        hook_plan=hook_plan,
        strict=replay_options.strict,
        divergence_count=divergence_count,
    )
    log.run_state = RunState.RERUN_PROPAGATED
    log.last_run_ctx = {
        "engine": "rerun",
        "timestamp": time.monotonic(),
        "started_at": started_at,
        "duration_s": time.monotonic() - started_at,
        "spec_revision": getattr(log, "_spec_revision", 0),
        "strict": replay_options.strict,
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


def _append_rerun(
    log: "ModelLog",
    model: nn.Module,
    x: Any,
    *,
    strict: bool,
) -> "ModelLog":
    """Append a compatible fresh rerun chunk into ``log``.

    Parameters
    ----------
    log:
        Existing accumulated log.
    model:
        Model to execute for the new chunk.
    x:
        New chunk input.
    strict:
        Accepted for API parity with rerun; append always rejects divergence.

    Returns
    -------
    ModelLog
        The same log after compatible tensors have been concatenated.
    """

    del strict
    _preflight(log, model, x)
    _preflight_append(log, model)
    _warn_if_direct_writes_will_be_overlaid(log)
    _warn_if_batch_sensitive_train_modules(model)

    spec = getattr(log, "_intervention_spec", None)
    hook_plan = normalize_hooks_from_spec(spec)
    _validate_append_hook_plan(log, hook_plan)
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

    _validate_append_candidate(log, new_log, hook_plan=hook_plan)
    log.append_run_state_from(new_log)
    log.is_appended = True
    log._append_sequence_id = int(getattr(log, "_append_sequence_id", 0)) + 1
    log.run_state = RunState.APPENDED
    log._has_direct_writes = False
    log._activation_recipe_revision = getattr(log, "_spec_revision", 0)

    duration_s = time.monotonic() - started_at
    chunk_size = _batch_size_from_input(x)
    total_batch_size = _first_saved_batch_size(log)
    log.last_run_ctx = {
        "engine": "append",
        "timestamp": time.monotonic(),
        "started_at": started_at,
        "duration_s": duration_s,
        "spec_revision": getattr(log, "_spec_revision", 0),
        "append": True,
        "strict": False,
        "hooks": len(hook_plan),
        "chunk_size": chunk_size,
        "total_batch_size": total_batch_size,
        "append_sequence_id": log._append_sequence_id,
        "old_graph_shape_hash": old_hash,
        "new_graph_shape_hash": getattr(new_log, "graph_shape_hash", None),
    }
    log._record_operation(
        "append",
        engine="append",
        started_at=started_at,
        duration_s=duration_s,
        hook_count=len(hook_plan),
        chunk_size=chunk_size,
        total_batch_size=total_batch_size,
        append_sequence_id=log._append_sequence_id,
        old_graph_shape_hash=old_hash,
        new_graph_shape_hash=getattr(new_log, "graph_shape_hash", None),
    )
    return log


def _preflight_append(log: "ModelLog", model: nn.Module) -> None:
    """Validate append preconditions that do not require a fresh capture.

    Parameters
    ----------
    log:
        Existing log.
    model:
        Candidate model for the append chunk.
    """

    if not log._recipe_is_clean():
        raise AppendMismatchError("recipe is stale; replay or rerun first")
    log._validate_supplied_model_matches_capture(model)


def _warn_if_batch_sensitive_train_modules(model: nn.Module) -> None:
    """Warn when train-mode batch-sensitive modules may make chunks differ.

    Parameters
    ----------
    model:
        Model inspected before append capture.
    """

    if not model.training:
        return
    has_batch_norm = any(
        isinstance(module, nn.modules.batchnorm._BatchNorm) for module in model.modules()
    )
    has_dropout = any(
        isinstance(module, nn.modules.dropout._DropoutNd) for module in model.modules()
    )
    if has_batch_norm:
        warnings.warn(
            "BatchNorm train mode changes statistics per chunk; appended results may differ "
            "from full-batch rerun.",
            BatchNormTrainModeWarning,
            stacklevel=3,
        )
    if has_dropout:
        warnings.warn(
            "Dropout train mode samples masks per chunk; appended results may differ from "
            "full-batch rerun.",
            BatchNormTrainModeWarning,
            stacklevel=3,
        )


def _validate_append_hook_plan(
    log: "ModelLog",
    hook_plan: list["NormalizedHookEntry"],
) -> None:
    """Reject append when active helpers are not explicitly batch-independent.

    Parameters
    ----------
    log:
        Log being appended into; used for one-time warning state.
    hook_plan:
        Normalized active hook entries.
    """

    for entry in hook_plan:
        helper = entry.helper_spec
        helper_name = getattr(helper, "name", None) or "user_hook"
        if helper is None:
            _warn_unknown_append_helper_once(log, helper_name)
            raise AppendBatchDependenceError(
                f"helper {helper_name!r} does not declare batch_independent=True"
            )
        if not hasattr(helper, "batch_independent"):
            _warn_unknown_append_helper_once(log, helper_name)
            raise AppendBatchDependenceError(
                f"helper {helper_name!r} does not declare batch_independent=True"
            )
        if not bool(getattr(helper, "batch_independent", False)):
            raise AppendBatchDependenceError(
                f"helper {helper_name!r} is not batch-independent; append is unsafe"
            )
        force_shape_change = bool(entry.metadata.get("force_shape_change", False)) or bool(
            dict(helper.kwargs).get("force_shape_change", False)
        )
        if force_shape_change and not bool(getattr(helper, "compatible_with_append", False)):
            raise AppendMismatchError(
                f"helper {helper_name!r} may change activation shape and is not "
                "compatible_with_append"
            )


def _warn_unknown_append_helper_once(log: "ModelLog", helper_name: str) -> None:
    """Emit a one-time warning for helpers without append-safety metadata.

    Parameters
    ----------
    log:
        Log receiving append.
    helper_name:
        Display name for the helper or callable.
    """

    warned: set[str] = getattr(log, "_warned_unknown_append_helper", set())
    if helper_name in warned:
        return
    warnings.warn(
        f"helper {helper_name!r} has no batch_independent flag; add it to enable append",
        UserWarning,
        stacklevel=3,
    )
    warned = set(warned)
    warned.add(helper_name)
    setattr(log, "_warned_unknown_append_helper", warned)


def _validate_append_candidate(
    old_log: "ModelLog",
    new_log: "ModelLog",
    *,
    hook_plan: list["NormalizedHookEntry"],
) -> None:
    """Validate a freshly captured append candidate against an existing log.

    Parameters
    ----------
    old_log:
        Existing accumulated log.
    new_log:
        Fresh chunk log.
    hook_plan:
        Active hook entries used to decide gradient support.
    """

    old_hash = getattr(old_log, "graph_shape_hash", None)
    new_hash = getattr(new_log, "graph_shape_hash", None)
    if old_hash != new_hash:
        raise AppendMismatchError("graph shape changed")

    old_labels = tuple(layer.layer_label_raw for layer in old_log.layer_list)
    new_labels = tuple(layer.layer_label_raw for layer in new_log.layer_list)
    if old_labels != new_labels:
        raise AppendMismatchError("topology or site labels changed")

    old_by_raw = {layer.layer_label_raw: layer for layer in old_log.layer_list}
    new_by_raw = {layer.layer_label_raw: layer for layer in new_log.layer_list}
    gradients_supported = _hook_plan_supports_append_gradients(hook_plan)
    for raw_label in old_labels:
        old_layer = old_by_raw[raw_label]
        new_layer = new_by_raw[raw_label]
        _validate_append_tensor_pair(old_layer, new_layer, "activation")
        _validate_append_tensor_pair(old_layer, new_layer, "transformed_activation")
        _validate_append_gradient_pair(
            old_layer, new_layer, gradients_supported=gradients_supported
        )


def _validate_append_tensor_pair(old_layer: Any, new_layer: Any, field_name: str) -> None:
    """Validate one tensor field for append concatenation.

    Parameters
    ----------
    old_layer:
        Existing pass.
    new_layer:
        New chunk pass.
    field_name:
        Tensor field to compare.
    """

    old_value = getattr(old_layer, field_name, None)
    new_value = getattr(new_layer, field_name, None)
    if old_value is None and new_value is None:
        return
    if not isinstance(old_value, torch.Tensor) or not isinstance(new_value, torch.Tensor):
        raise AppendMismatchError(
            f"{old_layer.layer_label_raw} {field_name} presence changed across chunks"
        )
    if old_value.ndim == 0 or new_value.ndim == 0:
        raise AppendMismatchError(
            f"{old_layer.layer_label_raw} {field_name} has no batch dimension"
        )
    if tuple(old_value.shape[1:]) != tuple(new_value.shape[1:]):
        raise AppendMismatchError(
            f"{old_layer.layer_label_raw} {field_name} shape changed outside batch "
            f"(old={tuple(old_value.shape)}, new={tuple(new_value.shape)})"
        )
    if old_value.dtype != new_value.dtype:
        raise AppendMismatchError(
            f"{old_layer.layer_label_raw} {field_name} dtype changed "
            f"(old={old_value.dtype}, new={new_value.dtype})"
        )
    if old_value.device != new_value.device:
        raise AppendMismatchError(
            f"{old_layer.layer_label_raw} {field_name} device changed "
            f"(old={old_value.device}, new={new_value.device})"
        )


def _validate_append_gradient_pair(
    old_layer: Any,
    new_layer: Any,
    *,
    gradients_supported: bool,
) -> None:
    """Validate gradient fields for append.

    Parameters
    ----------
    old_layer:
        Existing pass.
    new_layer:
        New chunk pass.
    gradients_supported:
        Whether every active helper opted into gradient concatenation.
    """

    gradient_fields = ("gradient", "transformed_gradient")
    has_any_gradient = any(
        isinstance(getattr(layer, field_name, None), torch.Tensor)
        for layer in (old_layer, new_layer)
        for field_name in gradient_fields
    )
    if not has_any_gradient:
        return
    if not gradients_supported:
        raise AppendBatchDependenceError(
            "append gradient concatenation requires helper-specific opt-in"
        )
    for field_name in gradient_fields:
        _validate_append_tensor_pair(old_layer, new_layer, field_name)


def _hook_plan_supports_append_gradients(hook_plan: list["NormalizedHookEntry"]) -> bool:
    """Return whether all active helpers opted into gradient append.

    Parameters
    ----------
    hook_plan:
        Active hook entries.

    Returns
    -------
    bool
        True only when at least one helper exists and all helpers opt in.
    """

    if not hook_plan:
        return False
    return all(
        entry.helper_spec is not None
        and bool(getattr(entry.helper_spec, "supports_append_gradients", False))
        for entry in hook_plan
    )


def _batch_size_from_input(x: Any) -> int | None:
    """Return the first tensor input's leading dimension when available.

    Parameters
    ----------
    x:
        User-supplied append input.

    Returns
    -------
    int | None
        Leading dimension, or ``None`` when no tensor with a batch axis exists.
    """

    if isinstance(x, torch.Tensor):
        return int(x.shape[0]) if x.ndim > 0 else None
    if isinstance(x, dict):
        for key in sorted(x.keys(), key=repr):
            value = _batch_size_from_input(x[key])
            if value is not None:
                return value
    if isinstance(x, (list, tuple)):
        for item in x:
            value = _batch_size_from_input(item)
            if value is not None:
                return value
    return None


def _first_saved_batch_size(log: "ModelLog") -> int | None:
    """Return the first saved activation's leading dimension.

    Parameters
    ----------
    log:
        Model log to inspect.

    Returns
    -------
    int | None
        Leading dimension from the first tensor activation, if any.
    """

    for layer in log.layer_list:
        activation = getattr(layer, "activation", None)
        if isinstance(activation, torch.Tensor) and activation.ndim > 0:
            return int(activation.shape[0])
    return None


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
