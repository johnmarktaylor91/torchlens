"""Full-forward rerun engine for TorchLens interventions."""

from __future__ import annotations

import time
import warnings
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from .._deprecations import MISSING, MissingType
from .._input_coerce import _coerce_input_args
from .._trace_state import TraceState
from ..options import ReplayOptions, merge_replay_options
from .errors import (
    AppendBatchDependenceError,
    AppendMismatchError,
    AppendStreamingNotSupportedError,
    BatchNormTrainModeWarning,
    ControlFlowDivergenceError,
    ControlFlowDivergenceWarning,
    DirectActivationWriteWarning,
)
from .hooks import normalize_hooks_from_spec
from .runtime import active_intervention_context

if TYPE_CHECKING:
    from ..data_classes.trace import Trace
    from .hooks import NormalizedHookEntry


def rerun(
    log: "Trace",
    model: nn.Module,
    x: Any = None,
    *,
    append: bool | MissingType = MISSING,
    strict: bool | MissingType = MISSING,
    replay: ReplayOptions | None = None,
    output_transform: Any | None = None,
) -> "Trace":
    """Full-forward rerun with the active intervention spec from ``log``.

    Re-executes ``model`` through TorchLens decorated wrappers with the current
    intervention spec installed in runtime context. A fresh ``Trace`` is
    built off to the side, validated, then atomically swapped into ``log``.
    Concurrent reads during rerun are unsupported; no lock is taken.

    Parameters
    ----------
    log:
        Trace to update in place after the fresh run validates.
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
    output_transform:
        Optional callable applied to the fresh model output for raw-output
        metadata storage.

    Returns
    -------
    Trace
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
            output_transform=output_transform,
        )

    divergence_count = _validate_rerun_result(new_log, log, strict=replay_options.strict)
    log.replace_state_from(new_log)
    log.is_appended = False
    log._append_sequence_id = 0
    log.append_history = []

    history_record = _build_ledger_record(
        log,
        started_at=started_at,
        old_hash=old_hash,
        new_hash=getattr(new_log, "graph_shape_hash", None),
        hook_plan=hook_plan,
        strict=replay_options.strict,
        divergence_count=divergence_count,
    )
    log.state = TraceState.RERUN_PROPAGATED
    log.last_run = {
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
    log._out_recipe_revision = getattr(log, "_spec_revision", 0)
    return log


def _append_rerun(
    log: "Trace",
    model: nn.Module,
    x: Any,
    *,
    strict: bool,
) -> "Trace":
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
    Trace
        The same log after compatible tensors have been concatenated.
    """

    del strict
    if _is_streaming_append_active(log):
        raise AppendStreamingNotSupportedError(_streaming_append_error_message(log))
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
            output_transform=getattr(log, "_output_transform", None),
        )

    _validate_append_candidate(log, new_log, hook_plan=hook_plan)
    log.append_state_from(new_log)
    log.is_appended = True
    log._append_sequence_id = int(getattr(log, "_append_sequence_id", 0)) + 1
    log.state = TraceState.APPENDED
    log._has_direct_writes = False
    log._out_recipe_revision = getattr(log, "_spec_revision", 0)

    duration_s = time.monotonic() - started_at
    chunk_size = _batch_size_from_input(x)
    total_batch_size = _first_saved_batch_size(log)
    log.last_run = {
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
    log.append_history.append(dict(log.last_run))
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


def _is_streaming_append_active(log: "Trace") -> bool:
    """Return whether append would need to update active streaming state.

    Parameters
    ----------
    log:
        Trace inspected before append capture.

    Returns
    -------
    bool
        True when a bundle writer or out sink is still attached.
    """

    return (
        getattr(log, "_out_writer", None) is not None or getattr(log, "_out_sink", None) is not None
    )


def _streaming_append_error_message(log: "Trace") -> str:
    """Build a descriptive streaming append rejection message.

    Parameters
    ----------
    log:
        Trace whose active streaming handles block append.

    Returns
    -------
    str
        User-facing exception message.
    """

    details = []
    if getattr(log, "_out_writer", None) is not None:
        details.append("bundle_path streaming")
    if getattr(log, "_out_sink", None) is not None:
        details.append("out_callback streaming")
    details_text = ", ".join(details) if details else "streaming"
    return (
        f"Trace {getattr(log, 'model_class_name', None)!r} has active {details_text}; "
        "rerun(append=True) cannot update streamed activation storage. Save and reload "
        "the trace before appending, or disable streaming for this capture."
    )


def _preflight_append(log: "Trace", model: nn.Module) -> None:
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
    log: "Trace",
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
                f"helper {helper_name!r} may change out shape and is not compatible_with_append"
            )


def _warn_unknown_append_helper_once(log: "Trace", helper_name: str) -> None:
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
    old_log: "Trace",
    new_log: "Trace",
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
        Active hook entries used to decide grad support.
    """

    old_hash = getattr(old_log, "graph_shape_hash", None)
    new_hash = getattr(new_log, "graph_shape_hash", None)
    if old_hash != new_hash:
        raise AppendMismatchError("graph shape changed")

    old_labels = tuple(layer._layer_label_raw for layer in old_log.layer_list)
    new_labels = tuple(layer._layer_label_raw for layer in new_log.layer_list)
    if old_labels != new_labels:
        raise AppendMismatchError("topology or site labels changed")

    old_by_raw = {layer._layer_label_raw: layer for layer in old_log.layer_list}
    new_by_raw = {layer._layer_label_raw: layer for layer in new_log.layer_list}
    grads_supported = _hook_plan_supports_append_grads(hook_plan)
    for raw_label in old_labels:
        old_layer = old_by_raw[raw_label]
        new_layer = new_by_raw[raw_label]
        _validate_append_tensor_pair(old_layer, new_layer, "out")
        _validate_append_tensor_pair(old_layer, new_layer, "transformed_out")
        _validate_append_grad_pair(old_layer, new_layer, grads_supported=grads_supported)


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
            f"{old_layer._layer_label_raw} {field_name} presence changed across chunks"
        )
    if old_value.ndim == 0 or new_value.ndim == 0:
        raise AppendMismatchError(
            f"{old_layer._layer_label_raw} {field_name} has no batch dimension"
        )
    if tuple(old_value.shape[1:]) != tuple(new_value.shape[1:]):
        raise AppendMismatchError(
            f"{old_layer._layer_label_raw} {field_name} shape changed outside batch "
            f"(old={tuple(old_value.shape)}, new={tuple(new_value.shape)})"
        )
    if old_value.dtype != new_value.dtype:
        raise AppendMismatchError(
            f"{old_layer._layer_label_raw} {field_name} dtype changed "
            f"(old={old_value.dtype}, new={new_value.dtype})"
        )
    if old_value.device != new_value.device:
        raise AppendMismatchError(
            f"{old_layer._layer_label_raw} {field_name} device changed "
            f"(old={old_value.device}, new={new_value.device})"
        )


def _validate_append_grad_pair(
    old_layer: Any,
    new_layer: Any,
    *,
    grads_supported: bool,
) -> None:
    """Validate grad fields for append.

    Parameters
    ----------
    old_layer:
        Existing pass.
    new_layer:
        New chunk pass.
    grads_supported:
        Whether every active helper opted into grad concatenation.
    """

    grad_fields = ("grad", "transformed_grad")
    has_any_grad = any(
        isinstance(getattr(layer, field_name, None), torch.Tensor)
        for layer in (old_layer, new_layer)
        for field_name in grad_fields
    )
    if not has_any_grad:
        return
    if not grads_supported:
        raise AppendBatchDependenceError(
            "append grad concatenation requires a batch-independent helper with "
            "supports_append_grads=True; use such a helper, disable backward_ready grad "
            "append, or replay chunks manually"
        )
    for field_name in grad_fields:
        _validate_append_tensor_pair(old_layer, new_layer, field_name)


def _hook_plan_supports_append_grads(hook_plan: list["NormalizedHookEntry"]) -> bool:
    """Return whether all active helpers opted into grad append.

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
        and bool(getattr(entry.helper_spec, "supports_append_grads", False))
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


def _first_saved_batch_size(log: "Trace") -> int | None:
    """Return the first saved out's leading dimension.

    Parameters
    ----------
    log:
        Model log to inspect.

    Returns
    -------
    int | None
        Leading dimension from the first tensor out, if any.
    """

    for layer in log.layer_list:
        out = getattr(layer, "out", None)
        if isinstance(out, torch.Tensor) and out.ndim > 0:
            return int(out.shape[0])
    return None


def _warn_if_direct_writes_will_be_overlaid(log: "Trace") -> None:
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
        "recipe and may overlay direct Op out writes.",
        DirectActivationWriteWarning,
        stacklevel=3,
    )
    setattr(log, "_warned_direct_write_propagation", True)


def _preflight(log: "Trace", model: nn.Module, x: Any) -> None:
    """Validate rerun preconditions before any fresh capture starts.

    Parameters
    ----------
    log:
        Trace that will be updated.
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
    log: "Trace",
    model: nn.Module,
    x: Any,
    *,
    intervention_spec: Any | None,
    hook_plan: list["NormalizedHookEntry"],
    output_transform: Any | None,
) -> "Trace":
    """Build a fresh rerun ``Trace`` with active hooks installed.

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
    output_transform:
        Optional callable applied to the fresh model output for raw-output
        metadata storage.

    Returns
    -------
    Trace
        Fresh log built off to the side.
    """

    from ..user_funcs import (  # type: ignore[attr-defined]
        _run_model_and_save_specified_outs,
        _unwrap_data_parallel,
        check_model_and_input_variants,
    )

    model = _unwrap_data_parallel(model)
    x = _coerce_input_args(model, x)
    check_model_and_input_variants(model, x, {})
    return _run_model_and_save_specified_outs(
        model=model,
        input_args=x,
        input_kwargs={},
        layers_to_save="all",
        output_device=getattr(log, "output_device", "same"),
        activation_transform=getattr(log, "activation_transform", None),
        grad_transform=getattr(log, "grad_transform", None),
        save_raw_activations=getattr(log, "save_raw_activations", True),
        save_raw_gradients=getattr(log, "save_raw_gradients", True),
        mark_layer_depths=getattr(log, "mark_layer_depths", False),
        detach_saved_activations=getattr(log, "detach_saved_activations", False),
        save_arg_values=getattr(log, "save_arg_values", False),
        save_gradients=getattr(log, "save_gradients", False),
        gradients_to_save=getattr(log, "gradients_to_save", "all"),
        random_seed=getattr(log, "random_seed", None),
        num_context_lines=getattr(log, "num_context_lines", 7),
        optimizer=getattr(log, "_optimizer", None),
        save_code_context=getattr(log, "save_code_context", False),
        save_rng_states=getattr(log, "save_rng_states", False),
        recurrence_detection=getattr(log, "recurrence_detection", True),
        intervention_ready=True,
        intervention_spec=intervention_spec,
        normalized_hook_plan=hook_plan,
        verbose=getattr(log, "verbose", False),
        backward_ready=getattr(log, "backward_ready", False),
        output_transform=output_transform,
        save_raw_output=getattr(log, "save_raw_output", "small"),
    )


def _validate_rerun_result(new_log: "Trace", old_log: "Trace", *, strict: bool) -> int:
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


def _build_ledger_record(
    log: "Trace",
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
        Trace after state replacement.
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
