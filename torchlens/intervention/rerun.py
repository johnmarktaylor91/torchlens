"""Full-forward rerun engine for TorchLens interventions."""

from __future__ import annotations

import time
import warnings
from collections import Counter
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from .._chunking import iter_chunked_inputs, normalize_chunk_paths, plan_chunks
from .._errors import InvalidArgumentError, TorchLensWarning
from .._input_coerce import _coerce_input_args
from .._trace_state import TraceState
from ..options import ReplayOptions, merge_replay_options
from .errors import (
    AppendBatchDependenceError,
    AppendMismatchError,
    AppendStreamingNotSupportedError,
    BatchNormTrainModeWarning,
    ChunkedForwardConfigError,
    ControlFlowDivergenceError,
    ControlFlowDivergenceWarning,
    DirectActivationWriteWarning,
)
from .hooks import normalize_hooks_from_spec
from .runtime import active_intervention_context

if TYPE_CHECKING:
    from ..data_classes.trace import Trace
    from .hooks import NormalizedHookEntry


def run(
    log: Trace,
    model: nn.Module,
    x: Any = None,
    *,
    chunk_paths: Any | None = None,
    replay: ReplayOptions | None = None,
    output_transform: Any | None = None,
) -> Trace:
    """Full-forward run with the active intervention spec from ``log``.

    Re-executes ``model`` through TorchLens decorated wrappers with the current
    intervention spec installed in runtime context. A fresh ``Trace`` is
    built off to the side, validated, then atomically swapped into ``log``.
    Concurrent reads during run are unsupported; no lock is taken.

    Parameters
    ----------
    log:
        Trace to update in place after the fresh run validates.
    model:
        Model to execute through the run engine.
    x:
        Forward input. Rerun does not retain strong references to original
        inputs, so callers must pass the input explicitly.
    chunk_paths:
        Optional explicit tensor leaf paths to split when multiple batched
        tensor leaves are present.
    replay:
        Grouped replay options (``ReplayOptions``: ``append``, ``chunk_size``,
        ``strict``).
    output_transform:
        Optional callable applied to the fresh model output for raw-output
        metadata storage.

    Returns
    -------
    Trace
        The same ``log`` object after atomic run-state replacement.
    """

    replay_options = merge_replay_options(replay=replay)
    if replay_options.chunk_size is not None:
        if replay_options.append:
            raise ChunkedForwardConfigError(
                "rerun(chunk_size=...) cannot be combined with append=True."
            )
        return _chunked_rerun(
            log,
            model,
            x,
            chunk_size=replay_options.chunk_size,
            chunk_paths=chunk_paths,
            strict=replay_options.strict,
            output_transform=output_transform,
        )
    if replay_options.append:
        return _append_rerun(log, model, x, strict=replay_options.strict)
    _preflight(log, model, x)
    model = _unwrap_compiled_model(model)
    _warn_if_direct_writes_will_be_overlaid(log)

    spec = getattr(log, "_intervention_spec", None)
    hook_plan = _assign_unique_plan_ids(normalize_hooks_from_spec(spec))
    started_at = time.monotonic()
    old_hash = getattr(log, "graph_shape_hash", None)
    old_raw_hash = getattr(log, "_raw_event_shape_hash", None)

    with active_intervention_context(intervention_spec=spec, hook_plan=hook_plan):
        new_log = _capture_with_active_spec(
            log,
            model,
            x,
            intervention_spec=spec,
            hook_plan=hook_plan,
            output_transform=output_transform,
        )
    _refuse_shifted_raw_index_resave(log, new_log)
    new_log.facet_registry_snapshot = getattr(log, "facet_registry_snapshot", None)
    hook_fire_count, unfired_hook_ids = _reconcile_rerun_hook_fires(new_log, hook_plan)

    divergence_count = _validate_rerun_result(new_log, log, strict=replay_options.strict)
    fast_refresh = False
    if divergence_count == 0:
        fast_refresh = log._refresh_matching_rerun_state_from(new_log)
    if not fast_refresh:
        log.replace_state_from(new_log)
    log.is_appended = False
    log._append_sequence_id = 0
    log.append_history = []

    history_record = _build_ledger_record(
        log,
        started_at=started_at,
        old_hash=old_hash,
        new_hash=getattr(new_log, "graph_shape_hash", None),
        old_raw_hash=old_raw_hash,
        new_raw_hash=getattr(new_log, "_raw_event_shape_hash", None),
        hook_plan=hook_plan,
        strict=replay_options.strict,
        divergence_count=divergence_count,
        fast_refresh=fast_refresh,
        hook_fire_count=hook_fire_count,
        unfired_hook_count=len(unfired_hook_ids),
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
        "hooks_fired": hook_fire_count,
        "hooks_unfired": len(unfired_hook_ids),
        "divergence_count": divergence_count,
        "fast_refresh": fast_refresh,
        "old_graph_shape_hash": old_hash,
        "new_graph_shape_hash": getattr(log, "graph_shape_hash", None),
        "old_raw_event_shape_hash": old_raw_hash,
        "new_raw_event_shape_hash": getattr(log, "_raw_event_shape_hash", None),
    }
    log._record_operation(**history_record)
    log._has_direct_writes = False
    log._out_recipe_revision = getattr(log, "_spec_revision", 0)
    return log


def _chunked_rerun(
    log: Trace,
    model: nn.Module,
    x: Any,
    *,
    chunk_size: int,
    chunk_paths: Any | None,
    strict: bool,
    output_transform: Any | None,
) -> Trace:
    """Rerun a trace by splitting one model-ready input tree into chunks.

    Parameters
    ----------
    log:
        Trace to mutate.
    model:
        Model to execute.
    x:
        Model-ready positional input tree.
    chunk_size:
        Maximum leading-dimension chunk size.
    chunk_paths:
        Optional explicit tensor leaf paths to split.
    strict:
        Whether graph-shape divergence should raise for the first rerun.
    output_transform:
        Optional output transform to mirror for fresh captures.

    Returns
    -------
    Trace
        Mutated trace after chunked rerun.
    """

    _preflight(log, model, x)
    model = _unwrap_compiled_model(model)
    model = _unwrap_model_for_chunk_plan(model)
    x = _coerce_input_args(model, x)
    plan = plan_chunks(x, chunk_size=chunk_size, chunk_paths=chunk_paths)
    if chunk_size >= plan.total_size:
        return run(
            log,
            model,
            x,
            replay=ReplayOptions(append=False, strict=strict),
            output_transform=output_transform,
        )
    chunks = iter_chunked_inputs(x, plan)
    result = run(
        log,
        model,
        chunks[0],
        replay=ReplayOptions(append=False, strict=strict),
        output_transform=output_transform,
    )
    initial_record = {
        "engine": "rerun",
        "append": False,
        "chunk_size": min(chunk_size, plan.total_size),
        "total_batch_size": plan.total_size,
        "append_sequence_id": 0,
        "chunk_paths": normalize_chunk_paths(chunk_paths),
    }
    for chunk in chunks[1:]:
        result = run(
            result,
            model,
            chunk,
            replay=ReplayOptions(append=True, strict=strict),
            output_transform=output_transform,
        )
    result.append_history = [initial_record, *result.append_history]
    result.chunked_forward = True
    result.last_run = dict(result.last_run or {})
    result.last_run["chunk_size"] = chunk_size
    result.last_run["chunk_paths"] = normalize_chunk_paths(chunk_paths)
    return result


def _unwrap_model_for_chunk_plan(model: nn.Module) -> nn.Module:
    """Return the model shape used by rerun capture for input coercion.

    Parameters
    ----------
    model:
        Candidate rerun model.

    Returns
    -------
    nn.Module
        DataParallel-unwrapped model when applicable.
    """

    if isinstance(model, nn.DataParallel):
        return model.module
    return model


def _append_rerun(
    log: Trace,
    model: nn.Module,
    x: Any,
    *,
    strict: bool,
) -> Trace:
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
    model = _unwrap_compiled_model(model)
    _preflight_append(log, model)
    _warn_if_direct_writes_will_be_overlaid(log)
    _warn_if_batch_sensitive_train_modules(model)

    spec = getattr(log, "_intervention_spec", None)
    hook_plan = _assign_unique_plan_ids(normalize_hooks_from_spec(spec))
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
    new_log.facet_registry_snapshot = getattr(log, "facet_registry_snapshot", None)
    hook_fire_count, unfired_hook_ids = _reconcile_rerun_hook_fires(new_log, hook_plan)

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
        "hooks_fired": hook_fire_count,
        "hooks_unfired": len(unfired_hook_ids),
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
        hook_fire_count=hook_fire_count,
        unfired_hook_count=len(unfired_hook_ids),
        chunk_size=chunk_size,
        total_batch_size=total_batch_size,
        append_sequence_id=log._append_sequence_id,
        old_graph_shape_hash=old_hash,
        new_graph_shape_hash=getattr(new_log, "graph_shape_hash", None),
    )
    return log


def _is_streaming_append_active(log: Trace) -> bool:
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


def _streaming_append_error_message(log: Trace) -> str:
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


def _preflight_append(log: Trace, model: nn.Module) -> None:
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
    log: Trace,
    hook_plan: list[NormalizedHookEntry],
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


def _warn_unknown_append_helper_once(log: Trace, helper_name: str) -> None:
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
    old_log: Trace,
    new_log: Trace,
    *,
    hook_plan: list[NormalizedHookEntry],
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
    old_by_label = {
        key: layer
        for layer in old_log.layer_list
        for key in (layer._layer_label_raw, layer.layer_label)
    }
    new_by_label = {
        key: layer
        for layer in new_log.layer_list
        for key in (layer._layer_label_raw, layer.layer_label)
    }
    grads_supported = _hook_plan_supports_append_grads(hook_plan)
    for raw_label in old_labels:
        old_layer = old_by_raw[raw_label]
        new_layer = new_by_raw[raw_label]
        if _is_append_buffer_side_effect_layer(
            old_layer, old_by_label
        ) or _is_append_buffer_side_effect_layer(new_layer, new_by_label):
            continue
        _validate_append_tensor_pair(old_layer, new_layer, "out")
        _validate_append_tensor_pair(old_layer, new_layer, "transformed_out")
        _validate_append_grad_pair(old_layer, new_layer, grads_supported=grads_supported)


def _is_append_buffer_side_effect_layer(layer: Any, layer_by_raw: dict[str, Any]) -> bool:
    """Return whether ``layer`` only feeds buffer version side effects.

    Parameters
    ----------
    layer:
        Candidate layer being considered for append tensor concatenation.
    layer_by_raw:
        Mapping from raw labels to layers in the same trace pass.

    Returns
    -------
    bool
        True when every tracked child is a buffer version node created by a
        buffer write. Such producer outputs describe state transitions rather
        than batch activations and are not append-concatenated.
    """

    child_labels = list(getattr(layer, "children", []))
    if not child_labels:
        return False
    saw_buffer_write = False
    for child_label in child_labels:
        child_layer = layer_by_raw.get(child_label)
        if child_layer is None:
            return False
        if not (
            getattr(child_layer, "is_buffer", False)
            and getattr(child_layer, "buffer_write_kind", None) is not None
        ):
            return False
        saw_buffer_write = True
    return saw_buffer_write


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

    if getattr(old_layer, "is_buffer", False) or getattr(new_layer, "is_buffer", False):
        return
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


def _hook_plan_supports_append_grads(hook_plan: list[NormalizedHookEntry]) -> bool:
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


def _first_saved_batch_size(log: Trace) -> int | None:
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


def _warn_if_direct_writes_will_be_overlaid(log: Trace) -> None:
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


def _preflight(log: Trace, model: nn.Module, x: Any) -> None:
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
        raise InvalidArgumentError(
            "run(..., x=None) cannot recover the original input",
            code="run_input_missing",
            remedy="pass the forward input explicitly as log.run(model, x)",
            argument="x",
        )
    from ..user_funcs import _reject_opaque_wrappers

    _reject_opaque_wrappers(model)


def _unwrap_compiled_model(model: nn.Module) -> nn.Module:
    """Return the eager source module for a compiled rerun target.

    Parameters
    ----------
    model:
        User-supplied rerun model.

    Returns
    -------
    nn.Module
        The eager source when ``model`` is a compiled Dynamo wrapper.
    """

    from .._capture_state_helpers import unwrap_compiled_model

    return unwrap_compiled_model(model)


def _capture_with_active_spec(
    log: Trace,
    model: nn.Module,
    x: Any,
    *,
    intervention_spec: Any | None,
    hook_plan: list[NormalizedHookEntry],
    output_transform: Any | None,
) -> Trace:
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

    from ..user_funcs import (
        _run_model_and_save_specified_outs,
        _unwrap_data_parallel,
        check_model_and_input_variants,
    )

    model = _unwrap_data_parallel(model)
    x = _coerce_input_args(model, x)
    check_model_and_input_variants(model, x, {})
    save_grads_policy = getattr(log, "save_grads", None)
    grads_to_save = "all" if save_grads_policy is True else save_grads_policy
    return _run_model_and_save_specified_outs(
        model=model,
        input_args=x,
        input_kwargs={},
        output_device=getattr(log, "output_device", "same"),
        activation_transform=getattr(log, "activation_transform", None),
        grad_transform=getattr(log, "grad_transform", None),
        save_raw_activations=getattr(log, "save_raw_activations", True),
        save_raw_gradients=getattr(log, "save_raw_gradients", True),
        save_mode=getattr(log, "save_mode", "copy"),
        capture_tensor_grad_hooks=getattr(log, "capture_tensor_grad_hooks", True),
        mark_layer_depths=getattr(log, "mark_layer_depths", False),
        detach_saved_activations=getattr(log, "detach_saved_activations", False),
        save_arg_values=getattr(log, "save_arg_values", False),
        save_grads=save_grads_policy not in (None, False),
        grads_to_save=grads_to_save,
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
        # An intervention rerun retains payloads like any capture; inherit the
        # source log's configured budget rather than silently rebudgeting at
        # the default.
        save_budget=getattr(log, "save_budget", "auto"),
        output_transform=output_transform,
        save_raw_output=getattr(log, "save_raw_output", "small"),
        **_rerun_save_kwargs(log),
    )


def _rerun_save_kwargs(log: Trace) -> dict[str, Any]:
    """Return the capture save kwargs a rerun of ``log`` must use.

    The capture's recorded save request is replayed against the rerun's own
    graph, so the rerun saves what a fresh capture with the same ``save=``
    saves. The resolved raw-index save set must not be reused: a staged edit
    inserts an ``intervention_replacement`` op that shifts every later raw
    index, and the old indices then name the wrong ops. ``tl.record`` keep-op
    scopes keep their precedence; a trace with no recorded request (restored
    from pickle) falls back to the resolved scope.

    Parameters
    ----------
    log:
        Existing trace whose save request should be honored during rerun.

    Returns
    -------
    dict[str, Any]
        Save keyword arguments for ``_run_model_and_save_specified_outs``.
    """

    options = getattr(log, "_predicate_save_options", None)
    request = getattr(log, "_rerun_save_request", None)
    if request is not None and getattr(options, "keep_op", None) is None:
        return dict(request)
    layers_to_save, save_predicate, lookback, lookback_payload_policy = _rerun_save_scope(log)
    return {
        "layers_to_save": layers_to_save,
        "save_predicate": save_predicate,
        "lookback": lookback,
        "lookback_payload_policy": lookback_payload_policy,
        "retain_output_parents_for_layers_to_save": getattr(
            log, "_retain_layers_to_save_output_parents", False
        ),
    }


def _validate_rerun_result(new_log: Trace, old_log: Trace, *, strict: bool) -> int:
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
        Number of graph-shape divergence events detected by rerun validation.
    """

    old_hash = getattr(old_log, "_raw_event_shape_hash", None) or getattr(
        old_log, "graph_shape_hash", None
    )
    new_hash = getattr(new_log, "_raw_event_shape_hash", None) or getattr(
        new_log, "graph_shape_hash", None
    )
    if old_hash == new_hash:
        return 0

    message = (
        "rerun raw-event shape hash diverged from the captured graph "
        f"(old={old_hash!r}, new={new_hash!r}). TorchLens uses ordered raw operation "
        "events with normalized parent-edge indices as the conservative control-flow "
        "divergence detector."
    )
    if strict:
        raise ControlFlowDivergenceError(message)
    warnings.warn(message, ControlFlowDivergenceWarning, stacklevel=3)
    return 1


def _rerun_save_scope(log: Trace) -> tuple[str | list[int | str] | None, Any | None, int, str]:
    """Return capture save settings that mirror the original trace scope.

    Parameters
    ----------
    log:
        Existing trace whose save policy should be honored during rerun.

    Returns
    -------
    tuple[str | list[int | str] | None, Any | None, int, str]
        ``layers_to_save``, optional save predicate, predicate lookback, and
        lookback payload policy for the fresh rerun capture.
    """

    options = getattr(log, "_predicate_save_options", None)
    if options is not None and hasattr(options, "keep_op"):
        keep_op = getattr(options, "keep_op", None)
        if keep_op is not None:
            return (
                "all",
                keep_op,
                int(getattr(options, "lookback", 0)),
                # R47-11: direct attribute access, not getattr-with-default -- a
                # RecordingOptions field rename must raise here, never silently
                # fall back to "metadata_only".
                str(options.lookback_payload_policy),
            )
    if getattr(log, "num_saved_ops", 0) == 0:
        return None, None, 0, "metadata_only"
    layer_nums = getattr(log, "_layer_nums_to_save", "all")
    if layer_nums == "all":
        return "all", None, 0, "metadata_only"
    selected_indices = {int(raw_index) for raw_index in layer_nums}
    return "all", _make_raw_index_save_predicate(selected_indices), 0, "metadata_only"


def _refuse_shifted_raw_index_resave(old_log: Trace, new_log: Trace) -> None:
    """Refuse a raw-index re-save whose indices no longer name the same ops.

    A trace with no recorded save request (restored from pickle) re-saves by
    the capture's raw op indices. Those indices are exact only when the rerun
    graph matches the capture op for op; an inserted or removed op (a staged
    edit's ``intervention_replacement``, or a cleared one) shifts every later
    index, and the rerun would silently save the wrong ops.

    Parameters
    ----------
    old_log:
        Existing trace being rerun (its state is not yet replaced).
    new_log:
        Freshly captured candidate trace.

    Raises
    ------
    ControlFlowDivergenceError
        With code ``rerun_resave_ops_shifted`` when the op sequences differ.
    """

    options = getattr(old_log, "_predicate_save_options", None)
    if (
        getattr(old_log, "_rerun_save_request", None) is not None
        or getattr(options, "keep_op", None) is not None
        or getattr(old_log, "num_saved_ops", 0) == 0
        or getattr(old_log, "_layer_nums_to_save", "all") == "all"
    ):
        return
    old_labels = [layer._layer_label_raw for layer in old_log.layer_list]
    new_labels = [layer._layer_label_raw for layer in new_log.layer_list]
    if old_labels == new_labels:
        return
    first = next(
        (index for index, pair in enumerate(zip(old_labels, new_labels)) if pair[0] != pair[1]),
        min(len(old_labels), len(new_labels)),
    )
    remedy = (
        "re-capture with tl.trace(model, x, save=..., intervene=...) on the live "
        "trace instead of rerunning a restored one; the rerun has no save= of its own"
    )
    raise ControlFlowDivergenceError(
        "this trace carries no recorded save request (it was restored from pickle), "
        "so the rerun would re-save by the capture's raw op indices, and the rerun "
        f"graph has {len(new_labels)} ops against the capture's {len(old_labels)}, "
        f"first differing at position {first}; the old indices would save the wrong "
        f"ops. Remedy: {remedy}.",
        code="rerun_resave_ops_shifted",
        remedy=remedy,
        first_differing_position=first,
    )


def _make_raw_index_save_predicate(selected_indices: set[int]) -> Callable[[Any], bool]:
    """Build a predicate that mirrors a static raw-index save subset.

    Parameters
    ----------
    selected_indices:
        Raw operation indices selected by the original capture.

    Returns
    -------
    Callable[[Any], bool]
        Predicate returning ``True`` for matching capture contexts.
    """

    def keep_selected_raw_index(ctx: Any) -> bool:
        """Return whether a predicate context matches the selected raw indices."""

        raw_index = getattr(ctx, "raw_index", None)
        if raw_index is None:
            raw_index = getattr(ctx, "event_index", None)
        return raw_index in selected_indices

    return keep_selected_raw_index


def _build_ledger_record(
    log: Trace,
    *,
    started_at: float,
    old_hash: str | None,
    new_hash: str | None,
    old_raw_hash: str | None,
    new_raw_hash: str | None,
    hook_plan: list[NormalizedHookEntry],
    strict: bool,
    divergence_count: int,
    fast_refresh: bool,
    hook_fire_count: int,
    unfired_hook_count: int,
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
    old_raw_hash:
        Raw-event shape hash before rerun.
    new_raw_hash:
        Candidate raw-event shape hash from the fresh capture.
    hook_plan:
        Normalized hook entries active during rerun.
    strict:
        Whether strict divergence handling was requested.
    divergence_count:
        Number of divergence events detected.
    fast_refresh:
        Whether the rerun refreshed existing graph containers in place.
    hook_fire_count:
        Number of live hook firings observed on the candidate capture.
    unfired_hook_count:
        Number of planned hook entries that fired nowhere.

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
        "hook_fire_count": hook_fire_count,
        "unfired_hook_count": unfired_hook_count,
        "divergence_count": divergence_count,
        "fast_refresh": fast_refresh,
        "old_graph_shape_hash": old_hash,
        "new_graph_shape_hash": new_hash,
        "old_raw_event_shape_hash": old_raw_hash,
        "new_raw_event_shape_hash": new_raw_hash,
    }


def _assign_unique_plan_ids(hook_plan: list[NormalizedHookEntry]) -> list[NormalizedHookEntry]:
    """Give every planned entry a unique, stable accounting identifier.

    The fallback identifier ladder (plan id -> hook id -> helper name ->
    callable qualname) can COLLIDE across entries with different targets, and
    the fire audit compares ``Counter`` values keyed by that string: two
    fires of one entry hid the other entry's total miss (``fired=2``,
    ``unfired=()``), so a partially-applied plan claimed every entry fired
    (incomplete f9f5b140). Colliding identifiers get a stable occurrence
    suffix stamped into ``metadata["plan_id"]``, which live execution writes
    into each ``FireResult``, so the audit is per-entry; unique identifiers
    are preserved verbatim.
    """

    import dataclasses as _dataclasses

    counts = Counter(_hook_plan_identifier(entry) for entry in hook_plan)
    seen: Counter[str] = Counter()
    unique_plan: list[NormalizedHookEntry] = []
    for entry in hook_plan:
        base = _hook_plan_identifier(entry)
        if counts[base] > 1:
            metadata = dict(entry.metadata)
            metadata["plan_id"] = f"{base}#occ{seen[base]}"
            metadata["plan_id_base"] = base
            entry = _dataclasses.replace(entry, metadata=metadata)
        seen[base] += 1
        unique_plan.append(entry)
    return unique_plan


def _hook_plan_identifier(entry: NormalizedHookEntry) -> str:
    """Return the identifier written into a live ``FireResult``.

    Parameters
    ----------
    entry:
        Planned normalized hook entry.

    Returns
    -------
    str
        Plan id using the same fallback order as live execution.
    """

    if "plan_id" in entry.metadata:
        return str(entry.metadata["plan_id"])
    if "hook_id" in entry.metadata:
        return str(entry.metadata["hook_id"])
    if entry.helper_spec is not None:
        return str(entry.helper_spec.name)
    return str(getattr(entry.normalized_callable, "__qualname__", "user_hook"))


def _reconcile_rerun_hook_fires(
    new_log: Trace,
    hook_plan: list[NormalizedHookEntry],
) -> tuple[int, tuple[str, ...]]:
    """Warn when sticky rerun hook entries fire nowhere on new inputs.

    Parameters
    ----------
    new_log:
        Candidate rerun trace carrying live ``FireResult`` records.
    hook_plan:
        Hook entries planned for the rerun.

    Returns
    -------
    tuple[int, tuple[str, ...]]
        Total observed hook fires and the plan identifiers of entries with no
        corresponding fire, retaining multiplicity for duplicate plans.
    """

    plan_ids = [_hook_plan_identifier(entry) for entry in hook_plan]
    base_by_id = {
        plan_id: str(entry.metadata.get("plan_id_base", plan_id))
        for entry, plan_id in zip(hook_plan, plan_ids)
    }
    planned = Counter(plan_ids)
    fired: Counter[str] = Counter()
    # FireRecord-only ops (no FireResult) carry no plan id, only the helper
    # NAME: those fires go into a separate base-name pool consumed AFTER the
    # exact per-entry accounting, capped at the shortfall, so the legacy
    # channel keeps its multiplicity semantics without letting one entry's
    # FireResult-channel fires hide another entry's miss.
    fallback_fired: Counter[str] = Counter()
    for op in getattr(new_log, "layer_list", ()):
        fire_results = tuple(getattr(op, "fire_results", None) or ())
        if fire_results:
            fired.update(str(result.plan_id) for result in fire_results)
            continue
        fallback_fired.update(
            str(record.helper_name)
            for record in (getattr(op, "interventions", None) or ())
            if getattr(record, "direction", None) == "forward"
            and getattr(record, "helper_name", None) is not None
        )
    total_fired = sum(fired.values()) + sum(fallback_fired.values())
    unfired: list[str] = []
    for plan_id, planned_count in planned.items():
        shortfall = max(0, planned_count - fired[plan_id])
        base = base_by_id[plan_id]
        consumed = min(shortfall, fallback_fired[base])
        fallback_fired[base] -= consumed
        unfired.extend([plan_id] * (shortfall - consumed))
    if unfired:
        # First CONTRACTED S-18 row (compo wave 0): the one warning standing
        # between a user and a silently un-applied intervention carries a
        # stable machine-readable code and a structured remedy -- consumers
        # branch on ``fields["code"]``, never on message text.
        warnings.warn(
            TorchLensWarning(
                "Rerun hook plan entries fired at zero sites on the new inputs: "
                f"{unfired!r}. The rerun completed, but those interventions were "
                "no-ops. Remedy: resolve the target sites against the rerun trace "
                "(trace.resolve_sites) before re-applying, or route the edit "
                "through the push engine (fork().do(...)), which validates sites "
                "at plan time",
                code="rerun_zero_fire",
                unfired_plan_ids=list(unfired),
            ),
            stacklevel=3,
        )
    return total_fired, tuple(unfired)


__all__ = ["run"]
