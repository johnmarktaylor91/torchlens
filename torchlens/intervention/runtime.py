"""Runtime context ownership for TorchLens intervention execution."""

from __future__ import annotations

from contextlib import contextmanager
import time
from collections.abc import Iterator
from typing import Any

import torch

from .. import _state
from .._state import pause_logging
from ..backends.torch._tl import copy_replacement_meta
from ..ir.intervention import FireResult
from .errors import HookSignatureError, HookValueError
from .hooks import (
    HookContext,
    NormalizedHookEntry,
    live_backward_selector_matches,
    live_selector_matches_site,
    make_hook_context,
    make_live_site_proxy,
)
from .types import FireRecord


@contextmanager
def active_intervention_context(
    *,
    intervention_spec: Any | None,
    hook_plan: Any | None,
) -> Iterator[None]:
    """Temporarily install a rerun/live intervention context in global state.

    Parameters
    ----------
    intervention_spec:
        Active intervention spec for the capture.
    hook_plan:
        Normalized hook entries consumed by live wrapper dispatch.

    Yields
    ------
    None
        Control while the context is installed.
    """

    previous_spec = _state._active_intervention_spec
    previous_hook_plan = _state._active_hook_plan
    _state._active_intervention_spec = intervention_spec
    _state._active_hook_plan = hook_plan
    try:
        yield
    finally:
        _state._active_intervention_spec = previous_spec
        _state._active_hook_plan = previous_hook_plan


class _HookReentrancyGuard:
    """Track recursive TorchLens hook execution depth in global state."""

    def __init__(self) -> None:
        """Initialise an inactive re-entrancy guard."""

        self.depth = 0
        self.active_log_id: int | None = None

    @property
    def active(self) -> bool:
        """Return whether hook execution is currently active.

        Returns
        -------
        bool
            Whether at least one hook is on the call stack.
        """

        return self.depth > 0

    def __enter__(self) -> "_HookReentrancyGuard":
        """Enter hook execution.

        Returns
        -------
        _HookReentrancyGuard
            This guard.
        """

        self.depth += 1
        _state._hook_reentrancy_depth = self.depth
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        """Leave hook execution.

        Parameters
        ----------
        exc_type:
            Exception type, if any.
        exc:
            Exception value, if any.
        traceback:
            Exception traceback, if any.
        """

        self.depth = max(0, self.depth - 1)
        _state._hook_reentrancy_depth = self.depth
        if self.depth == 0:
            self.active_log_id = None


HOOK_REENTRANCY_GUARD = _HookReentrancyGuard()


def _execute_hook(
    hook_callable: Any,
    out: torch.Tensor,
    hook_context: HookContext,
    *,
    force_shape_change: bool = False,
) -> torch.Tensor:
    """Run and validate one hook callable under ``pause_logging()``.

    Parameters
    ----------
    hook_callable:
        User or helper hook callable.
    out:
        Current out tensor at the hook site.
    hook_context:
        Metadata snapshot passed as the keyword-only ``hook`` argument.
    force_shape_change:
        Escape hatch allowing dtype, device, and shape changes.

    Returns
    -------
    torch.Tensor
        Replacement out tensor.

    Raises
    ------
    HookSignatureError
        If the callable cannot be invoked with the hook signature.
    HookValueError
        If the callable returns ``None`` or an incompatible value.
    """

    try:
        with HOOK_REENTRANCY_GUARD:
            with pause_logging():
                result = hook_callable(out, hook=hook_context)
    except TypeError as exc:
        raise HookSignatureError(
            f"hook {hook_context.name!r} could not be called at "
            f"{_site_name(hook_context)} with signature (out, *, hook)"
        ) from exc
    return validate_hook_output(
        result,
        out,
        hook_context=hook_context,
        force_shape_change=force_shape_change,
    )


def validate_hook_output(
    result: Any,
    out: torch.Tensor,
    *,
    hook_context: HookContext | None = None,
    force_shape_change: bool = False,
) -> torch.Tensor:
    """Validate a hook return value against the input out metadata.

    Parameters
    ----------
    result:
        Hook return value.
    out:
        Original out tensor.
    hook_context:
        Optional context for error messages.
    force_shape_change:
        If true, allow dtype, device, and shape changes.

    Returns
    -------
    torch.Tensor
        Validated replacement tensor.

    Raises
    ------
    HookValueError
        If the return value is invalid.
    """

    if result is None:
        raise HookValueError(
            f"hook returned None at {_site_name(hook_context)}; expected torch.Tensor"
        )
    if not isinstance(result, torch.Tensor):
        raise HookValueError(
            f"hook returned {type(result).__name__} at {_site_name(hook_context)}; "
            "expected torch.Tensor"
        )
    if force_shape_change:
        _copy_tl_replacement_attrs(out, result)
        return result
    if result.dtype != out.dtype:
        raise HookValueError(
            f"hook returned dtype {result.dtype} at {_site_name(hook_context)}; "
            f"expected {out.dtype}"
        )
    if result.device != out.device:
        raise HookValueError(
            f"hook returned device {result.device} at {_site_name(hook_context)}; "
            f"expected {out.device}"
        )
    if tuple(result.shape) != tuple(out.shape):
        raise HookValueError(
            f"hook returned shape {tuple(result.shape)} at {_site_name(hook_context)}; "
            f"expected {tuple(out.shape)}"
        )
    _copy_tl_replacement_attrs(out, result)
    return result


def _copy_tl_replacement_attrs(source: torch.Tensor, replacement: torch.Tensor) -> None:
    """Copy TorchLens tensor metadata from an original out to a replacement.

    Parameters
    ----------
    source:
        Original activation supplied to a user hook.
    replacement:
        Tensor returned by the hook.

    Returns
    -------
    None
        The replacement tensor is annotated in place when PyTorch permits
        dynamic tensor attributes.
    """

    if replacement is source:
        return
    try:
        copy_replacement_meta(source, replacement)
    except Exception:
        pass


def _apply_live_hooks(
    out: torch.Tensor,
    *,
    site: Any,
    container_path: tuple[Any, ...] = (),
    call_args: tuple[Any, ...] = (),
    call_kwargs: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, tuple[FireResult, ...]]:
    """Apply active live post-hooks to one capture-time output tensor.

    Parameters
    ----------
    out:
        Tensor returned by the decorated torch function after in-place safe-copy.
    site:
        Capture-time site proxy for selector matching and hook context.
    container_path:
        Stable path inside a multi-output container.
    call_args:
        Original positional call inputs for input-routed helpers.
    call_kwargs:
        Original keyword call inputs for input-routed helpers.

    Returns
    -------
    tuple[torch.Tensor, tuple[FireResult, ...]]
        Original or hook-replaced tensor plus typed fire results for the
        corresponding capture event.
    """

    hook_plan = _state._active_hook_plan
    if not hook_plan:
        return out, ()

    current_out = out
    fire_results: list[FireResult] = []
    for entry in hook_plan:
        normalized_entry = _coerce_hook_entry(entry)
        if normalized_entry.metadata.get("direction", "forward") != "forward":
            continue
        if normalized_entry.metadata.get("timing", "post") != "post":
            continue
        module_scope = _input_splice_module_scope(normalized_entry.site_target, normalized_entry)
        if module_scope is not None and not getattr(site, "_tl_module_boundary", False):
            if _is_plain_module_selector(normalized_entry.site_target):
                continue
            raise HookValueError(
                "splice_module(input='in') with a module-scoped op selector is ambiguous. "
                "Use tl.module(...)/tl.in_module(...) alone for one module-call splice, or use "
                "an op selector without tl.in_module(...) for op-level splicing."
            )
        if not live_selector_matches_site(normalized_entry.site_target, site):
            continue

        hook_args, hook_kwargs = _hook_call_inputs_for_site(
            normalized_entry,
            site=site,
            call_args=call_args,
            call_kwargs=call_kwargs,
        )
        hook_context = make_hook_context(
            name=_hook_display_name(normalized_entry),
            timing="post",
            direction="forward",
            layer_log=site,
            run_ctx=_live_run_ctx(),
            args=hook_args or (current_out,),
            kwargs=hook_kwargs,
        )
        previous_notes = tuple(hook_context.run_ctx.get("ledger_notes", ()))
        pre_hook_shape = tuple(current_out.shape)
        pre_hook_dtype = str(current_out.dtype)
        result = _execute_hook(
            normalized_entry.normalized_callable,
            current_out,
            hook_context,
            force_shape_change=bool(normalized_entry.metadata.get("force_shape_change", False)),
        )
        record = _build_live_fire_record(
            normalized_entry,
            site=site,
            container_path=container_path,
            previous_notes=previous_notes,
            run_ctx=hook_context.run_ctx,
            replaced=result is not current_out,
        )
        _append_active_spec_records([record])
        fire_results.append(
            FireResult(
                plan_id=str(
                    normalized_entry.metadata.get(
                        "plan_id",
                        normalized_entry.metadata.get(
                            "hook_id", _hook_display_name(normalized_entry)
                        ),
                    )
                ),
                site_label=site._layer_label_raw,
                fired_at_capture_index=int(getattr(site, "raw_index", 0) or 0),
                pre_hook_shape=pre_hook_shape,
                post_hook_shape=tuple(result.shape),
                pre_hook_dtype=pre_hook_dtype,
                post_hook_dtype=str(result.dtype),
                replaced=result is not current_out,
                fire_record=record,
            )
        )
        current_out = result
    return current_out, tuple(fire_results)


def _apply_module_boundary_live_hooks(
    out_orig: Any,
    *,
    module_address: str,
    module_call_index: int,
    module_type: str,
    call_args: tuple[Any, ...],
    call_kwargs: dict[str, Any],
) -> Any:
    """Apply module-boundary live hooks to module forward outputs.

    Parameters
    ----------
    out_orig:
        Raw module forward output.
    module_address:
        TorchLens module address.
    module_call_index:
        One-based module call index.
    module_type:
        Module type name.
    call_args:
        Original module positional inputs.
    call_kwargs:
        Original module keyword inputs.

    Returns
    -------
    Any
        Module output with any tensor replacements applied.
    """

    if not _state._active_hook_plan:
        return out_orig
    module_call = (module_address, module_call_index)
    replacements: dict[tuple[Any, ...], torch.Tensor] = {}
    for out, container_path in _iter_tensor_outputs(out_orig):
        site = make_live_site_proxy(
            _layer_label_raw=f"{module_address}:{module_call_index}",
            func_name=module_type,
            layer_type=module_type.lower(),
            tensor=out,
            func_call_id=0,
            container_path=container_path,
            fields={
                "raw_index": 0,
                "module": module_call,
                "modules": (module_call,),
                "output_of_module_calls": (module_call,),
                "_tl_module_boundary": True,
            },
        )
        setattr(site, "_tl_module_boundary", True)
        hooked, fire_results = _apply_live_hooks(
            out,
            site=site,
            container_path=container_path,
            call_args=call_args,
            call_kwargs=call_kwargs,
        )
        if fire_results:
            _set_tensor_live_fire_results_if_available(hooked, fire_results)
        if hooked is not out:
            replacements[container_path] = hooked
    if not replacements:
        return out_orig
    return _replace_tensor_outputs(out_orig, replacements)


def _set_tensor_live_fire_results_if_available(
    tensor: torch.Tensor, fire_results: tuple[FireResult, ...]
) -> None:
    """Attach fire results to a tensor when dynamic attributes are available.

    Parameters
    ----------
    tensor:
        Tensor that received a live module-boundary hook.
    fire_results:
        Fire results emitted by the live hook dispatcher.
    """

    try:
        setattr(tensor, "_tl_live_fire_results", fire_results)
    except Exception:
        pass


def _iter_tensor_outputs(
    value: Any, path: tuple[Any, ...] = ()
) -> Iterator[tuple[torch.Tensor, tuple[Any, ...]]]:
    """Yield tensor leaves and their paths from a module output structure.

    Parameters
    ----------
    value:
        Module output value to traverse.
    path:
        Current container path prefix.

    Yields
    ------
    tuple[torch.Tensor, tuple[Any, ...]]
        Tensor leaf and stable path.
    """

    if isinstance(value, torch.Tensor):
        yield value, path
        return
    if isinstance(value, tuple):
        for index, item in enumerate(value):
            yield from _iter_tensor_outputs(item, (*path, index))
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            yield from _iter_tensor_outputs(item, (*path, index))
        return
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _iter_tensor_outputs(item, (*path, key))


def _replace_tensor_outputs(value: Any, replacements: dict[tuple[Any, ...], torch.Tensor]) -> Any:
    """Return ``value`` with tensor leaves replaced by path.

    Parameters
    ----------
    value:
        Original module output structure.
    replacements:
        Mapping from tensor leaf path to replacement tensor.

    Returns
    -------
    Any
        Output structure with replacements applied.
    """

    if () in replacements:
        return replacements[()]
    if isinstance(value, tuple):
        return type(value)(
            _replace_tensor_outputs_by_child(item, replacements, (index,))
            for index, item in enumerate(value)
        )
    if isinstance(value, list):
        return [
            _replace_tensor_outputs_by_child(item, replacements, (index,))
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        return {
            key: _replace_tensor_outputs_by_child(item, replacements, (key,))
            for key, item in value.items()
        }
    return value


def _replace_tensor_outputs_by_child(
    value: Any, replacements: dict[tuple[Any, ...], torch.Tensor], prefix: tuple[Any, ...]
) -> Any:
    """Return a child value with replacements beneath ``prefix`` applied.

    Parameters
    ----------
    value:
        Child value to rebuild.
    replacements:
        Full replacement mapping.
    prefix:
        Path prefix for ``value`` within its parent.

    Returns
    -------
    Any
        Child value with matching replacements applied.
    """

    child_replacements = {
        path[len(prefix) :]: replacement
        for path, replacement in replacements.items()
        if path[: len(prefix)] == prefix
    }
    if not child_replacements:
        return value
    return _replace_tensor_outputs(value, child_replacements)


def _hook_call_inputs_for_site(
    entry: NormalizedHookEntry,
    *,
    site: Any,
    call_args: tuple[Any, ...],
    call_kwargs: dict[str, Any] | None,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Return the positional and keyword inputs for one hook fire.

    Parameters
    ----------
    entry:
        Normalized hook entry being executed.
    site:
        Live site proxy.
    call_args:
        Captured positional inputs.
    call_kwargs:
        Captured keyword inputs.

    Returns
    -------
    tuple[tuple[Any, ...], dict[str, Any]]
        Hook positional and keyword inputs.
    """

    if _input_splice_module_scope(entry.site_target, entry) is None:
        return call_args, dict(call_kwargs or {})
    if not getattr(site, "_tl_module_boundary", False):
        return call_args, dict(call_kwargs or {})
    return call_args, dict(call_kwargs or {})


def _input_splice_module_scope(site_target: Any, entry: NormalizedHookEntry) -> str | None:
    """Return the module address for input-spliced module-scoped selectors.

    Parameters
    ----------
    site_target:
        Selector-like hook target.
    entry:
        Normalized hook entry.

    Returns
    -------
    str | None
        Module address when the entry is an input-spliced module scope.
    """

    helper = entry.helper_spec
    if helper is None or helper.name != "splice_module":
        return None
    helper_metadata = dict(helper.metadata)
    if helper_metadata.get("input") != "in":
        return None
    return _module_scope_address(site_target)


def _module_scope_address(site_target: Any) -> str | None:
    """Return a module address when a selector tree contains module scope.

    Parameters
    ----------
    site_target:
        Selector-like hook target.

    Returns
    -------
    str | None
        First module address found in the selector tree.
    """

    kind = getattr(site_target, "selector_kind", None)
    if kind in {"module", "in_module"}:
        value = getattr(site_target, "selector_value", None)
        return None if value is None else str(value)
    selectors = getattr(site_target, "selectors", None)
    if selectors is not None:
        for selector in selectors:
            address = _module_scope_address(selector)
            if address is not None:
                return address
    selector = getattr(site_target, "selector", None)
    if selector is not None:
        return _module_scope_address(selector)
    return None


def _is_plain_module_selector(site_target: Any) -> bool:
    """Return whether a selector is exactly a module boundary or containment selector.

    Parameters
    ----------
    site_target:
        Selector-like hook target.

    Returns
    -------
    bool
        Whether the selector is exactly ``tl.module`` or ``tl.in_module``.
    """

    return getattr(site_target, "selector_kind", None) in {"module", "in_module"}


def _apply_live_backward_hooks(
    grad_input: tuple[torch.Tensor | None, ...] | None,
    grad_output: tuple[torch.Tensor | None, ...] | None,
    grad_fn_handle: Any,
    call_index: int,
) -> tuple[tuple[torch.Tensor | None, ...] | None, tuple[FireRecord, ...]]:
    """Apply active grad_fn_handle post-hook helpers.

    Parameters
    ----------
    grad_input:
        Current autograd grad_input tuple.
    grad_output:
        Autograd grad_output tuple.
    grad_fn_handle:
        GradFn site for selector matching.
    call_index:
        One-based grad_fn_handle call index.

    Returns
    -------
    tuple[tuple[torch.Tensor | None, ...] | None, tuple[FireRecord, ...]]
        Mutated grad_input tuple, or None when no helper mutates it, plus
        immutable fire records for every matching helper.
    """

    hook_plan = _state._active_hook_plan
    if not hook_plan or grad_input is None:
        return None, ()

    current = grad_input
    mutated = False
    fire_records: list[FireRecord] = []
    for entry in hook_plan:
        normalized_entry = _coerce_hook_entry(entry)
        if normalized_entry.metadata.get("direction", "forward") != "backward":
            continue
        if not live_backward_selector_matches(
            normalized_entry.site_target,
            grad_fn_handle,
            call_index,
            grad_input=current,
            grad_output=grad_output,
        ):
            continue
        previous = current
        with HOOK_REENTRANCY_GUARD:
            with pause_logging():
                result = normalized_entry.normalized_callable(
                    current,
                    grad_output=grad_output,
                    grad_fn_handle=grad_fn_handle,
                    call_index=call_index,
                    run_ctx=_live_run_ctx(),
                )
        if result is not None:
            current = _validate_grad_tuple(result, current, grad_fn_handle=grad_fn_handle)
            mutated = True
        fire_records.append(
            _build_live_backward_fire_record(
                normalized_entry,
                grad_fn_handle=grad_fn_handle,
                call_index=call_index,
                grad_kind="grad_input",
                timing="post",
                previous=previous,
                current=current,
            )
        )
    _append_active_spec_records(fire_records)
    return (current if mutated else None), tuple(fire_records)


def _apply_live_backward_prehooks(
    grad_input: tuple[torch.Tensor | None, ...],
    grad_fn_handle: Any,
    call_index: int,
) -> tuple[tuple[torch.Tensor | None, ...] | None, tuple[FireRecord, ...]]:
    """Apply active AccumulateGrad prehook helpers.

    Parameters
    ----------
    grad_input:
        Current autograd prehook grad_input tuple.
    grad_fn_handle:
        GradFn site for selector matching.
    call_index:
        One-based grad_fn_handle call index expected for the matching post-hook.

    Returns
    -------
    tuple[tuple[torch.Tensor | None, ...] | None, tuple[FireRecord, ...]]
        Mutated grad_input tuple, or None when no helper mutates it, plus
        immutable fire records for every matching helper.
    """

    hook_plan = _state._active_hook_plan
    if not hook_plan:
        return None, ()

    current = grad_input
    mutated = False
    fire_records: list[FireRecord] = []
    for entry in hook_plan:
        normalized_entry = _coerce_hook_entry(entry)
        if normalized_entry.metadata.get("direction", "forward") != "backward":
            continue
        if not live_backward_selector_matches(
            normalized_entry.site_target,
            grad_fn_handle,
            call_index,
            grad_input=current,
            grad_output=None,
        ):
            continue
        previous = current
        with HOOK_REENTRANCY_GUARD:
            with pause_logging():
                result = normalized_entry.normalized_callable(
                    current,
                    grad_output=None,
                    grad_fn_handle=grad_fn_handle,
                    call_index=call_index,
                    run_ctx=_live_run_ctx(),
                )
        if result is not None:
            current = _validate_grad_tuple(result, current, grad_fn_handle=grad_fn_handle)
            mutated = True
        fire_records.append(
            _build_live_backward_fire_record(
                normalized_entry,
                grad_fn_handle=grad_fn_handle,
                call_index=call_index,
                grad_kind="grad_input",
                timing="pre",
                previous=previous,
                current=current,
            )
        )
    _append_active_spec_records(fire_records)
    return (current if mutated else None), tuple(fire_records)


def _validate_grad_tuple(
    result: Any,
    reference: tuple[torch.Tensor | None, ...],
    *,
    grad_fn_handle: Any,
) -> tuple[torch.Tensor | None, ...]:
    """Validate a grad_fn_handle helper return tuple.

    Parameters
    ----------
    result:
        Helper return value.
    reference:
        Original grad tuple.
    grad_fn_handle:
        GradFn used in diagnostics.

    Returns
    -------
    tuple[torch.Tensor | None, ...]
        Validated tuple.
    """

    if not isinstance(result, tuple):
        raise HookValueError(
            f"backward helper at {getattr(grad_fn_handle, 'label', '<unknown>')} returned "
            f"{type(result).__name__}; expected tuple or None"
        )
    if len(result) != len(reference):
        raise HookValueError(
            f"backward helper at {getattr(grad_fn_handle, 'label', '<unknown>')} returned "
            f"{len(result)} gradients; expected {len(reference)}"
        )
    return result


def _coerce_hook_entry(entry: Any) -> NormalizedHookEntry:
    """Return a normalized hook entry or raise a deterministic type error.

    Parameters
    ----------
    entry:
        Hook-plan entry from runtime state.

    Returns
    -------
    NormalizedHookEntry
        Valid normalized entry.
    """

    if isinstance(entry, NormalizedHookEntry):
        return entry
    raise HookValueError("live hook execution requires a normalized hook plan entry")


def _live_run_ctx() -> dict[str, Any]:
    """Return the shared run context for the active model log.

    Returns
    -------
    dict[str, Any]
        Mutable context shared by hooks in this live run.
    """

    trace = _state._active_trace
    if trace is None:
        return {}
    run_ctx = getattr(trace, "last_run", None)
    if run_ctx is None:
        run_ctx = {"engine": "live", "timestamp": time.monotonic()}
        trace.last_run = run_ctx
    else:
        run_ctx.setdefault("engine", "live")
        run_ctx.setdefault("timestamp", time.monotonic())
    return run_ctx


def _hook_display_name(entry: NormalizedHookEntry) -> str:
    """Return a stable display name for a hook entry.

    Parameters
    ----------
    entry:
        Normalized hook entry.

    Returns
    -------
    str
        Helper name or callable qualname.
    """

    if entry.helper_spec is not None:
        return entry.helper_spec.name
    return getattr(entry.normalized_callable, "__qualname__", "user_hook")


def _build_live_fire_record(
    entry: NormalizedHookEntry,
    *,
    site: Any,
    container_path: tuple[Any, ...],
    previous_notes: tuple[Any, ...],
    run_ctx: dict[str, Any],
    replaced: bool,
) -> FireRecord:
    """Build a record for one live hook fire.

    Parameters
    ----------
    entry:
        Hook entry that fired.
    site:
        Capture-time site proxy.
    container_path:
        Output path for the hooked tensor.
    previous_notes:
        Operation-history notes present before hook execution.
    run_ctx:
        Shared hook run context after execution.
    replaced:
        Whether the hook returned a replacement tensor object.

    Returns
    -------
    FireRecord
        Immutable fire record appended to the eventual layer pass.
    """

    helper_name = _hook_display_name(entry)
    new_notes = tuple(run_ctx.get("ledger_notes", ()))[len(previous_notes) :]
    helper_kwargs = dict(entry.helper_spec.kwargs) if entry.helper_spec is not None else {}
    return FireRecord(
        target_label=site._layer_label_raw,
        call_label=site._layer_label_raw,
        func_call_id=site.func_call_id,
        container_path=container_path,
        engine="live",
        helper=entry.helper_spec,
        site_label=site._layer_label_raw,
        timing="post",
        direction="forward",
        helper_name=helper_name,
        seed=helper_kwargs.get("seed"),
        determinism_note="; ".join(str(note) for note in new_notes) if new_notes else None,
        timestamp=time.monotonic(),
        replaced=replaced,
    )


def _build_live_backward_fire_record(
    entry: NormalizedHookEntry,
    *,
    grad_fn_handle: Any,
    call_index: int,
    grad_kind: str,
    timing: str,
    previous: tuple[torch.Tensor | None, ...],
    current: tuple[torch.Tensor | None, ...],
) -> FireRecord:
    """Build an audit record for one live backward hook fire.

    Parameters
    ----------
    entry:
        Hook entry that fired.
    grad_fn_handle:
        Backward site receiving the hook.
    call_index:
        One-based callback index for this grad_fn.
    grad_kind:
        Gradient tuple kind mutated by the hook.
    timing:
        Execution timing for the callback site.
    previous:
        Tuple before this helper ran.
    current:
        Tuple after this helper ran.

    Returns
    -------
    FireRecord
        Immutable backward fire record.
    """

    helper_kwargs = dict(entry.helper_spec.kwargs) if entry.helper_spec is not None else {}
    tuple_index = _first_replaced_tuple_index(previous, current)
    label = str(getattr(grad_fn_handle, "label", ""))
    pass_index = _active_backward_pass_index(grad_fn_handle)
    return FireRecord(
        target_label=label,
        call_label=f"{label}:{call_index}" if label else str(call_index),
        func_call_id=None,
        container_path=(),
        engine="live",
        helper=entry.helper_spec,
        site_label=label,
        timing=timing,  # type: ignore[arg-type]
        direction="backward",
        helper_name=_hook_display_name(entry),
        seed=helper_kwargs.get("seed"),
        timestamp=time.monotonic(),
        backward_pass_index=pass_index,
        call_index=call_index,
        grad_kind=grad_kind,  # type: ignore[arg-type]
        tuple_index=tuple_index,
        replaced=tuple_index is not None or current is not previous,
    )


def _first_replaced_tuple_index(
    previous: tuple[torch.Tensor | None, ...],
    current: tuple[torch.Tensor | None, ...],
) -> int | None:
    """Return the first tuple slot whose object identity changed.

    Parameters
    ----------
    previous:
        Tuple before helper execution.
    current:
        Tuple after helper execution.

    Returns
    -------
    int | None
        First changed slot index, or ``None`` when object identities match.
    """

    for index, (before, after) in enumerate(zip(previous, current, strict=False)):
        if before is not after:
            return index
    return None


def _active_backward_pass_index(grad_fn_handle: Any) -> int | None:
    """Return the active trace backward pass index for a grad_fn handle.

    Parameters
    ----------
    grad_fn_handle:
        Runtime GradFn record or compatible test stub.

    Returns
    -------
    int | None
        Active one-based backward pass index, when available.
    """

    trace = getattr(grad_fn_handle, "source_trace", None)
    if trace is None:
        return None
    value = getattr(trace, "_active_backward_pass_index", None)
    return int(value) if isinstance(value, int) else None


def _append_active_spec_records(records: list[FireRecord]) -> None:
    """Append live fire records to the active intervention spec ledger.

    Parameters
    ----------
    records:
        Fire records produced by the current live callback.

    Returns
    -------
    None
        Mutates the active intervention spec when one is installed.
    """

    if not records:
        return
    spec = _state._active_intervention_spec
    if spec is not None and hasattr(spec, "records"):
        spec.records.extend(records)


def _site_name(hook_context: HookContext | None) -> str:
    """Return a readable site name for hook diagnostics.

    Parameters
    ----------
    hook_context:
        Optional hook context.

    Returns
    -------
    str
        Layer label or ``"<unknown site>"``.
    """

    if hook_context is None:
        return "<unknown site>"
    layer_label = hook_context.layer_log.get("layer_label")
    if layer_label is None:
        return "<unknown site>"
    return str(layer_label)


def do(log: Any, *args: Any, **kwargs: Any) -> Any:
    """Apply a one-shot intervention operation to a model log.

    Parameters
    ----------
    log:
        Trace-like object that will eventually receive the operation.
    *args:
        Positional arguments forwarded to ``log.do``.
    **kwargs:
        Keyword arguments forwarded to ``log.do``.

    Returns
    -------
    Any
        Operation result from ``log.do``.
    """

    return log.do(*args, **kwargs)


__all__ = [
    "HOOK_REENTRANCY_GUARD",
    "_HookReentrancyGuard",
    "_apply_live_hooks",
    "_apply_live_backward_hooks",
    "_apply_live_backward_prehooks",
    "_execute_hook",
    "active_intervention_context",
    "do",
    "validate_hook_output",
]
