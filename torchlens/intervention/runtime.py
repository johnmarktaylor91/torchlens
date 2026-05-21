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
    """Prevent recursive TorchLens tracing while hook code is active.

    Phase 3 defines the guard object only. Phase 4a will connect it to
    ``active_logging()`` so ``trace`` can fail before entering a
    nested logging context.
    """

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
            f"hook returned None at {_site_name(hook_context)}; Phase 3 requires a tensor return"
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
        if not live_selector_matches_site(normalized_entry.site_target, site):
            continue

        hook_context = make_hook_context(
            name=_hook_display_name(normalized_entry),
            timing="post",
            direction="forward",
            layer_log=site,
            run_ctx=_live_run_ctx(),
            args=(current_out,),
            kwargs={},
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
        )
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


def _apply_live_backward_hooks(
    grad_input: tuple[torch.Tensor | None, ...] | None,
    grad_output: tuple[torch.Tensor | None, ...] | None,
    grad_fn_handle: Any,
    call_index: int,
) -> tuple[torch.Tensor | None, ...] | None:
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
    tuple[torch.Tensor | None, ...] | None
        Mutated grad_input tuple, or None when no helper mutates it.
    """

    hook_plan = _state._active_hook_plan
    if not hook_plan or grad_input is None:
        return None

    current = grad_input
    mutated = False
    for entry in hook_plan:
        normalized_entry = _coerce_hook_entry(entry)
        if normalized_entry.metadata.get("direction", "forward") != "backward":
            continue
        if not live_backward_selector_matches(
            normalized_entry.site_target, grad_fn_handle, call_index
        ):
            continue
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
    return current if mutated else None


def _apply_live_backward_prehooks(
    grad_input: tuple[torch.Tensor | None, ...],
    grad_fn_handle: Any,
    call_index: int,
) -> tuple[torch.Tensor | None, ...] | None:
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
    tuple[torch.Tensor | None, ...] | None
        Mutated grad_input tuple, or None when no helper mutates it.
    """

    hook_plan = _state._active_hook_plan
    if not hook_plan:
        return None

    current = grad_input
    mutated = False
    for entry in hook_plan:
        normalized_entry = _coerce_hook_entry(entry)
        if normalized_entry.metadata.get("direction", "forward") != "backward":
            continue
        if not live_backward_selector_matches(
            normalized_entry.site_target, grad_fn_handle, call_index
        ):
            continue
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
    return current if mutated else None


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
    )


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
