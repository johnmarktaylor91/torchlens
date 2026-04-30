"""Runtime context ownership for TorchLens intervention execution."""

from __future__ import annotations

import time
from typing import Any

import torch

from .. import _state
from .._state import pause_logging
from .errors import HookSignatureError, HookValueError, _not_implemented
from .hooks import HookContext, NormalizedHookEntry, make_hook_context, live_selector_matches_site
from .types import FireRecord


class _HookReentrancyGuard:
    """Prevent recursive TorchLens tracing while hook code is active.

    Phase 3 defines the guard object only. Phase 4a will connect it to
    ``active_logging()`` so ``log_forward_pass`` can fail before entering a
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
    activation: torch.Tensor,
    hook_context: HookContext,
    *,
    force_shape_change: bool = False,
) -> torch.Tensor:
    """Run and validate one hook callable under ``pause_logging()``.

    Parameters
    ----------
    hook_callable:
        User or helper hook callable.
    activation:
        Current activation tensor at the hook site.
    hook_context:
        Metadata snapshot passed as the keyword-only ``hook`` argument.
    force_shape_change:
        Escape hatch allowing dtype, device, and shape changes.

    Returns
    -------
    torch.Tensor
        Replacement activation tensor.

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
                result = hook_callable(activation, hook=hook_context)
    except TypeError as exc:
        raise HookSignatureError(
            f"hook {hook_context.name!r} could not be called at "
            f"{_site_name(hook_context)} with signature (activation, *, hook)"
        ) from exc
    return validate_hook_output(
        result,
        activation,
        hook_context=hook_context,
        force_shape_change=force_shape_change,
    )


def validate_hook_output(
    result: Any,
    activation: torch.Tensor,
    *,
    hook_context: HookContext | None = None,
    force_shape_change: bool = False,
) -> torch.Tensor:
    """Validate a hook return value against the input activation metadata.

    Parameters
    ----------
    result:
        Hook return value.
    activation:
        Original activation tensor.
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
        return result
    if result.dtype != activation.dtype:
        raise HookValueError(
            f"hook returned dtype {result.dtype} at {_site_name(hook_context)}; "
            f"expected {activation.dtype}"
        )
    if result.device != activation.device:
        raise HookValueError(
            f"hook returned device {result.device} at {_site_name(hook_context)}; "
            f"expected {activation.device}"
        )
    if tuple(result.shape) != tuple(activation.shape):
        raise HookValueError(
            f"hook returned shape {tuple(result.shape)} at {_site_name(hook_context)}; "
            f"expected {tuple(activation.shape)}"
        )
    return result


def _apply_live_hooks(
    activation: torch.Tensor,
    *,
    site: Any,
    output_path: tuple[Any, ...] = (),
) -> torch.Tensor:
    """Apply active live post-hooks to one capture-time output tensor.

    Parameters
    ----------
    activation:
        Tensor returned by the decorated torch function after in-place safe-copy.
    site:
        Capture-time site proxy for selector matching and hook context.
    output_path:
        Stable path inside a multi-output container.

    Returns
    -------
    torch.Tensor
        Original or hook-replaced tensor to pass into output logging.
    """

    hook_plan = _state._active_hook_plan
    if not hook_plan:
        return activation

    current_activation = activation
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
            args=(current_activation,),
            kwargs={},
        )
        previous_notes = tuple(hook_context.run_ctx.get("operation_history_notes", ()))
        result = _execute_hook(
            normalized_entry.normalized_callable,
            current_activation,
            hook_context,
            force_shape_change=bool(normalized_entry.metadata.get("force_shape_change", False)),
        )
        record = _build_live_fire_record(
            normalized_entry,
            site=site,
            output_path=output_path,
            previous_notes=previous_notes,
            run_ctx=hook_context.run_ctx,
        )
        _state._pending_live_fire_records.setdefault(site.layer_label_raw, []).append(record)
        current_activation = result
    return current_activation


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

    model_log = _state._active_model_log
    if model_log is None:
        return {}
    run_ctx = getattr(model_log, "last_run_ctx", None)
    if run_ctx is None:
        run_ctx = {}
        model_log.last_run_ctx = run_ctx
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
    output_path: tuple[Any, ...],
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
    output_path:
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
    new_notes = tuple(run_ctx.get("operation_history_notes", ()))[len(previous_notes) :]
    helper_kwargs = dict(entry.helper_spec.kwargs) if entry.helper_spec is not None else {}
    return FireRecord(
        target_label=site.layer_label_raw,
        pass_label=site.layer_label_raw,
        func_call_id=site.func_call_id,
        output_path=output_path,
        engine="live",
        helper=entry.helper_spec,
        site_label=site.layer_label_raw,
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
    """Apply a future one-shot intervention operation to a model log.

    Parameters
    ----------
    log:
        ModelLog-like object that will eventually receive the operation.
    *args:
        Reserved positional arguments for future site and hook/value inputs.
    **kwargs:
        Reserved keyword arguments for future engine dispatch.

    Returns
    -------
    Any
        Reserved operation result.

    Raises
    ------
    NotImplementedError
        Always raised until Phase 8b implements the dispatch operation.
    """

    return _not_implemented("do", "Phase 8b")


__all__ = [
    "HOOK_REENTRANCY_GUARD",
    "_HookReentrancyGuard",
    "_apply_live_hooks",
    "_execute_hook",
    "do",
    "validate_hook_output",
]
