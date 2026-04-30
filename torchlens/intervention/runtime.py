"""Runtime context ownership for TorchLens intervention execution."""

from __future__ import annotations

from typing import Any

import torch

from .._state import pause_logging
from .errors import HookSignatureError, HookValueError, _not_implemented
from .hooks import HookContext


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
    "_execute_hook",
    "do",
    "validate_hook_output",
]
