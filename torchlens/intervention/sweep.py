"""Convenience constructor for value-sweep intervention bundles."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import torch
from torch import nn

from .._deprecations import MISSING, MissingType, warn_deprecated_alias
from .bundle import Bundle
from .hooks import HookContext
from .predicates import when
from .selectors import BaseSelector, func, label

SWEEP_NAME = "sweep"


def sweep(
    model: nn.Module,
    x: Any,
    at: str | BaseSelector | Callable[[Any], bool] | MissingType = MISSING,
    values: Iterable[Any] | MissingType = MISSING,
    *,
    param: str | BaseSelector | Callable[[Any], bool] | MissingType = MISSING,
    input_kwargs: dict[Any, Any] | None = None,
    names: Sequence[str] | None = None,
    **trace_kwargs: Any,
) -> Bundle:
    """Capture one intervened trace per swept replacement value.

    Parameters
    ----------
    model:
        PyTorch module to capture.
    x:
        Positional input argument or argument container passed to ``tl.trace``.
    at:
        Intervention site target. Strings match either an exact TorchLens label
        or a function name; selectors and predicate callables are used directly.
        (Formerly named ``param``; ``param=`` is accepted as a deprecated alias.)
    values:
        Replacement values to sweep at ``at``.
    param:
        Deprecated alias for ``at``. Do not pass both.
    input_kwargs:
        Optional keyword inputs passed to ``model.forward``.
    names:
        Optional Bundle member names. When omitted, names are derived from
        ``SWEEP_NAME`` and the value index.
    **trace_kwargs:
        Additional keyword arguments forwarded to ``tl.trace``.

    Returns
    -------
    Bundle
        Bundle containing one trace per swept value.

    Raises
    ------
    ValueError
        If no values are provided, names do not match values, or an explicit
        ``intervene`` argument is supplied.
    TypeError
        If ``at`` cannot be used as a capture-time intervention predicate.
    """

    # Resolve deprecated `param=` kwarg alias
    if param is not MISSING and at is not MISSING:
        raise TypeError("sweep() received both `at` and `param`; `param` is deprecated, use `at`.")
    if param is not MISSING:
        warn_deprecated_alias("param", "at")
        resolved_at = param
    elif at is MISSING:
        raise TypeError("sweep() requires the `at` argument (site target).")
    else:
        resolved_at = at

    # Resolve positional `values` (may be MISSING if it was skipped when param was keyword-only)
    if values is MISSING:
        raise TypeError("sweep() requires the `values` argument.")

    if "intervene" in trace_kwargs:
        raise ValueError(f"{SWEEP_NAME} owns intervene= and cannot combine another predicate.")

    # At this point both resolved_at and values are fully resolved (not MISSING).
    from typing import cast, Iterable as _Iterable

    resolved_at_typed = cast("str | BaseSelector | Callable[[Any], bool]", resolved_at)
    resolved_values = cast("_Iterable[Any]", values)

    swept_values = list(resolved_values)
    if not swept_values:
        raise ValueError(f"{SWEEP_NAME} requires at least one value.")
    if names is not None and len(names) != len(swept_values):
        raise ValueError("names length must match values length.")

    site = _coerce_sweep_site(resolved_at_typed)
    member_names = list(names) if names is not None else _default_member_names(len(swept_values))
    traces = {}
    from ..user_funcs import trace as _trace

    for member_name, value in zip(member_names, swept_values):
        traces[member_name] = _trace(
            model,
            x,
            input_kwargs=input_kwargs,
            intervene=when(site, _replacement_hook(value)),
            **trace_kwargs,
        )
    return Bundle(traces)


def _coerce_sweep_site(param: str | BaseSelector | Callable[[Any], bool]) -> Callable[[Any], bool]:
    """Normalize a sweep site target to a capture-time predicate.

    Parameters
    ----------
    param:
        String, selector, or predicate target.

    Returns
    -------
    Callable[[Any], bool]
        Predicate suitable for ``tl.when``.

    Raises
    ------
    TypeError
        If ``param`` is not a supported target.
    """

    if isinstance(param, str):
        return label(param) | func(param)
    if callable(param):
        return param
    raise TypeError("param must be a string, selector, or predicate callable.")


def _replacement_hook(value: Any) -> Callable[..., torch.Tensor]:
    """Create a hook that replaces an out with one swept value.

    Parameters
    ----------
    value:
        Scalar or tensor replacement value.

    Returns
    -------
    Callable[..., torch.Tensor]
        Runtime hook callable.
    """

    def _hook(out: torch.Tensor, *, hook: HookContext) -> torch.Tensor:
        """Replace ``out`` with ``value`` aligned to ``out`` metadata."""

        del hook
        replacement = _value_to_tensor(value, out)
        if replacement.shape == torch.Size([]):
            return torch.zeros_like(out) + replacement
        if replacement.shape == out.shape:
            return replacement.clone()
        try:
            return replacement.expand_as(out).clone()
        except RuntimeError:
            return replacement

    return _hook


def _value_to_tensor(value: Any, out: torch.Tensor) -> torch.Tensor:
    """Convert one swept value to an out-compatible tensor.

    Parameters
    ----------
    value:
        Scalar or tensor replacement value.
    out:
        Captured output tensor whose dtype and device should be matched.

    Returns
    -------
    torch.Tensor
        Replacement tensor on the same device and dtype as ``out``.
    """

    if isinstance(value, torch.Tensor):
        return value.to(device=out.device, dtype=out.dtype)
    return torch.as_tensor(value, device=out.device, dtype=out.dtype)


def _default_member_names(count: int) -> list[str]:
    """Return default Bundle member names for a sweep.

    Parameters
    ----------
    count:
        Number of swept values.

    Returns
    -------
    list[str]
        Stable member names.
    """

    return [f"{SWEEP_NAME}_{index}" for index in range(count)]


__all__ = ["SWEEP_NAME", "sweep"]
