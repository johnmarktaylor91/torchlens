"""Storage-mutation probes for live intervention hook outputs."""

from __future__ import annotations

import warnings
from contextlib import nullcontext
from typing import Any

import torch

from .._state import pause_logging


def _tensor_version(value: Any) -> int | None:
    """Return a tensor's in-place mutation counter, or ``None`` when unreadable.

    ``Tensor._version`` is the cheap autograd version counter; inference-mode
    tensors (no counter) and exotic subclasses read as ``None``, which callers
    treat as "no in-place evidence" rather than a refusal.
    """

    if not isinstance(value, torch.Tensor):
        return None
    try:
        return int(value._version)
    except Exception:
        return None


def _tuple_versions(values: tuple[torch.Tensor | None, ...]) -> tuple[int | None, ...]:
    """Version counters for one grad tuple, ``None`` per non-tensor slot."""

    return tuple(_tensor_version(value) for value in values)


_CONTENT_PROBE_SAMPLES = 8
"""Bounded per-fire sample width for the storage-alias mutation probe."""


def _tensor_content_probe(value: Any) -> tuple[Any, ...] | None:
    """Bounded strided content sample witnessing storage-alias mutations.

    ``Tensor._version`` misses mutations routed through a DIFFERENT impl over
    the same storage: ``.data`` mints a storage-sharing alias with a FRESH
    version counter, so ``out.data.mul_(0); return out`` changed execution
    with the counter witness reading "no mutation" (the incomplete half of
    2289e56c). Identity returns therefore pair the counter with this O(1)
    sample -- numel plus up to :data:`_CONTENT_PROBE_SAMPLES` evenly strided
    elements. Wholesale in-place edits (zeroing, scaling) are caught; a
    mutation confined to unsampled elements remains a documented residual,
    with replay validation the fail-closed authority. ``None`` (non-tensor /
    unreadable / exotic subclass) reads as "no evidence", never a refusal.
    """

    if not isinstance(value, torch.Tensor):
        return None
    try:
        with pause_logging():
            numel = int(value.numel())
            if numel == 0:
                return (0, ())
            flat = value.detach().reshape(-1)
            count = min(_CONTENT_PROBE_SAMPLES, numel)
            step = max(1, numel // count)
            sample = flat[::step][:count].tolist()
        return (numel, tuple(sample))
    except Exception:
        return None


def _content_probe_mutated(before: tuple[Any, ...] | None, after: tuple[Any, ...] | None) -> bool:
    """NaN-aware inequality between two content probes (missing = no evidence)."""

    if before is None or after is None:
        return False
    numel_before, sample_before = before
    numel_after, sample_after = after
    if numel_before != numel_after or len(sample_before) != len(sample_after):
        return True
    for left, right in zip(sample_before, sample_after, strict=True):
        if left != right and not (left != left and right != right):  # NaN == NaN here
            return True
    return False


def _tuple_content_probes(
    values: tuple[torch.Tensor | None, ...],
) -> tuple[tuple[Any, ...] | None, ...]:
    """Content probes for one grad tuple, ``None`` per non-tensor slot."""

    return tuple(_tensor_content_probe(value) for value in values)


def _tuple_probes_mutated(
    before: tuple[tuple[Any, ...] | None, ...],
    after: tuple[tuple[Any, ...] | None, ...],
) -> bool:
    """Whether any grad-tuple slot's content probe changed (NaN-aware)."""

    if len(before) != len(after):
        return True
    return any(
        _content_probe_mutated(left, right) for left, right in zip(before, after, strict=True)
    )


def _apply_inplace_replacement_to_mutated_storage(
    result: torch.Tensor,
    *,
    current_out: torch.Tensor,
    site: Any,
    call_args: tuple[Any, ...],
    call_kwargs: dict[str, Any] | None,
) -> tuple[torch.Tensor, bool]:
    """Make a live output replacement effective for discarded in-place returns.

    Parameters
    ----------
    result:
        Validated hook result.
    current_out:
        Output tensor supplied to the hook.
    site:
        Capture-time site metadata.
    call_args:
        Live positional inputs after the operation executed.
    call_kwargs:
        Live keyword inputs after the operation executed.

    Returns
    -------
    tuple[torch.Tensor, bool]
        Effective output and whether execution accepted the replacement.
    """

    if result is current_out:
        return result, False
    if not bool(getattr(site, "is_inplace", False)):
        return result, True

    func_name = str(getattr(site, "func_name", ""))
    destination: torch.Tensor | None = None
    if _is_inplace_style_func_name(func_name) and call_args:
        candidate = call_args[0]
        if isinstance(candidate, torch.Tensor):
            destination = candidate
    elif call_kwargs is not None:
        candidate = call_kwargs.get("out")
        if isinstance(candidate, torch.Tensor):
            destination = candidate

    safe_to_copy = destination is not None and (
        tuple(destination.shape) == tuple(result.shape)
        and destination.dtype == result.dtype
        and destination.device == result.device
    )
    if safe_to_copy:
        assert destination is not None
        # A Parameter receiver is only legally written untracked (eager runs
        # its in-place op under ``no_grad`` or on a frozen Parameter), so the
        # replacement write is untracked too.
        grad_ctx = torch.no_grad() if isinstance(destination, torch.nn.Parameter) else nullcontext()
        try:
            with pause_logging(), grad_ctx:
                destination.copy_(result)
        except Exception:
            safe_to_copy = False
    if safe_to_copy:
        return result, True

    warnings.warn(
        f"Refused an output replacement for in-place operation {func_name!r} because "
        "TorchLens could not safely copy it into the mutated tensor storage; execution "
        "continues with the original output and the fire record is marked replaced=False.",
        UserWarning,
        stacklevel=3,
    )
    return current_out, False


def _is_inplace_style_func_name(func_name: str) -> bool:
    """Return whether ``func_name`` follows TorchLens' in-place op naming rules.

    Parameters
    ----------
    func_name:
        Decorated torch function name.

    Returns
    -------
    bool
        Whether the function is a candidate in-place op before execution.
    """

    return (
        (func_name.endswith("_") and not func_name.startswith("__"))
        or func_name.startswith("__i")
        or func_name in {"__setitem__", "__delitem__"}
    )
