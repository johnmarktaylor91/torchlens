"""Model-state and global-RNG guard shared by the facet patching helpers.

``_CounterfactualStateGuard`` snapshots the model's parameters, buffers,
gradients and the global RNG before a counterfactual rerun and restores them
afterwards; ``_teardown`` closes the guard after every trace cleanup even when
one of those cleanups raises.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from contextlib import ExitStack
from typing import Any

import torch
from torch import nn

from .._utils._cuda_probe import _is_cuda_initialized
from .._utils.rng import _snapshot_cuda_rng_states


class _CounterfactualStateGuard:
    """Snapshot and restore model state and global RNG around patching runs.

    Activation and attribution patching run several forward passes (a clean
    baseline, a corrupted baseline, and one per patched cell) on the SAME live
    model. Without resetting state between runs, mutable buffers (e.g.
    BatchNorm running statistics or any buffer written during forward) and the
    global RNG drift: every counterfactual after the first starts from a
    different model/RNG state than the clean baseline, so the reported effect is
    a silently wrong comparison, and the caller's model + global RNG are left
    mutated when the helper returns.

    Call :meth:`open` before the first run and :meth:`close` in a ``finally``.
    ``open`` snapshots the model's parameters/buffers and enters
    ``torch.random.fork_rng`` so the caller's global RNG is restored robustly on
    ``close`` no matter what the traced forwards do. Call :meth:`reset` before
    each counterfactual run to return the model and RNG to the captured baseline
    so every counterfactual is a true comparison. ``close`` restores the model
    state and releases the forked RNG.
    """

    def __init__(self, model: nn.Module) -> None:
        """Snapshot the model's parameters/buffers/grads and the global RNG state."""

        self._tensors: list[tuple[torch.Tensor, torch.Tensor]] = [
            (tensor, tensor.detach().clone()) for tensor in _stateful_tensors(model)
        ]
        self._grads: list[tuple[torch.Tensor, torch.Tensor | None]] = [
            (tensor, None if tensor.grad is None else tensor.grad.detach().clone())
            for tensor, _saved in self._tensors
        ]
        self._rng: dict[str, Any] = _snapshot_rng()
        self._fork: Any = None
        # Forward keyword inputs of the baselines; every patched rerun reuses them so the
        # counterfactual runs the same code path (set by ``patching._baseline_traces``).
        self.forward_kwargs: dict[Any, Any] | None = None

    def open(self) -> None:
        """Enter a forked-RNG scope so the caller's global RNG is preserved."""

        self._fork = torch.random.fork_rng(devices=_fork_rng_devices())
        self._fork.__enter__()

    def close(self) -> None:
        """Restore the model state and release the forked RNG scope."""

        try:
            self._restore_model()
        finally:
            fork, self._fork = self._fork, None
            if fork is not None:
                fork.__exit__(None, None, None)

    def reset(self) -> None:
        """Reset model state and global RNG to the captured baseline."""

        self._restore_model()
        _restore_rng(self._rng)

    def parameters_drifted(self) -> bool:
        """Return whether any parameter's value differs from the snapshot.

        Used after a capture that kept parameter arguments by reference: a
        forward that wrote a parameter (through ``param.data`` too, which no
        version counter on the parameter records) leaves aliases that no
        longer hold the call-time value.
        """

        return any(
            isinstance(tensor, nn.Parameter) and not torch.equal(tensor, saved)
            for tensor, saved in self._tensors
        )

    def _restore_model(self) -> None:
        """Restore every DRIFTED parameter/buffer value and grad in place.

        The copy is equality-gated on purpose: an unconditional ``copy_``
        bumps the autograd version counter of every parameter consumed by an
        earlier captured forward, so the later ``backward`` on the clean
        and corrupted baselines raised "modified by an inplace operation" on
        ANY model with parameters (attribution patching was unusable outside
        zero-parameter toys). A value-equal tensor needs no write and keeps
        its autograd graphs valid; a genuinely drifted tensor (a mutated
        buffer, a forward that writes a parameter) is still written back --
        comparison honesty beats preserving a graph that no longer matches
        the model state, and autograd's in-place tripwire then fires on a
        REAL divergence instead of on bookkeeping.

        Grad slots are restored to their snapshot (usually ``None``) so
        counterfactual runs start from the baseline grad state and the
        caller's model does not accumulate patching-run gradients. Grad
        assignment never touches version counters.
        """

        with torch.no_grad():
            for tensor, saved in self._tensors:
                if not torch.equal(tensor, saved):
                    tensor.copy_(saved)
        for tensor, saved_grad in self._grads:
            current = tensor.grad
            if saved_grad is None:
                if current is not None:
                    tensor.grad = None
            elif current is None or not torch.equal(current, saved_grad):
                tensor.grad = saved_grad.clone()


def _teardown(guard: _CounterfactualStateGuard, *logs: Any) -> None:
    """Run every teardown step even when an earlier one raises.

    The entry points used to tear down as bare sequential statements, so a
    raising ``Trace.cleanup()`` skipped ``guard.close()`` and left the USER's
    model parameters unrestored and the global RNG permanently advanced.
    ``ExitStack`` callbacks run last-registered-first, so the guard is
    registered first (closes LAST, after every log cleanup) and each
    ``logs``-order cleanup still runs even if an earlier one raises.
    """

    with ExitStack() as stack:
        stack.callback(guard.close)
        for log in reversed(logs):
            if log is not None:
                stack.callback(log.cleanup)


def _stateful_tensors(model: nn.Module) -> list[torch.Tensor]:
    """Return every distinct parameter and buffer tensor of a model.

    Uses object identity to include non-persistent buffers and to stage tied /
    double-registered tensors exactly once.
    """

    seen: set[int] = set()
    tensors: list[torch.Tensor] = []
    for _, tensor in itertools.chain(model.named_parameters(), model.named_buffers()):
        if id(tensor) in seen:
            continue
        seen.add(id(tensor))
        tensors.append(tensor)
    return tensors


def _fork_rng_devices() -> list[int]:
    """Return the CUDA device ordinals to fork RNG for (empty when CPU-only).

    Gated on ``is_initialized``, not ``is_available``: if this
    process never initialized CUDA, no CUDA generator fed any captured op,
    and forking RNG for every visible device would allocate the very CUDA
    contexts (~300-600 MB each) this path does not need -- the exact stale
    pre-fix gate ``utils/rng.py`` documents.
    """

    if _is_cuda_initialized() and torch.cuda.is_available():
        return list(range(torch.cuda.device_count()))
    return []


def _snapshot_rng() -> dict[str, Any]:
    """Capture the global CPU (and CUDA, when live) RNG state."""

    snapshot: dict[str, Any] = {"cpu": torch.default_generator.get_state()}
    # Initialized-CUDA-only, latch-guarded: returns [] on a
    # CUDA-less or never-initialized process without touching the driver.
    cuda_states = _snapshot_cuda_rng_states()
    if cuda_states:
        snapshot["cuda"] = cuda_states
    return snapshot


def _restore_rng(snapshot: Mapping[str, Any]) -> None:
    """Restore the global CPU (and CUDA, when available) RNG state."""

    torch.set_rng_state(snapshot["cpu"])
    cuda_states = snapshot.get("cuda")
    if cuda_states is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_states)
