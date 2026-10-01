"""The private concrete-input capture primitive (quickstart B5).

Layer 2 of the four-layer split (memo D1): this primitive accepts ONLY a
concrete :class:`~torchlens.quickstart._plan.InputPlan` -- it can never
invoke zero-input resolution, synthesis, or inference, so the
resolver -> inference -> capture import cycle is impossible by construction.
The inference search and every quickstart verb capture THROUGH this door.

The pinned execution policy (memo D8, surface-specific): summary and render
pin eval + no-grad, restore every training flag, host/device RNG, and
normalization buffers, verify with a state-dict hash, and disclose -- with
the promise SCOPED to what is actually restored (arbitrary Python side
effects are outside it). ``tl.trace`` keeps the caller's mode and does not
route through the pinned path.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm

from ._plan import InputPlan

__tl_layer__ = "L2"


@dataclass(frozen=True)
class ExecutionReceipt:
    """Disclosure of what the pinned capture policy did and restored.

    Attributes
    ----------
    policy:
        ``"caller_mode"`` or ``"eval_no_grad_restored"``.
    flags_restored:
        Number of modules whose ``training`` flag was saved and restored.
    rng_restored:
        Whether host (and any live CUDA) RNG state was restored.
    norm_buffers_restored:
        Number of normalization buffers restored from their snapshots.
    state_verified:
        ``True`` when the post-capture state-dict hash matched the
        pre-capture hash; ``None`` when verification was not requested.
    state_hash:
        The (pre-capture) state-dict blake2b hex digest, when computed.
    """

    policy: str
    flags_restored: int = 0
    rng_restored: bool = False
    norm_buffers_restored: int = 0
    state_verified: bool | None = None
    state_hash: str | None = None


def _tensor_digest(tensor: torch.Tensor, hasher: hashlib.blake2b) -> None:
    """Fold one tensor's exact bytes into ``hasher`` (dtype-agnostic)."""

    concrete = tensor.detach().cpu().contiguous()
    hasher.update(str(concrete.dtype).encode())
    hasher.update(str(tuple(concrete.shape)).encode())
    if concrete.numel() == 0:
        return
    # bytes(UntypedStorage) falls back to per-BYTE __getitem__ iteration
    # (minutes for one resnet18 hash); reinterpret the tensor's own bytes as
    # uint8 and hand hashlib ONE buffer instead. This also scopes the digest
    # to exactly this tensor's numel*itemsize bytes -- never the whole
    # (possibly shared) backing storage.
    hasher.update(concrete.flatten().view(torch.uint8).numpy().tobytes())


def state_dict_hash(model: nn.Module) -> str:
    """Return a blake2b digest over the model's full ``state_dict`` bytes.

    Names, shapes, dtypes, and exact values all enter the digest, so any
    parameter or persistent-buffer mutation across a pinned capture is
    detectable. This is a VALUE hash -- deliberately not the cheap
    name/shape fingerprint, whose collisions the panel measured across
    checkpoints (memo D9).
    """

    hasher = hashlib.blake2b(digest_size=16)
    for name, tensor in model.state_dict().items():
        hasher.update(name.encode())
        _tensor_digest(tensor, hasher)
    return hasher.hexdigest()


def _norm_buffer_snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    """Clone every normalization-module buffer (running stats + counters)."""

    snapshot: dict[str, torch.Tensor] = {}
    for module_name, module in model.named_modules():
        if not isinstance(module, (_BatchNorm, _InstanceNorm)):
            continue
        for buffer_name, buffer in module.named_buffers(recurse=False):
            if buffer is not None:
                snapshot[f"{module_name}.{buffer_name}" if module_name else buffer_name] = (
                    buffer.detach().clone()
                )
    return snapshot


def _restore_norm_buffers(model: nn.Module, snapshot: dict[str, torch.Tensor]) -> int:
    """Copy snapshotted normalization buffers back in place; return the count."""

    if not snapshot:
        return 0
    buffers = dict(model.named_buffers())
    restored = 0
    for name, saved in snapshot.items():
        live = buffers.get(name)
        if live is not None and live.shape == saved.shape:
            with torch.no_grad():
                live.copy_(saved)
            restored += 1
    return restored


def _trace_input_spelling(plan: InputPlan) -> Any:
    """Map a plan's positional pack onto ``tl.trace``'s input_args convention."""

    if len(plan.input_args) == 1:
        return plan.input_args[0]
    return list(plan.input_args)


def capture_concrete(  # noqa: PLR0913 -- the ONE capture door: a concrete plan plus the four named execution-policy dials (memo D8/D9)
    model: nn.Module,
    plan: InputPlan,
    *,
    pinned_eval: bool = False,
    metadata_only: bool = False,
    verify_state: bool = False,
    trace_kwargs: dict[str, Any] | None = None,
) -> tuple[Any, ExecutionReceipt]:
    """Run ONE capture from a concrete plan and return (trace, receipt).

    Parameters
    ----------
    model:
        The model to capture.
    plan:
        Concrete input plan. When the plan already carries a verified trace
        (rung-3 in-call reuse, memo D9), that trace is returned directly and
        NO new forward runs -- the capture counter stays at the inference
        search's own count.
    pinned_eval:
        Pin eval + no-grad, then restore training flags, RNG state, and
        normalization buffers (the summary/render policy, memo D8).
    metadata_only:
        Capture with ``save=None`` (no activation payloads).
    verify_state:
        Compute the state-dict value hash before and after and record the
        comparison on the receipt (pinned surfaces enable this).
    trace_kwargs:
        Extra keyword arguments forwarded verbatim to ``tl.trace``.

    Returns
    -------
    tuple
        ``(trace, ExecutionReceipt)``.
    """

    if plan.verified_trace is not None:
        return plan.verified_trace, ExecutionReceipt(policy="reused_verified_trace")

    from ..options import CaptureOptions
    from ..user_funcs import trace as _trace

    kwargs: dict[str, Any] = dict(trace_kwargs or {})
    if metadata_only and "save" not in kwargs:
        kwargs["save"] = None
    if plan.input_kwargs and "input_kwargs" not in kwargs:
        kwargs["input_kwargs"] = dict(plan.input_kwargs)
    if not pinned_eval:
        result = _trace(model, _trace_input_spelling(plan), **kwargs)
        return result, ExecutionReceipt(policy="caller_mode")

    kwargs.setdefault("capture", CaptureOptions(inference_only=True))
    state_hash = state_dict_hash(model) if verify_state else None
    training_flags = [(module, module.training) for module in model.modules()]
    cpu_rng = torch.get_rng_state()
    from ..utils.rng import _snapshot_cuda_rng_states

    cuda_rng = _snapshot_cuda_rng_states() or None
    norm_snapshot = _norm_buffer_snapshot(model)
    try:
        model.eval()
        result = _trace(model, _trace_input_spelling(plan), **kwargs)
    finally:
        for module, flag in training_flags:
            module.training = flag
        restored = _restore_norm_buffers(model, norm_snapshot)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
    verified: bool | None = None
    if verify_state and state_hash is not None:
        verified = state_dict_hash(model) == state_hash
    receipt = ExecutionReceipt(
        policy="eval_no_grad_restored",
        flags_restored=len(training_flags),
        rng_restored=True,
        norm_buffers_restored=restored,
        state_verified=verified,
        state_hash=state_hash,
    )
    return result, receipt
