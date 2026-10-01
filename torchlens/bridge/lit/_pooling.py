"""Explicit, mask-aware pooling for LIT ``Embeddings`` fields.

Pooling is explicit and mask-aware (LIT-panel memo D10): never batch-zero,
never PAD positions averaged in, never silent flattening of head/spatial/
sequence axes. Named strategies are ``mean_masked`` / ``first_token`` /
``last_unmasked`` plus a validated callable escape hatch; nontensor, missing,
and unsupported-rank payloads refuse with remedies. Strategy names are
[UI-SPRINT] placeholders (memo section 11 item 5).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from . import _refusals

PoolingFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
PoolingSpec = str | PoolingFn

POOLING_STRATEGIES: tuple[str, ...] = ("mean_masked", "first_token", "last_unmasked")


def validate_pooling(pooling: Any) -> PoolingSpec:
    """Validate the ``pooling=`` argument at construction time.

    Parameters
    ----------
    pooling:
        A named strategy or a callable ``(tensor, attention_mask) -> [B, F]``.

    Returns
    -------
    PoolingSpec
        The validated strategy.

    Raises
    ------
    torchlens._errors.InvalidArgumentError
        Code ``lit_pooling_invalid`` for unknown names or non-callables.
    """

    if callable(pooling):
        return pooling
    if isinstance(pooling, str) and pooling in POOLING_STRATEGIES:
        return pooling
    _refusals.refuse_pooling_invalid(pooling, POOLING_STRATEGIES)


def pool(
    field_name: str,
    value: Any,
    attention_mask: torch.Tensor,
    pooling: PoolingSpec,
) -> torch.Tensor:
    """Pool one site payload to ``[batch, features]`` under the mask contract.

    Rank contract: rank-2 ``[B, F]`` passes through; rank-3 ``[B, T, H]`` pools
    over the token axis mask-aware; rank-4 ``[B, C, H, W]`` (conv maps) pools
    by spatial mean, explicitly disclosed in the docs. Anything else refuses.

    Parameters
    ----------
    field_name:
        The LIT field being served (for teaching messages).
    value:
        The site payload read from the resolved op.
    attention_mask:
        ``[B, T]`` mask of real (non-PAD) token positions.
    pooling:
        A validated pooling strategy.

    Returns
    -------
    torch.Tensor
        Pooled ``[batch, features]`` tensor.

    Raises
    ------
    torchlens._errors.RecordBindingError
        Code ``lit_pooling_unsupported`` for payloads outside the contract.
    """

    if not isinstance(value, torch.Tensor):
        _refusals.refuse_pooling_unsupported(
            field_name, f"payload is {type(value).__name__}, not a tensor"
        )
    if callable(pooling) and not isinstance(pooling, str):
        return _validated_callable_result(field_name, pooling, value, attention_mask)
    if value.ndim == 2:
        return value
    if value.ndim == 3:
        return _pool_tokens(field_name, value, attention_mask, str(pooling))
    if value.ndim == 4:
        return value.mean(dim=(2, 3))
    _refusals.refuse_pooling_unsupported(
        field_name, f"rank-{value.ndim} payload of shape {tuple(value.shape)}"
    )


def _pool_tokens(
    field_name: str,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    strategy: str,
) -> torch.Tensor:
    """Pool a ``[B, T, H]`` payload over tokens, mask-aware.

    Parameters
    ----------
    field_name:
        The LIT field being served.
    value:
        Token-major payload.
    attention_mask:
        ``[B, T]`` real-token mask.
    strategy:
        A named strategy from ``POOLING_STRATEGIES``.

    Returns
    -------
    torch.Tensor
        Pooled ``[B, H]`` tensor.
    """

    if value.shape[:2] != attention_mask.shape:
        _refusals.refuse_pooling_unsupported(
            field_name,
            f"payload shape {tuple(value.shape)} does not lead with the request's "
            f"[batch, tokens] geometry {tuple(attention_mask.shape)}",
        )
    mask = attention_mask.to(dtype=value.dtype, device=value.device)
    if strategy == "mean_masked":
        counts = mask.sum(dim=1).clamp(min=1.0).unsqueeze(-1)
        return (value * mask.unsqueeze(-1)).sum(dim=1) / counts
    if strategy == "first_token":
        first = attention_mask.to(torch.int64).argmax(dim=1)
        return value[torch.arange(value.shape[0], device=value.device), first]
    last = attention_mask.shape[1] - 1 - attention_mask.flip(1).to(torch.int64).argmax(dim=1)
    return value[torch.arange(value.shape[0], device=value.device), last]


def _validated_callable_result(
    field_name: str,
    pooling: PoolingFn,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Run a user pooling callable and validate its result geometry.

    Parameters
    ----------
    field_name:
        The LIT field being served.
    pooling:
        The user callable.
    value:
        The site payload.
    attention_mask:
        ``[B, T]`` real-token mask.

    Returns
    -------
    torch.Tensor
        The callable's ``[batch, features]`` result.
    """

    result = pooling(value, attention_mask)
    if (
        not isinstance(result, torch.Tensor)
        or result.ndim != 2
        or result.shape[0] != attention_mask.shape[0]
    ):
        _refusals.refuse_pooling_unsupported(
            field_name,
            "custom pooling callable must return a [batch, features] tensor; got "
            f"{type(result).__name__}"
            + (f" of shape {tuple(result.shape)}" if isinstance(result, torch.Tensor) else ""),
        )
    return result
