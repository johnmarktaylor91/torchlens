"""Pool presets: axis-explicit, mask-aware, refuse ambiguity (extract D10).

Presets (spellings DOCUMENTED-UNSTABLE): ``flatten``, ``spatial_mean`` /
``spatial_max`` (4D feature maps), ``token_mean`` / ``token_max`` /
``token_sum`` (3D token sequences, MASK-AWARE), ``cls`` (first token), and
``last_token`` — the last VALID token ("last" means last non-pad, so the
mask is required). Global or per-key.

Ambiguous axes, missing required masks, and deliberate unmasked pooling
over a pad-bearing mask refuse typed unless an explicit recorded override
applies — the memo number is the measured 20.66% error from mean-pooling
over pads. Pool runs in CAPTURED dtype on RAW axes with the mask, BEFORE
postprocess transforms, and is the documented raggedness-killer for text.
Pool is in the resume signature.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from .._errors import InvalidArgumentError

__tl_layer__ = "L5"

__all__ = ["POOL_PRESETS", "apply_pool", "resolve_pool_policy"]

#: Closed preset vocabulary (extract D10).
POOL_PRESETS: tuple[str, ...] = (
    "flatten",
    "spatial_mean",
    "spatial_max",
    "token_mean",
    "token_max",
    "token_sum",
    "cls",
    "last_token",
)

#: Presets that read the attention mask.
_MASK_AWARE = frozenset({"token_mean", "token_max", "token_sum", "last_token"})


def _validate_spec(spec: Any) -> dict[str, Any]:
    """Validate one pool spec value into its canonical dict form.

    Parameters
    ----------
    spec:
        Preset string, or ``{"preset": ..., "unmasked": bool}`` — the
        explicit recorded unmasked override.

    Returns
    -------
    dict[str, Any]
        ``{"preset": str, "unmasked": bool}``.

    Raises
    ------
    torchlens.errors.InvalidArgumentError
        ``extraction_pool_invalid`` outside the closed vocabulary.
    """

    if isinstance(spec, str):
        spec_dict: dict[str, Any] = {"preset": spec, "unmasked": False}
    elif isinstance(spec, Mapping):
        unknown = set(spec) - {"preset", "unmasked"}
        if unknown:
            raise InvalidArgumentError(
                f"pool spec {dict(spec)!r} carries unknown fields {sorted(unknown)}.",
                code="extraction_pool_invalid",
                remedy='use {"preset": <name>, "unmasked": bool}',
                unknown_fields=sorted(unknown),
            )
        spec_dict = {"preset": spec.get("preset"), "unmasked": bool(spec.get("unmasked", False))}
    else:
        raise InvalidArgumentError(
            f"pool= value {spec!r} has type {type(spec).__name__}; pooling is "
            "a closed preset vocabulary, never a callable (use transform= "
            "for custom reductions).",
            code="extraction_pool_invalid",
            remedy=f"pass one of {POOL_PRESETS} or a per-key mapping of them",
            value=repr(spec),
        )
    if spec_dict["preset"] not in POOL_PRESETS:
        raise InvalidArgumentError(
            f"pool preset {spec_dict['preset']!r} is not in the closed vocabulary {POOL_PRESETS}.",
            code="extraction_pool_invalid",
            remedy=f"pass one of {POOL_PRESETS}",
            preset=repr(spec_dict["preset"]),
        )
    return spec_dict


def resolve_pool_policy(pool: Any) -> tuple[dict[str, Any] | None, Any]:
    """Resolve the ``pool=`` kwarg into per-key specs plus the signature record.

    Parameters
    ----------
    pool:
        ``None`` | preset string | ``{"preset": ..., "unmasked": ...}`` |
        per-key ``Mapping[str, spec]``.

    Returns
    -------
    tuple[dict | None, Any]
        ``(policy, record)``: the policy is ``{"global": spec}`` or
        ``{"per_key": {key: spec}}`` (``None`` when pooling is off), and
        the record is its JSON-portable signature form.
    """

    if pool is None:
        return None, None
    if isinstance(pool, Mapping) and "preset" not in pool:
        per_key = {str(key): _validate_spec(value) for key, value in pool.items()}
        return {"per_key": per_key}, {"per_key": per_key}
    spec = _validate_spec(pool)
    return {"global": spec}, {"global": spec}


def _spec_for_key(policy: Mapping[str, Any] | None, key: str) -> dict[str, Any] | None:
    """Look up the pool spec applying to one output key.

    Parameters
    ----------
    policy:
        Resolved pool policy.
    key:
        Output key.

    Returns
    -------
    dict | None
        The spec, or ``None`` when this key is unpooled.
    """

    if policy is None:
        return None
    if "global" in policy:
        return policy["global"]
    return policy.get("per_key", {}).get(key)


def _refuse_axes(key: str, preset: str, tensor: torch.Tensor, expectation: str) -> None:
    """Raise the axis-ambiguity refusal (extract D10).

    Parameters
    ----------
    key:
        Output key.
    preset:
        Requested preset.
    tensor:
        The captured tensor.
    expectation:
        Sentence naming the required geometry.
    """

    raise InvalidArgumentError(
        f"pool preset {preset!r} cannot apply to output key {key!r} with "
        f"captured shape {tuple(tensor.shape)}: {expectation} Guessing an "
        "axis would silently pool the wrong dimension.",
        code="extraction_pool_axes_ambiguous",
        remedy=(
            "pick the preset matching this site's geometry (spatial_* for "
            "4D maps, token_*/cls/last_token for 3D sequences, flatten for "
            "anything), or pool this key differently via a per-key mapping"
        ),
        key=key,
        preset=preset,
        shape=list(tensor.shape),
    )


def _masked_token_pool(
    key: str, preset: str, tensor: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Apply one mask-aware token pool over axis 1.

    Parameters
    ----------
    key:
        Output key (refusal text).
    preset:
        ``token_mean`` / ``token_max`` / ``token_sum`` / ``last_token``.
    tensor:
        ``[batch, tokens, ...]`` captured tensor.
    mask:
        ``[batch, tokens]`` attention mask (nonzero = valid).

    Returns
    -------
    torch.Tensor
        Pooled ``[batch, ...]`` tensor in the captured dtype.
    """

    valid = (mask != 0).to(tensor.device)
    expanded = valid.reshape(valid.shape + (1,) * (tensor.ndim - 2))
    if preset == "token_sum":
        return (tensor * expanded.to(tensor.dtype)).sum(dim=1)
    if preset == "token_mean":
        counts = valid.sum(dim=1).clamp(min=1).reshape((-1,) + (1,) * (tensor.ndim - 2))
        return (tensor * expanded.to(tensor.dtype)).sum(dim=1) / counts.to(tensor.dtype)
    if preset == "token_max":
        if not tensor.is_floating_point():
            _refuse_axes(key, preset, tensor, "token_max needs a floating tensor.")
        lowest = torch.finfo(tensor.dtype).min
        filled = tensor.masked_fill(~expanded, lowest)
        return filled.max(dim=1).values
    # last_token: index of each row's last valid token.
    last = valid.long().cumsum(dim=1).argmax(dim=1)
    rows = torch.arange(tensor.shape[0], device=tensor.device)
    return tensor[rows, last]


def apply_pool(
    key: str,
    tensor: torch.Tensor,
    policy: Mapping[str, Any] | None,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    """Apply the resolved pool policy to one captured tensor (extract D10).

    Runs in the CAPTURED dtype on RAW axes, before postprocess transforms.

    Parameters
    ----------
    key:
        Output key.
    tensor:
        Captured batch tensor, stimulus axis leading.
    policy:
        Resolved pool policy from :func:`resolve_pool_policy`.
    mask:
        The batch's attention mask, when the collator supplied one.

    Returns
    -------
    torch.Tensor
        The pooled tensor (or the input unchanged when this key is
        unpooled).

    Raises
    ------
    torchlens.errors.InvalidArgumentError
        ``extraction_pool_axes_ambiguous`` on geometry the preset cannot
        prove; ``extraction_pool_mask_required`` when a mask-aware preset
        has no mask (or a mismatched one);
        ``extraction_pool_dtype_unsupported`` for fp8 pool compute.
    """

    spec = _spec_for_key(policy, key)
    if spec is None:
        return tensor
    preset = spec["preset"]
    if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise InvalidArgumentError(
            f"pool preset {preset!r} cannot run in captured dtype "
            f"{tensor.dtype} (fp8 reduction kernels are unsupported); pool "
            "runs in the captured dtype by contract (D10), so the compute "
            "refuses rather than silently upcasting.",
            code="extraction_pool_dtype_unsupported",
            remedy="capture in a wider dtype and use dtype= to store fp8 after pooling",
            key=key,
            dtype=str(tensor.dtype),
        )
    dense = _pool_dense(key, preset, tensor)
    if dense is not None:
        return dense
    return _pool_token(key, preset, tensor, mask, bool(spec.get("unmasked", False)))


def _pool_dense(key: str, preset: str, tensor: torch.Tensor) -> torch.Tensor | None:
    """Apply the geometry-checked mask-free presets (flatten/spatial/cls).

    Parameters
    ----------
    key:
        Output key.
    preset:
        Validated preset name.
    tensor:
        Captured batch tensor.

    Returns
    -------
    torch.Tensor | None
        The pooled tensor, or ``None`` when the preset is a token preset
        (handled by the mask-aware path after the 3D geometry gate).
    """

    if preset == "flatten":
        if tensor.ndim < 2:
            _refuse_axes(key, preset, tensor, "flatten needs at least a batch axis plus one more.")
        return tensor.reshape(tensor.shape[0], -1)
    if preset in ("spatial_mean", "spatial_max"):
        if tensor.ndim != 4:
            _refuse_axes(
                key,
                preset,
                tensor,
                "spatial pooling is defined for 4D [batch, channel, height, "
                "width] feature maps only (3D sequences use token_*).",
            )
        return tensor.mean(dim=(2, 3)) if preset == "spatial_mean" else tensor.amax(dim=(2, 3))
    if tensor.ndim != 3:
        _refuse_axes(
            key,
            preset,
            tensor,
            "token pooling is defined for 3D [batch, tokens, features] "
            "sequences only (4D maps use spatial_*).",
        )
    if preset == "cls":
        return tensor[:, 0]
    return None


def _pool_token(
    key: str,
    preset: str,
    tensor: torch.Tensor,
    mask: torch.Tensor | None,
    unmasked: bool,
) -> torch.Tensor:
    """Apply the mask-aware token presets with the D10 mask rules.

    Parameters
    ----------
    key:
        Output key.
    preset:
        Validated token preset.
    tensor:
        3D ``[batch, tokens, features]`` tensor.
    mask:
        The batch's attention mask, when supplied.
    unmasked:
        Whether the explicit recorded unmasked override is active.

    Returns
    -------
    torch.Tensor
        The pooled tensor.
    """

    if mask is not None and (mask.shape[0] != tensor.shape[0] or mask.shape[1] != tensor.shape[1]):
        raise InvalidArgumentError(
            f"pool preset {preset!r} for output key {key!r}: the batch mask "
            f"has shape {tuple(mask.shape)} but the captured tensor is "
            f"{tuple(tensor.shape)}; the mask cannot be proven to describe "
            "this site's token axis.",
            code="extraction_pool_mask_required",
            remedy=(
                "pool a site whose token axis matches the input mask, or "
                'use the explicit {"preset": ..., "unmasked": true} override'
            ),
            key=key,
            mask_shape=list(mask.shape),
            tensor_shape=list(tensor.shape),
        )
    if not unmasked:
        if mask is None:
            raise InvalidArgumentError(
                f"pool preset {preset!r} for output key {key!r} is "
                "mask-aware and this batch carries no attention mask; "
                "pooling over unmasked pad positions was measured 20.66% "
                "wrong on real text.",
                code="extraction_pool_mask_required",
                remedy=(
                    "collate with an attention mask (the HF collate helper "
                    "supplies one), or record the deliberate override "
                    f'pool={{"preset": "{preset}", "unmasked": true}}'
                ),
                key=key,
                preset=preset,
            )
        return _masked_token_pool(key, preset, tensor, mask)
    # From here the EXPLICIT unmasked override is active; with real pads
    # present that is only legal because it was recorded in the signature.
    if preset == "last_token":
        raise InvalidArgumentError(
            f"pool preset 'last_token' for output key {key!r} cannot run "
            "unmasked: 'last' means last VALID token, which only the mask "
            "defines (an unmasked last row slot is a pad for every "
            "right-padded row).",
            code="extraction_pool_mask_required",
            remedy="drop the unmasked override for last_token pooling",
            key=key,
        )
    if preset == "token_sum":
        return tensor.sum(dim=1)
    if preset == "token_mean":
        return tensor.mean(dim=1)
    return tensor.amax(dim=1)
