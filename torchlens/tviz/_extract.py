"""Trace-side extractors: attention views with mask provenance (memo D5/D6).

``attention_view`` / ``attention_views`` read the semantic ``pattern`` facet
(served head-major ``[batch, head, destination, source]``) into the canonical
:class:`AttentionView` record. Provenance is derived from the facet recipe:
an eager captured softmax output is ``"captured"``; a fused-kernel
reconstruction is ``"reconstructed"`` (read-only, disclosed on the artifact).

Mask provenance is the D6 three-source hierarchy and NOTHING else: recorded
SDPA call arguments first, the captured eager additive-mask operand second,
explicit user metadata third; a view with none of these renders unmarked
with the exact "zero and masked positions are not distinguished" wording.
Deriving a mask from zero-valued pattern cells is banned (it passes the
canonical causal row perfectly), and the additive fill is compared against
``finfo(dtype).min`` -- NOT ``-inf`` -- because that is what real HF models
put there (memo finding f).

All spellings are DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

from typing import Any

import torch

from ..errors._base import TorchLensError
from ._errors import refuse
from ._records import AttentionView, GqaInfo, MaskInfo, TokenAxis

__all__ = ["attention_view", "attention_views"]


def _has_facet(module: Any, name: str) -> bool:
    """Return whether a module exposes an available facet ``name``.

    Multi-call modules (reused within one forward) raise on module-level
    facet access; they read as facet-less here -- per-call attention views
    are wave-2 episode territory.
    """

    try:
        view = module.facets
    except TorchLensError:
        return False
    if view is None:
        return False
    if hasattr(view, "has"):
        return bool(view.has(name))
    return name in view


def _modules_with_pattern(trace: Any) -> list[str]:
    """Return module addresses exposing an available ``pattern`` facet."""

    return [
        str(module.address)
        for module in getattr(trace, "modules", ())
        if getattr(module, "address", None) != "self" and _has_facet(module, "pattern")
    ]


def _pattern_tensor(trace: Any, address: str, batch_index: int) -> tuple[torch.Tensor, str]:
    """Read one module's pattern facet; return ``([head, dst, src], provenance)``."""

    module = trace.modules[address]
    facet = module.facets["pattern"]
    spec = getattr(facet, "spec", None)
    value = facet.value
    if not isinstance(value, torch.Tensor):
        refuse(
            code="tv_payload_missing",
            message=f"The pattern facet at {address!r} did not resolve to a tensor "
            f"({type(value).__name__}); its payload was not captured.",
            remedy="recapture with the attention ops saved (the default exhaustive save "
            "suffices; for fused kernels pass reconstruction_ready=True)",
            address=address,
        )
    if value.ndim != 4:
        refuse(
            code="tv_record_invalid",
            message=f"Expected a [batch, head, destination, source] pattern at {address!r}; "
            f"got rank {value.ndim}.",
            remedy="this facet layout is unexpected -- report the architecture",
            address=address,
        )
    recipe = getattr(spec, "recipe_id", None) if spec is not None else None
    provenance = "reconstructed" if recipe == "attention_reconstruction" else "captured"
    return value[batch_index].detach().to(torch.float32), provenance


def _mask_from_sdpa(module: Any, n_dst: int, n_src: int, batch_index: int) -> MaskInfo | None:
    """Mask source 1: recorded SDPA call arguments (memo D6/M8)."""

    try:
        from ..semantic.reconstruction import find_sdpa_op
    except ImportError:
        return None
    op = find_sdpa_op(module)
    if op is None:
        return None
    kwargs = dict(getattr(op, "saved_kwargs", {}) or {})
    args = list(getattr(op, "saved_args", ()) or ())
    attn_mask = kwargs.get("attn_mask", args[3] if len(args) > 3 else None)
    if isinstance(attn_mask, torch.Tensor):
        masked = _masked_positions(attn_mask, n_dst, n_src, batch_index)
        if masked is not None:
            return MaskInfo(mask=masked, source="sdpa_call_args")
    is_causal = bool(kwargs.get("is_causal", args[5] if len(args) > 5 else False))
    if is_causal:
        causal = torch.triu(torch.ones(n_dst, n_src, dtype=torch.bool), diagonal=1)
        return MaskInfo(mask=causal, source="sdpa_call_args")
    return None


def _masked_positions(
    mask: torch.Tensor, n_dst: int, n_src: int, batch_index: int
) -> torch.Tensor | None:
    """Reduce a recorded mask tensor to ``[n_dst, n_src]`` bool (True=masked)."""

    work = mask.detach()
    if work.dtype == torch.bool:
        masked = ~work  # torch semantics: True = allowed to attend
    else:
        # Real HF fills are finfo(dtype).min, NOT -inf (memo M7): an
        # isinf-keyed detector silently finds nothing.
        floor = torch.finfo(work.dtype).min
        masked = work <= floor / 2
    masked = _squeeze_to_grid(masked, batch_index)
    if masked.shape[-2:] != (n_dst, n_src):
        try:
            masked = masked.expand(n_dst, n_src)
        except RuntimeError:
            return None
    return masked.contiguous()


def _squeeze_to_grid(masked: torch.Tensor, batch_index: int) -> torch.Tensor:
    """Reduce leading broadcast/batch dims, honoring the batch index.

    A leading dim of extent > 1 is the batch axis (select ``batch_index``);
    a singleton leading dim is broadcast (select 0).
    """

    while masked.ndim > 2:
        index = batch_index if masked.shape[0] > max(1, batch_index) else 0
        masked = masked[index]
    return masked


def _mask_from_eager(trace: Any, module: Any, batch_index: int) -> MaskInfo | None:
    """Mask source 2: the captured eager additive-mask operand (memo M6).

    Reads the ADDITIVE MASK OPERAND itself (the tensor the model added to
    the scores), never the zero pattern cells: find the op that produced
    the ``scores`` facet; when it is an add, inspect its saved parent
    values for a tensor carrying the ``finfo.min`` fill.
    """

    if not _has_facet(module, "scores"):
        return None
    spec = getattr(module.facets["scores"], "spec", None)
    home_label = getattr(spec, "home_label", None)
    if home_label is None:
        return None
    try:
        scores_op = trace[home_label]
    except (KeyError, TorchLensError):
        return None
    shape = tuple(getattr(scores_op, "shape", ()) or ())
    if "add" not in str(getattr(scores_op, "func_name", "")) or len(shape) < 2:
        return None
    n_dst, n_src = shape[-2], shape[-1]
    for parent_label in tuple(getattr(scores_op, "parents", ()) or ()):
        candidate = _additive_mask_from_parent(trace, parent_label, n_dst, n_src, batch_index)
        if candidate is not None:
            return MaskInfo(mask=candidate, source="eager_additive_mask")
    return None


def _additive_mask_from_parent(
    trace: Any, parent_label: str, n_dst: int, n_src: int, batch_index: int
) -> torch.Tensor | None:
    """Return the masked-position grid if a parent op holds the additive fill."""

    try:
        value = trace[parent_label].out
    except (KeyError, TorchLensError):
        return None
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        return None
    floor = torch.finfo(value.dtype).min
    filled = value <= floor / 2
    if not bool(filled.any()):
        return None
    masked = _squeeze_to_grid(filled, batch_index)
    if masked.shape != (n_dst, n_src):
        return None
    return masked.contiguous()


def _gqa_info(module: Any, n_heads: int) -> GqaInfo | None:
    """Return the GQA disclosure when kv grouping is derivable.

    Graph-derived, never config-name-keyed: the captured ``k`` facet is
    served position-major ``[batch, pos, head, d_head]``, so a k head count
    strictly below the pattern's query head count IS the kv grouping.
    """

    if not _has_facet(module, "k"):
        return None
    try:
        k_value = module.facets["k"].value
    except (KeyError, TorchLensError):
        return None
    if not isinstance(k_value, torch.Tensor) or k_value.ndim != 4:
        return None
    n_kv = int(k_value.shape[2])
    if 0 < n_kv < n_heads and n_heads % n_kv == 0:
        return GqaInfo(n_query_heads=n_heads, n_kv_heads=n_kv)
    return None


def _token_axes(
    tokens: Any,
    query_tokens: Any,
    key_tokens: Any,
    n_dst: int,
    n_src: int,
) -> tuple[TokenAxis, TokenAxis]:
    """Build separate query/key axes from the user's token spelling."""

    if tokens is not None and (query_tokens is not None or key_tokens is not None):
        refuse(
            code="tv_record_invalid",
            message="Pass tokens= (self-attention, both axes) XOR "
            "query_tokens=/key_tokens= (cross-attention), not both.",
            remedy="pick one spelling",
        )
    if tokens is not None:
        query_tokens = key_tokens = tuple(str(token) for token in tokens)
    if query_tokens is None:
        query_tokens = tuple(str(index) for index in range(n_dst))
    if key_tokens is None:
        key_tokens = tuple(str(index) for index in range(n_src))
    return (
        TokenAxis(role="query", tokens=tuple(query_tokens)),
        TokenAxis(role="key", tokens=tuple(key_tokens)),
    )


def attention_view(  # noqa: PLR0913 -- the record-shaped extraction surface (axes + batch + mask)
    trace: Any,
    layer: str,
    *,
    tokens: Any = None,
    query_tokens: Any = None,
    key_tokens: Any = None,
    batch_index: int = 0,
    mask: torch.Tensor | None = None,
) -> AttentionView:
    """Extract one layer's attention pattern as a typed view (memo D5/D6).

    Parameters
    ----------
    trace:
        A finished TorchLens trace of a transformer forward.
    layer:
        Module address of the attention module (see
        :func:`attention_views` for discovery).
    tokens:
        Token strings for BOTH axes (self-attention convenience).
    query_tokens:
        Destination-axis tokens (cross-attention; T5 is rectangular).
    key_tokens:
        Source-axis tokens (cross-attention).
    batch_index:
        Batch element to extract.
    mask:
        Explicit user/model mask metadata (source 3 of the hierarchy);
        bool ``[n_destination, n_source]``, True = masked out. Used only
        when no captured source resolves.

    Returns
    -------
    AttentionView
        The canonical ``[head, destination, source]`` view with provenance,
        mask provenance (or none), and the GQA disclosure when declared.
    """

    if layer not in getattr(trace, "modules", {}):
        available = _modules_with_pattern(trace)
        refuse(
            code="tv_facet_missing",
            message=f"No module at address {layer!r} in this trace.",
            remedy=f"pick one of the pattern-bearing modules: {available[:8]}",
            available=available,
        )
    module = trace.modules[layer]
    if not _has_facet(module, "pattern"):
        available = _modules_with_pattern(trace)
        refuse(
            code="tv_facet_missing",
            message=f"Module {layer!r} exposes no attention pattern facet.",
            remedy="pick a pattern-bearing module "
            f"({available[:8]}), or check tl.compat.report(model, x) for the "
            "attention-detection rows",
            available=available,
        )
    pattern, provenance = _pattern_tensor(trace, layer, batch_index)
    n_heads, n_dst, n_src = pattern.shape
    query_axis, key_axis = _token_axes(tokens, query_tokens, key_tokens, n_dst, n_src)
    mask_info = _mask_from_sdpa(module, n_dst, n_src, batch_index)
    if mask_info is None:
        mask_info = _mask_from_eager(trace, module, batch_index)
    if mask_info is None and mask is not None:
        mask_info = MaskInfo(mask=mask.bool(), source="user_metadata")
    return AttentionView(
        pattern=pattern,
        query_tokens=query_axis,
        key_tokens=key_axis,
        heads=tuple(range(n_heads)),
        layer=layer,
        domain="probability",
        provenance=provenance,
        mask=mask_info,
        gqa=_gqa_info(module, n_heads),
    )


def attention_views(  # noqa: PLR0913 -- same record-shaped surface, list form
    trace: Any,
    *,
    layers: list[str] | None = None,
    tokens: Any = None,
    query_tokens: Any = None,
    key_tokens: Any = None,
    batch_index: int = 0,
) -> list[AttentionView]:
    """Extract every (or the named) attention layer's view, in trace order.

    Parameters
    ----------
    trace:
        A finished TorchLens trace.
    layers:
        Explicit module addresses; defaults to every pattern-bearing module.
    tokens:
        Token strings for BOTH axes (see :func:`attention_view`).
    query_tokens:
        Destination-axis tokens (cross-attention).
    key_tokens:
        Source-axis tokens (cross-attention).
    batch_index:
        Batch element to extract.

    Returns
    -------
    list[AttentionView]
        One view per layer.
    """

    if layers is None:
        layers = _modules_with_pattern(trace)
    if not layers:
        refuse(
            code="tv_facet_missing",
            message="No module in this trace exposes an attention pattern facet.",
            remedy="capture a transformer forward (eager attention captures directly; "
            "fused kernels need reconstruction_ready=True), or check "
            "tl.compat.report(model, x)",
        )
    return [
        attention_view(
            trace,
            layer,
            tokens=tokens,
            query_tokens=query_tokens,
            key_tokens=key_tokens,
            batch_index=batch_index,
        )
        for layer in layers
    ]
