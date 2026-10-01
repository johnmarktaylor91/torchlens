"""Read-only reconstruction helpers for semantic facets."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import torch

from .._errors import PostTraceParamUnavailable
from .facets import FacetCapabilityFlags, FacetSpec, MissingFacet

ReconstructionFacet = Literal["scores", "pattern", "z", "result"]


@dataclass(frozen=True)
class SDPAReconstruction:
    """Captured SDPA tensors and options needed for replay validation.

    Parameters
    ----------
    op:
        Fused ``scaled_dot_product_attention`` op record.
    q:
        Actual query tensor consumed by the fused op.
    k:
        Actual key tensor consumed by the fused op.
    v:
        Actual value tensor consumed by the fused op.
    attn_mask:
        Optional attention mask tensor or scalar value.
    dropout_p:
        SDPA dropout probability.
    is_causal:
        Whether SDPA applied a causal mask.
    scale:
        Effective scale. ``None`` means PyTorch's default ``1 / sqrt(d_head)``.
    enable_gqa:
        Whether PyTorch's SDPA GQA expansion convention was requested.
    """

    op: Any
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    attn_mask: Any
    dropout_p: float
    is_causal: bool
    scale: float | None
    enable_gqa: bool


def sdpa_reconstruction_spec(
    module: Any,
    facet: ReconstructionFacet,
    *,
    recipe_id: str,
) -> FacetSpec | MissingFacet:
    """Return a read-only reconstructed SDPA facet spec for an attention module.

    Parameters
    ----------
    module:
        TorchLens module record whose ops may contain an SDPA call.
    facet:
        Public facet to reconstruct.
    recipe_id:
        Recipe identifier for provenance.

    Returns
    -------
    FacetSpec | MissingFacet
        Computed spec when reconstruction prerequisites are present, otherwise
        an actionable missing-facet sentinel.
    """

    sdpa_op = find_sdpa_op(module)
    if sdpa_op is None:
        return MissingFacet(
            f"{facet} reconstruction missing prerequisite: eager attention op or fused "
            f"scaled_dot_product_attention op inside module {getattr(module, 'address', '<unknown>')!r}."
        )
    missing = _missing_sdpa_prerequisite(sdpa_op)
    if missing is not None:
        return MissingFacet(f"{facet} reconstruction missing prerequisite: {missing}.")
    if facet == "result" and _output_projection(module) is None:
        return MissingFacet(
            "result reconstruction missing prerequisite: output projection child "
            "(o_proj/c_proj/out_proj/dense/out_lin/o)."
        )
    spec = FacetSpec.computed(
        lambda: _reconstruct_checked(module, sdpa_op, facet),
        recipe_id=recipe_id,
        recipe_version="p3",
    )
    return _as_reconstructed_spec(spec)


def find_sdpa_op(module: Any) -> Any | None:
    """Return the first fused SDPA op contained in ``module``.

    Parameters
    ----------
    module:
        TorchLens module record.

    Returns
    -------
    Any | None
        Matching op record, or ``None``.
    """

    trace = getattr(module, "trace", None)
    if trace is None:
        return None
    labels: list[str] = []
    try:
        labels = list(module._op_labels())
    except (AttributeError, TypeError, ValueError):
        labels = list(getattr(module, "output_ops", ()) or ())
    for label in labels:
        try:
            op = trace.ops[label]
        except (KeyError, TypeError):
            continue
        if _is_sdpa_op(op):
            return op
    return None


def _as_reconstructed_spec(spec: FacetSpec) -> FacetSpec:
    """Return a computed spec marked as reconstructed.

    Parameters
    ----------
    spec:
        Computed spec to annotate.

    Returns
    -------
    FacetSpec
        Annotated read-only reconstructed spec.
    """

    return FacetSpec(
        home_kind=spec.home_kind,
        home_label=spec.home_label,
        home_address=spec.home_address,
        pass_index=spec.pass_index,
        call_index=spec.call_index,
        output_path=spec.output_path,
        transforms=spec.transforms,
        capability_class=spec.capability_class,
        capability_flags=FacetCapabilityFlags(
            read=True,
            grad=False,
            write=False,
            portable=False,
            reconstructed=True,
        ),
        value_version=spec.value_version,
        conflict_alias_group=spec.conflict_alias_group,
        recipe_id=spec.recipe_id,
        recipe_version=spec.recipe_version,
        home=spec.home,
    )


def _is_sdpa_op(op: Any) -> bool:
    """Return whether an op record is a fused SDPA call.

    Parameters
    ----------
    op:
        Candidate op record.

    Returns
    -------
    bool
        Whether the op is an SDPA call.
    """

    fields = (
        getattr(op, "func_name", None),
        getattr(op, "func_qualname", None),
        getattr(op, "label", None),
    )
    return any("scaled_dot_product_attention" in str(field) for field in fields)


def _missing_sdpa_prerequisite(op: Any) -> str | None:
    """Return a named missing prerequisite for SDPA reconstruction.

    Parameters
    ----------
    op:
        SDPA op record.

    Returns
    -------
    str | None
        Missing prerequisite message, or ``None`` when ready.
    """

    if getattr(op, "saved_args", None) is None or getattr(op, "saved_kwargs", None) is None:
        return "save_arg_values=True or reconstruction_ready=True"
    args = list(getattr(op, "saved_args", ()) or ())
    if len(args) < 3 or not all(isinstance(value, torch.Tensor) for value in args[:3]):
        return "saved SDPA Q/K/V tensor args"
    kwargs = dict(getattr(op, "saved_kwargs", {}) or {})
    dropout_p = float(kwargs.get("dropout_p", _positional_or_default(args, 4, 0.0)) or 0.0)
    if dropout_p:
        return "dropout_p=0.0; stochastic SDPA dropout reconstruction is fail-closed"
    return None


def _sdpa_record(op: Any) -> SDPAReconstruction | MissingFacet:
    """Build an SDPA reconstruction record from saved op args.

    Parameters
    ----------
    op:
        Fused SDPA op.

    Returns
    -------
    SDPAReconstruction | MissingFacet
        Reconstruction record or named missing prerequisite.
    """

    missing = _missing_sdpa_prerequisite(op)
    if missing is not None:
        return MissingFacet(f"SDPA reconstruction missing prerequisite: {missing}.")
    args = list(getattr(op, "saved_args", ()) or ())
    kwargs = dict(getattr(op, "saved_kwargs", {}) or {})
    q, k, v = args[:3]
    attn_mask = kwargs.get("attn_mask", _positional_or_default(args, 3, None))
    dropout_p = float(kwargs.get("dropout_p", _positional_or_default(args, 4, 0.0)) or 0.0)
    is_causal = bool(kwargs.get("is_causal", _positional_or_default(args, 5, False)))
    scale = kwargs.get("scale")
    enable_gqa = bool(kwargs.get("enable_gqa", False))
    if scale is not None:
        scale = float(scale)
    return SDPAReconstruction(
        op=op,
        q=q,
        k=k,
        v=v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
    )


def _positional_or_default(args: list[Any], index: int, default: Any) -> Any:
    """Return a positional value when present.

    Parameters
    ----------
    args:
        Saved positional args.
    index:
        Position to read.
    default:
        Default returned when the position is absent.

    Returns
    -------
    Any
        Positional value or default.
    """

    return args[index] if len(args) > index else default


def _reconstruct_checked(
    module: Any,
    sdpa_op: Any,
    facet: ReconstructionFacet,
) -> torch.Tensor | MissingFacet:
    """Reconstruct an SDPA facet and validate it against the captured target.

    Parameters
    ----------
    module:
        TorchLens attention module record.
    sdpa_op:
        Fused SDPA op.
    facet:
        Facet to reconstruct.

    Returns
    -------
    torch.Tensor | MissingFacet
        Validated reconstructed tensor or a missing sentinel.
    """

    record = _sdpa_record(sdpa_op)
    if isinstance(record, MissingFacet):
        return record
    scores = _attention_scores(record)
    pattern = torch.softmax(scores.float(), dim=-1).to(record.q.dtype)
    expanded_v = _expanded_v(record)
    z = torch.matmul(pattern, expanded_v)
    # Cancellation-aware magnitude bound for the z check: the summation behind
    # each z element is sum_k pattern[..,k] * V[..,k,:] with pattern >= 0, so
    # the accumulated-|addend| bound is exactly pattern @ |V|.
    z_magnitude = torch.matmul(pattern.float(), expanded_v.abs().float())
    if not _allclose_sdpa(
        z, sdpa_op.out, magnitude=z_magnitude, reduction_length=int(expanded_v.shape[-2])
    ):
        return MissingFacet(
            f"{facet} reconstruction validation failed: recomputed z did not match "
            f"SDPA op output {getattr(sdpa_op, 'label', '<unknown>')!r}."
        )
    if facet == "scores":
        return scores
    if facet == "pattern":
        return pattern
    if facet == "z":
        return z
    return _result_or_missing(module, z, sdpa_op)


def _attention_scores(record: SDPAReconstruction) -> torch.Tensor:
    """Return scaled, masked attention scores for an SDPA record.

    Parameters
    ----------
    record:
        SDPA reconstruction record.

    Returns
    -------
    torch.Tensor
        Attention score tensor.
    """

    q = record.q.float()
    k = _expanded_k(record).float()
    scale = record.scale if record.scale is not None else 1.0 / math.sqrt(record.q.shape[-1])
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if record.attn_mask is not None:
        scores = _apply_attention_mask(scores, record.attn_mask)
    if record.is_causal:
        scores = _apply_causal_mask(scores)
    return scores.to(record.q.dtype)


def _expanded_k(record: SDPAReconstruction) -> torch.Tensor:
    """Return K expanded with PyTorch SDPA GQA convention when needed."""

    return _expand_gqa(record.k, record.q.shape[-3], record.enable_gqa)


def _expanded_v(record: SDPAReconstruction) -> torch.Tensor:
    """Return V expanded with PyTorch SDPA GQA convention when needed."""

    return _expand_gqa(record.v, record.q.shape[-3], record.enable_gqa)


def _expand_gqa(value: torch.Tensor, n_q_heads: int, enable_gqa: bool) -> torch.Tensor:
    """Expand grouped K/V heads to query heads when SDPA GQA was enabled.

    Parameters
    ----------
    value:
        K or V tensor with head dimension at ``-3``.
    n_q_heads:
        Number of query heads.
    enable_gqa:
        Whether grouped-query expansion was requested.

    Returns
    -------
    torch.Tensor
        Original or repeated tensor.
    """

    n_kv_heads = value.shape[-3]
    if not enable_gqa or n_kv_heads == n_q_heads:
        return value
    if n_q_heads % n_kv_heads:
        return value
    return value.repeat_interleave(n_q_heads // n_kv_heads, dim=-3)


def _apply_attention_mask(scores: torch.Tensor, mask: Any) -> torch.Tensor:
    """Apply a boolean or additive SDPA mask to scores.

    Parameters
    ----------
    scores:
        Attention scores.
    mask:
        SDPA attention mask.

    Returns
    -------
    torch.Tensor
        Masked scores.
    """

    if not isinstance(mask, torch.Tensor):
        raise RuntimeError("unsupported mask prerequisite: non-tensor SDPA mask")
    mask_value = mask.to(device=scores.device)
    if mask_value.dtype == torch.bool:
        return scores.masked_fill(~mask_value, float("-inf"))
    return scores + mask_value.to(dtype=scores.dtype)


def _apply_causal_mask(scores: torch.Tensor) -> torch.Tensor:
    """Apply a lower-triangular causal mask to SDPA scores.

    Parameters
    ----------
    scores:
        Attention scores with query length at ``-2`` and key length at ``-1``.

    Returns
    -------
    torch.Tensor
        Causally masked scores.
    """

    q_len = scores.shape[-2]
    k_len = scores.shape[-1]
    diagonal = k_len - q_len
    causal = torch.ones((q_len, k_len), dtype=torch.bool, device=scores.device).tril(diagonal)
    return scores.masked_fill(~causal, float("-inf"))


def _allclose_sdpa(
    reconstructed: torch.Tensor,
    target: torch.Tensor,
    *,
    magnitude: torch.Tensor | None = None,
    reduction_length: int | None = None,
) -> bool:
    """Return whether a reconstructed tensor matches an SDPA output.

    This gate decides whether a reconstructed facet (``scores`` / ``pattern``
    / ``z`` / ``result``) is served to the user as REAL or refused as
    ``MissingFacet``, so its tolerances are verification tolerances, not
    formatting slack. The former hand-picked absolute floors (atol 2e-2 for
    fp16/bf16, 1e-5 for fp32) blessed an ALL-ZERO and a SIGN-FLIPPED
    reconstruction of any payload living below the floor -- post-softmax
    attention values do exactly that (b4-opus F13-1, probe-proven). The
    replay-table pair that replaced them (denormal-scale absolute term) then
    refused numerically CORRECT reconstructions wherever the summation behind
    the target CANCELLED (measured: 19 of 36 correct SDPA reconstructions,
    input-dependently) -- a reconstruction is independent math, not a replay,
    so its achievable absolute accuracy at cancellation sites is set by the
    ADDEND magnitudes. The gate now uses the SEPARATE cancellation-aware
    reconstruction error model (``torchlens.semantic.tolerances``): same
    ULP-headroom relative term and denormal floor, plus an elementwise
    ``magnitude``-scaled term supplied by each call site. The replay table in
    ``torchlens.utils.tensor_utils`` is deliberately untouched.

    Parameters
    ----------
    reconstructed:
        Reconstructed tensor.
    target:
        Captured target tensor.
    magnitude:
        Elementwise accumulated-|addend| bound of the summation that produced
        ``target``; ``None`` drops the cancellation term (strict comparison).
    reduction_length:
        Number of addends behind ``magnitude``.

    Returns
    -------
    bool
        Whether values match within the reconstruction error budget.
    """

    from .tolerances import within_reconstruction_tolerance

    return within_reconstruction_tolerance(
        reconstructed, target, magnitude=magnitude, reduction_length=reduction_length
    )


def _output_projection(module: Any) -> Any | None:
    """Return a common output-projection child module record.

    Parameters
    ----------
    module:
        Attention module record.

    Returns
    -------
    Any | None
        Child module record when present.
    """

    from .recipes._helpers import child_module

    for name in ("o_proj", "c_proj", "out_proj", "dense", "out_lin", "o"):
        child = child_module(module, name)
        if child is not None:
            return child
    return None


def _result_or_missing(module: Any, z: torch.Tensor, sdpa_op: Any) -> torch.Tensor | MissingFacet:
    """Return per-head output projection contributions or a missing sentinel.

    Parameters
    ----------
    module:
        Attention module record.
    z:
        Validated SDPA output shaped ``(..., H, S, D)``.
    sdpa_op:
        SDPA op for diagnostics.

    Returns
    -------
    torch.Tensor | MissingFacet
        Per-head residual contributions shaped ``(..., S, H, E)``.
    """

    projection = _output_projection(module)
    if projection is None:
        return MissingFacet("result reconstruction missing prerequisite: output projection child.")
    weight = _projection_weight(projection)
    if weight is None:
        return MissingFacet("result reconstruction missing prerequisite: output projection weight.")
    if z.ndim < 4:
        return MissingFacet("result reconstruction missing prerequisite: SDPA output rank >= 4.")
    # Orientation comes from the projection MODULE CLASS, never from shape:
    # ``transformers.pytorch_utils.Conv1D`` stores its weight [in, out]
    # (forward is ``x @ W + b``) while ``nn.Linear`` stores [out, in], and a
    # square projection (d_model == n_heads * d_head, i.e. every standard
    # transformer) passes any shape guard with either orientation -- the
    # GPT-2 per-head result was deterministically wrong this way (sum-check
    # error 18.80).
    if _projection_weight_is_in_out(projection):
        weight = weight.transpose(0, 1)
    n_heads = z.shape[-3]
    d_head = z.shape[-1]
    if weight.shape[1] != n_heads * d_head:
        return MissingFacet(
            "result reconstruction missing prerequisite: output projection input dimension "
            f"{weight.shape[1]} does not equal heads*d_head {n_heads * d_head}."
        )
    z_by_sequence = z.transpose(-3, -2)
    per_head_weight = weight.reshape(weight.shape[0], n_heads, d_head).transpose(0, 1)
    result = torch.einsum("...shd,hod->...sho", z_by_sequence, per_head_weight)
    projected = _projection_output(projection)
    if projected is not None:
        bias = _projection_bias(projection)
        summed = result.sum(dim=-2)
        # Accumulated-|addend| bound for the sum check: the head axis is the
        # summation, plus the bias addend when present.
        summed_magnitude = result.detach().abs().float().sum(dim=-2)
        reduction_length = int(result.shape[-2])
        if bias is not None:
            summed = summed + bias
            summed_magnitude = summed_magnitude + bias.detach().abs().float()
            reduction_length += 1
        if not _allclose_sdpa(
            summed, projected, magnitude=summed_magnitude, reduction_length=reduction_length
        ):
            return MissingFacet(
                f"result reconstruction validation failed: summed per-head contributions "
                f"did not match output projection after SDPA op {getattr(sdpa_op, 'label', '<unknown>')!r}."
            )
    return result


#: Projection classes whose weight is stored [in, out] (forward ``x @ W``).
#: ``transformers.pytorch_utils.Conv1D`` is the GPT-2-family projection; the
#: nn.Linear family stores [out, in]. Class-keyed on purpose: square weights
#: make orientation undecidable from shape, and a wrong orientation on an
#: UNKNOWN class is caught fail-closed by the sum check against the captured
#: projection output rather than served silently.
_IN_OUT_PROJECTION_CLASS_NAMES = frozenset({"Conv1D"})


def _projection_weight_is_in_out(module: Any) -> bool:
    """Return whether a projection module's weight is stored [in, out]."""

    class_name = str(getattr(module, "class_name", "") or "")
    if not class_name:
        cls = getattr(module, "cls", None)
        if cls is not None:
            class_name = type(cls).__name__
    return class_name in _IN_OUT_PROJECTION_CLASS_NAMES


def _projection_weight(module: Any) -> torch.Tensor | None:
    """Return a module record's projection weight tensor."""

    cls = getattr(module, "cls", None)
    weight = getattr(cls, "weight", None)
    if isinstance(weight, torch.Tensor):
        return weight
    return _parameter_tensor(module, "weight")


def _projection_bias(module: Any) -> torch.Tensor | None:
    """Return a module record's projection bias tensor."""

    cls = getattr(module, "cls", None)
    bias = getattr(cls, "bias", None)
    if isinstance(bias, torch.Tensor):
        return bias
    return _parameter_tensor(module, "bias")


def _parameter_tensor(module: Any, name: str) -> torch.Tensor | None:
    """Return a live parameter tensor from a module record.

    Parameters
    ----------
    module:
        TorchLens module record.
    name:
        Parameter short name.

    Returns
    -------
    torch.Tensor | None
        Live parameter tensor when available.
    """

    params = getattr(module, "params", None)
    if params is None:
        return None
    try:
        iterable = params.values() if hasattr(params, "values") else params
        for param in iterable:
            if getattr(param, "name", None) != name:
                continue
            try:
                value = getattr(param, "value", None)
            except PostTraceParamUnavailable:
                value = getattr(param, "_param_ref", None)
            if value is None:
                value = getattr(param, "_param_ref", None)
            if isinstance(value, torch.Tensor):
                return value
    except TypeError:
        return None
    return None


def _projection_output(module: Any) -> torch.Tensor | None:
    """Return a projection module's captured single output tensor."""

    try:
        return module.calls[0].out
    except (AttributeError, KeyError, IndexError, RuntimeError, ValueError):
        return None
