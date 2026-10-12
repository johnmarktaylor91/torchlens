"""TorchLens as a registered Hugging Face attention backend (DOCUMENTED-UNSTABLE).

``register_attention()`` registers a ``"torchlens"`` attention function in
transformers' ``AttentionInterface`` together with its MATCHING mask function
in ``AttentionMaskInterface``. Registering the attention function alone is the
mask trap: transformers builds no mask for a key its mask registry does not
know, so a custom function silently computes non-causal attention. The mask
function registered here is transformers' own boolean ``sdpa_mask`` with the
causal skip disabled, so the attention function always receives an explicit
mask whenever one exists.

The attention function computes the masked QK-softmax explicitly, multiplies
the very probabilities it computed by the values, and returns those
probabilities as the attention weights (so ``output_attentions=True`` works
under this backend). Inside ``record_attention(model)`` every call records its
per-head probabilities, keyed by the attention module's qualified name,
without a TorchLens trace.

Memory cost: unlike fused SDPA or flash kernels, this backend materializes
the full ``(batch, heads, q_len, kv_len)`` probability matrix in every
attention call. Recording keeps every matrix alive until the recording is
dropped: ``batch * heads * q_len * kv_len * itemsize`` bytes per call. For a
Llama-2-7B-shaped model (32 layers, 32 heads) at 4096 tokens in bfloat16 that
is 1 GiB per layer and 32 GiB per sequence. ``AttentionRecording.nbytes``
reports the retained total.

Every spelling here is DOCUMENTED-UNSTABLE pending naming-session
ratification; no deprecation shim is owed on rename.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

import torch

from torchlens._errors import InvalidArgumentError, MissingDependencyError
from torchlens.errors import ValidationError

BACKEND_KEY = "torchlens"

#: Attention-function kwargs this backend does not implement; a non-None
#: value would change the math (sparse-index or packed-sequence attention).
_UNSUPPORTED_KWARGS = ("indices", "block_indices", "cu_seq_lens_q", "cu_seq_lens_k")


@dataclass(frozen=True)
class AttentionCall:
    """One recorded attention call.

    Attributes
    ----------
    module_name:
        Qualified name of the attention module inside the recorded model.
    probs:
        Detached per-head probabilities ``(batch, heads, q_len, kv_len)``: the
        exact tensor the backend multiplied by the values.
    num_key_value_heads:
        Key/value heads before grouped-query expansion.
    mask_kind:
        ``"boolean"``, ``"additive"``, ``"causal"`` (no mask tensor; causal
        applied by the backend) or ``"none"``.
    sdpa_checked:
        Whether this call's output was checked against
        ``torch.nn.functional.scaled_dot_product_attention``.
    """

    module_name: str
    probs: torch.Tensor
    num_key_value_heads: int
    mask_kind: str
    sdpa_checked: bool


@dataclass
class AttentionRecording(Mapping[str, torch.Tensor]):
    """Per-head attention probabilities recorded by the TorchLens backend.

    A read-only mapping from attention-module name to its probabilities.
    ``recording[name]`` serves a module called exactly once; a module called
    several times (generation steps, reused blocks) refuses typed and
    ``calls_for(name)`` returns every call in order.
    """

    calls: list[AttentionCall] = field(default_factory=list)
    check_sdpa: bool = False
    _names: dict[int, str] = field(default_factory=dict, repr=False)

    def __getitem__(self, name: str) -> torch.Tensor:
        """Return the probabilities of the module's single call."""

        matches = self.calls_for(name)
        if not matches:
            raise KeyError(name)
        if len(matches) > 1:
            raise InvalidArgumentError(
                f"attention module {name!r} was called {len(matches)} times in this recording",
                code="hf_attention_multiple_calls",
                remedy=f"read every call with recording.calls_for({name!r})",
            )
        return matches[0].probs

    def __iter__(self) -> Iterator[str]:
        """Iterate module names in first-call order."""

        return iter(dict.fromkeys(call.module_name for call in self.calls))

    def __len__(self) -> int:
        """Return the number of distinct recorded modules."""

        return len(dict.fromkeys(call.module_name for call in self.calls))

    def calls_for(self, name: str) -> list[AttentionCall]:
        """Return every recorded call of one attention module, in call order."""

        return [call for call in self.calls if call.module_name == name]

    @property
    def nbytes(self) -> int:
        """Total bytes held by the recorded probability matrices."""

        return sum(call.probs.numel() * call.probs.element_size() for call in self.calls)

    def _module_name(self, module: Any) -> str:
        """Resolve a module's qualified name (``<ClassName>`` if foreign)."""

        return self._names.get(id(module), f"<{type(module).__name__}>")


_ACTIVE_RECORDING: ContextVar[AttentionRecording | None] = ContextVar(
    "torchlens_hf_attention_recording", default=None
)


def _require_registries() -> tuple[Any, Any, Any]:
    """Import the transformers attention, mask registries and ``sdpa_mask``.

    Raises
    ------
    MissingDependencyError
        When transformers or its attention-mask registry is unavailable.
    """

    try:
        from transformers import AttentionInterface
        from transformers.masking_utils import AttentionMaskInterface, sdpa_mask
    except ImportError as error:
        raise MissingDependencyError(
            "the TorchLens Hugging Face attention backend needs transformers with both "
            "the AttentionInterface and the AttentionMaskInterface registries",
            code="hf_attention_registry_unavailable",
            remedy='pip install "transformers>=4.53"',
        ) from error
    return AttentionInterface, AttentionMaskInterface, sdpa_mask


def torchlens_mask(*args: Any, **kwargs: Any) -> torch.Tensor | None:
    """Build the boolean attention mask matching the TorchLens backend.

    transformers' ``sdpa_mask`` with the causal skip forced off, so a causal
    pattern always arrives as an explicit mask (True = attend). A fully
    bidirectional unpadded pattern may still return ``None``.
    """

    _, _, sdpa_mask = _require_registries()
    kwargs["allow_is_causal_skip"] = False
    kwargs.pop("dtype", None)
    kwargs.pop("config", None)
    mask: torch.Tensor | None = sdpa_mask(*args, **kwargs)
    return mask


def _repeat_kv(states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand grouped key/value heads to the query head count."""

    if n_rep == 1:
        return states
    batch, kv_heads, length, dim = states.shape
    expanded = states[:, :, None, :, :].expand(batch, kv_heads, n_rep, length, dim)
    return expanded.reshape(batch, kv_heads * n_rep, length, dim)


def _apply_mask(
    scores: torch.Tensor, attention_mask: torch.Tensor | None, causal: bool
) -> tuple[torch.Tensor, str]:
    """Apply a boolean, additive or implicit causal mask to float32 scores."""

    if attention_mask is not None and attention_mask.dtype == torch.bool:
        return scores.masked_fill(~attention_mask, float("-inf")), "boolean"
    if attention_mask is not None:
        return scores + attention_mask.to(scores.dtype), "additive"
    q_len, kv_len = scores.shape[-2], scores.shape[-1]
    if causal and q_len > 1:
        # Upper-left alignment, matching SDPA's is_causal semantics.
        keep = torch.ones(q_len, kv_len, dtype=torch.bool, device=scores.device).tril()
        return scores.masked_fill(~keep, float("-inf")), "causal"
    return scores, "none"


def _masked_softmax(scores: torch.Tensor, sinks: torch.Tensor | None) -> torch.Tensor:
    """Softmax over keys; fully masked rows get zero probability, never NaN."""

    if sinks is not None:
        sink_logits = sinks.reshape(1, -1, 1, 1).to(scores.dtype)
        sink_logits = sink_logits.expand(scores.shape[0], -1, scores.shape[-2], 1)
        combined = torch.cat([scores, sink_logits], dim=-1)
        return torch.softmax(combined, dim=-1)[..., :-1]
    fully_masked = torch.isneginf(scores).all(dim=-1, keepdim=True)
    probs = torch.softmax(scores, dim=-1)
    return probs.masked_fill(fully_masked, 0.0)


def _check_unsupported(kwargs: Mapping[str, Any]) -> None:
    """Refuse attention kwargs whose math this backend does not implement."""

    for name in _UNSUPPORTED_KWARGS:
        if kwargs.get(name) is not None:
            raise InvalidArgumentError(
                f"the TorchLens attention backend does not implement {name!r} "
                "(sparse-index or packed-sequence attention)",
                code="hf_attention_kwarg_unsupported",
                remedy='load this model with attn_implementation="eager" or "sdpa"',
                kwarg=name,
            )


def torchlens_attention_forward(  # noqa: PLR0913 -- the transformers attention-function contract
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float | None = None,
    dropout: float = 0.0,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute explicit masked attention and record per-head probabilities.

    Follows the transformers attention-function contract: ``query`` is
    ``(batch, heads, q_len, head_dim)``, ``key``/``value`` are
    ``(batch, kv_heads, kv_len, head_dim)``, and the return value is
    ``(attn_output transposed to (batch, q_len, heads, head_dim), probs)``.
    Scores and softmax run in float32; probabilities are cast to the value
    dtype before the value product, and that cast tensor is what is
    returned and recorded. ``softcap``, ``position_bias`` and attention
    sinks (``s_aux``) follow the transformers eager references.
    """

    _check_unsupported(kwargs)
    n_rep = query.shape[1] // key.shape[1]
    key_states = _repeat_kv(key, n_rep)
    value_states = _repeat_kv(value, n_rep)
    scale = scaling if scaling is not None else query.shape[-1] ** -0.5
    scores = torch.matmul(query.float(), key_states.float().transpose(-1, -2)) * scale
    softcap = kwargs.get("softcap")
    if softcap is not None:
        scores = torch.tanh(scores / softcap) * softcap
    position_bias = kwargs.get("position_bias")
    if position_bias is not None:
        scores = scores + position_bias.float()
    causal = kwargs.get("is_causal")
    causal = getattr(module, "is_causal", True) if causal is None else causal
    scores, mask_kind = _apply_mask(scores, attention_mask, bool(causal))
    probs = _masked_softmax(scores, kwargs.get("s_aux")).to(value_states.dtype)
    probs = torch.nn.functional.dropout(probs, p=dropout, training=module.training)
    heads_out = torch.matmul(probs, value_states)
    recording = _ACTIVE_RECORDING.get()
    if recording is not None:
        facts = _CallFacts(
            query=query,
            key_states=key_states,
            value_states=value_states,
            heads_out=heads_out,
            probs=probs,
            attention_mask=attention_mask,
            position_bias=position_bias,
            scale=scale,
            mask_kind=mask_kind,
            plain=dropout == 0.0 and softcap is None and kwargs.get("s_aux") is None,
            kv_heads=key.shape[1],
        )
        _record_call(recording, module, facts)
    return heads_out.transpose(1, 2).contiguous(), probs


@dataclass(frozen=True)
class _CallFacts:
    """Tensors and facts of one attention call, for recording and the SDPA check."""

    query: torch.Tensor
    key_states: torch.Tensor
    value_states: torch.Tensor
    heads_out: torch.Tensor
    probs: torch.Tensor
    attention_mask: torch.Tensor | None
    position_bias: torch.Tensor | None
    scale: float
    mask_kind: str
    plain: bool
    kv_heads: int


def _record_call(recording: AttentionRecording, module: torch.nn.Module, facts: _CallFacts) -> None:
    """Append one call to the active recording, checking SDPA when asked."""

    name = recording._module_name(module)
    checked = recording.check_sdpa and facts.plain
    if checked:
        _check_against_sdpa(name, facts)
    recording.calls.append(
        AttentionCall(
            module_name=name,
            probs=facts.probs.detach(),
            num_key_value_heads=int(facts.kv_heads),
            mask_kind=facts.mask_kind,
            sdpa_checked=checked,
        )
    )


def _sdpa_reference_mask(facts: _CallFacts) -> tuple[torch.Tensor | None, bool]:
    """Return the SDPA ``attn_mask`` and ``is_causal`` matching one call.

    A position bias folds into a float mask, as transformers' own SDPA
    function does (``create_position_bias_mask``).
    """

    mask = facts.attention_mask
    is_causal = mask is None and facts.mask_kind == "causal"
    if facts.position_bias is None:
        return mask, is_causal
    bias = facts.position_bias.float()
    if is_causal:
        q_len, kv_len = facts.query.shape[-2], facts.key_states.shape[-2]
        mask = torch.ones(q_len, kv_len, dtype=torch.bool, device=bias.device).tril()
    if mask is None:
        return bias, False
    if mask.dtype == torch.bool:
        return bias.masked_fill(~mask, float("-inf")), False
    return bias + mask.float(), False


def _check_against_sdpa(name: str, facts: _CallFacts) -> None:
    """Check one call's output against SDPA at the reconstruction tolerance.

    Uses the same cancellation-aware budget as the semantic SDPA ``z``
    reconstruction gate (``torchlens.semantic.tolerances``).
    """

    from torchlens.semantic.tolerances import within_reconstruction_tolerance

    attn_mask, is_causal = _sdpa_reference_mask(facts)
    if attn_mask is not None and attn_mask.is_floating_point():
        attn_mask = attn_mask.to(facts.query.dtype)
    with torch.no_grad():
        reference = torch.nn.functional.scaled_dot_product_attention(
            facts.query, facts.key_states, facts.value_states, attn_mask=attn_mask,
            scale=facts.scale, is_causal=is_causal,
        )  # fmt: skip
        magnitude = torch.matmul(facts.probs.float(), facts.value_states.float().abs())
        if facts.mask_kind == "boolean" and facts.attention_mask is not None:
            # SDPA leaves fully masked query rows undefined across versions.
            defined = facts.attention_mask.any(dim=-1, keepdim=True).expand_as(reference)
            reference = torch.where(defined, reference, facts.heads_out)
        ok = within_reconstruction_tolerance(
            facts.heads_out,
            reference,
            magnitude=magnitude,
            reduction_length=facts.value_states.shape[-2],
        )
    if not ok:
        diff = float((facts.heads_out.float() - reference.float()).abs().max())
        raise ValidationError(
            f"TorchLens attention output for {name!r} disagrees with SDPA "
            f"(max abs difference {diff:.3g}). Remedy: report this model; "
            'compare against attn_implementation="eager".',
            code="hf_attention_sdpa_mismatch",
            module_name=name,
            max_abs_diff=diff,
        )


def _key_taken(interface: Any) -> bool:
    """Return whether ``BACKEND_KEY`` is registered to a foreign function."""

    mapping = interface()
    return BACKEND_KEY in mapping and mapping[BACKEND_KEY] not in (
        torchlens_attention_forward,
        torchlens_mask,
    )


def register_attention(model: Any | None = None) -> str:
    """Register the ``"torchlens"`` attention backend and its matching mask.

    Idempotent. Registers ``torchlens_attention_forward`` in transformers'
    ``AttentionInterface`` and ``torchlens_mask`` in its
    ``AttentionMaskInterface`` under the same key, so the model always
    receives an explicit causal/padding mask. Never registers over a
    built-in key such as ``"sdpa"`` (that would hijack every model in the
    process). Read the module docstring for the full-matrix memory cost.

    Parameters
    ----------
    model:
        Optional Hugging Face model; when given, it is switched to the
        backend with ``model.set_attn_implementation("torchlens")``.

    Returns
    -------
    str
        The registry key, for ``from_pretrained(..., attn_implementation=key)``.

    Raises
    ------
    MissingDependencyError
        When transformers or its mask registry is unavailable.
    InvalidArgumentError
        When ``"torchlens"`` is already registered to a foreign function.
    """

    attention_interface, mask_interface, _ = _require_registries()
    if _key_taken(attention_interface) or _key_taken(mask_interface):
        raise InvalidArgumentError(
            f"the attention key {BACKEND_KEY!r} is already registered to a non-TorchLens function",
            code="hf_attention_key_taken",
            remedy=f"remove the foreign {BACKEND_KEY!r} registration before registering TorchLens",
        )
    attention_interface.register(BACKEND_KEY, torchlens_attention_forward)
    mask_interface.register(BACKEND_KEY, torchlens_mask)
    if model is not None:
        model.set_attn_implementation(BACKEND_KEY)
    return BACKEND_KEY


def _selected_backend(model: Any) -> str | None:
    """Return the model's configured attention implementation, if readable."""

    config = getattr(model, "config", None)
    value = getattr(config, "_attn_implementation", None)
    return value if isinstance(value, str) else None


@contextlib.contextmanager
def record_attention(model: Any, *, check_sdpa: bool = False) -> Iterator[AttentionRecording]:
    """Record per-head attention probabilities from the TorchLens backend.

    Parameters
    ----------
    model:
        A Hugging Face model switched to the backend (``register_attention``
        with the model, or ``attn_implementation="torchlens"``).
    check_sdpa:
        Also run ``scaled_dot_product_attention`` on each call's inputs and
        refuse (``hf_attention_sdpa_mismatch``) when the outputs disagree
        beyond the semantic SDPA reconstruction tolerance. Calls with
        dropout, softcapping or attention sinks are not checked
        (``AttentionCall.sdpa_checked`` says which were). Doubles the
        attention cost.

    Yields
    ------
    AttentionRecording
        Filled as the model runs inside the block.

    Raises
    ------
    InvalidArgumentError
        When the model is not configured for the TorchLens backend (a
        recording would otherwise stay silently empty).
    """

    selected = _selected_backend(model)
    if selected != BACKEND_KEY:
        raise InvalidArgumentError(
            f"the model's attention implementation is {selected!r}, not {BACKEND_KEY!r}, "
            "so nothing would be recorded",
            code="hf_attention_backend_not_selected",
            remedy="call tl.bridge.hf.register_attention(model) first",
        )
    names = {id(module): name for name, module in model.named_modules()}
    recording = AttentionRecording(check_sdpa=check_sdpa, _names=names)
    token = _ACTIVE_RECORDING.set(recording)
    try:
        yield recording
    finally:
        _ACTIVE_RECORDING.reset(token)


__all__ = [
    "BACKEND_KEY",
    "AttentionCall",
    "AttentionRecording",
    "record_attention",
    "register_attention",
    "torchlens_attention_forward",
    "torchlens_mask",
]
