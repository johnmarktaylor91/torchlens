"""Head-taxonomy scores (mikit section 8; credit: TLens ``head_detector``).

``head_scores`` consumes ``[batch, head, destination, source]`` attention
patterns; masks are TOKEN-EQUALITY definitions matching TransformerLens
semantics (previous-token: source ``i-1``; duplicate: earlier equal-token
sources, diagonal excluded; induction: the duplicate relation shifted one
right), with TLens-compatible ``mul`` (attention-mass) and ``abs``
conventions so numbers are comparable. Reads only, so FUSED attention is
fully supported -- TLens needs an eager reload to find induction heads on a
fused model. Empty-eligible rows are masked and reported, never scored zero.

The seeded repeated-random-token prompt is a diagnostic INPUT and test
fixture, not part of these definitions. ``pattern_source=`` is the socket
through which SSM pseudo-patterns arrive later.

Spellings DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ._errors import refuse
from ._heads import _attention_modules

__all__ = ["HeadScoreResult", "head_scores"]

_KINDS = ("previous_token", "duplicate_token", "induction")
_MEASURES = ("mul", "abs")


@dataclass(frozen=True)
class HeadScoreResult:
    """Per-head taxonomy scores.

    ``scores[layer_index][head]`` is the score; ``eligible_rows`` counts the
    (batch, destination) rows carrying at least one mask-eligible source per
    layer -- a zero count means the score is reported ``None`` (masked),
    never a silent 0.0.
    """

    kind: str
    measure: str
    layer_addresses: tuple[str, ...]
    scores: tuple[tuple[float | None, ...], ...]
    eligible_rows: tuple[int, ...]
    provenance: dict[str, Any]

    def top(self, k: int = 5) -> tuple[tuple[str, float], ...]:
        """Return the ``k`` highest-scoring heads as ``(address.headN, score)``."""

        pairs = []
        for address, layer_scores in zip(self.layer_addresses, self.scores, strict=True):
            for head, score in enumerate(layer_scores):
                if score is not None:
                    pairs.append((f"{address}.head{head}", score))
        pairs.sort(key=lambda pair: pair[1], reverse=True)
        return tuple(pairs[: max(0, k)])


def _token_ids(trace: Any, seq_len: int) -> torch.Tensor:
    """Return the captured input token ids matching the pattern's length."""

    for label in getattr(trace, "input_ops", ()) or ():
        op = label if hasattr(label, "out") else trace.ops[str(label)]
        value = getattr(op, "out", None)
        if (
            isinstance(value, torch.Tensor)
            and not value.is_floating_point()
            and value.dim() == 2
            and int(value.shape[-1]) == seq_len
        ):
            return value
    refuse(
        code="mi_token_ids_unavailable",
        message=f"No captured integer input of sequence length {seq_len} found; the "
        "token-equality masks have no token stream to compare.",
        remedy="trace the model on token ids (the standard LM input); pass patterns from "
        "a capture whose input op was saved",
        seq_len=seq_len,
    )
    raise AssertionError("unreachable")


def _mask_for(kind: str, tokens: torch.Tensor, size: int) -> torch.Tensor:
    """Build the [batch, dst, src] token-equality mask for one kind."""

    batch = tokens.shape[0]
    dst = torch.arange(size).reshape(1, size, 1)
    src = torch.arange(size).reshape(1, 1, size)
    causal = src <= dst
    if kind == "previous_token":
        return ((src == dst - 1) & causal).expand(batch, size, size)
    equal = tokens.reshape(batch, size, 1) == tokens.reshape(batch, 1, size)
    if kind == "duplicate_token":
        return equal & (src < dst)
    # induction: the duplicate relation shifted one right -- the source FOLLOWS
    # an earlier occurrence of the destination's token.
    shifted = torch.zeros_like(equal)
    shifted[:, :, 1:] = equal[:, :, :-1]
    return shifted & causal & (src >= 1)


def _score_layer(
    pattern: torch.Tensor,
    mask: torch.Tensor,
    row_eligible: torch.Tensor,
    n_heads: int,
    measure: str,
) -> tuple[float | None, ...]:
    """Score every head of one layer against the mask (mul / abs measures)."""

    layer_scores: list[float | None] = []
    for head in range(n_heads):
        head_pattern = pattern[:, head]  # [b, dst, src]
        if measure == "mul":
            mass = (head_pattern * mask).sum(dim=-1)[row_eligible]
            layer_scores.append(float(mass.mean()))
        else:
            row_mask = mask / mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
            distance = (head_pattern - row_mask).abs().sum(dim=-1)[row_eligible]
            layer_scores.append(float(1.0 - distance.mean() / 2.0))
    return tuple(layer_scores)


def head_scores(
    trace: Any,
    *,
    kind: str = "induction",
    measure: str = "mul",
    layers: Any = None,
    pattern_source: Any = None,
) -> HeadScoreResult:
    """Score every head against one taxonomy mask (mikit section 8).

    Parameters
    ----------
    trace:
        A finished torchlens trace with attention patterns readable.
    kind:
        ``previous_token`` / ``duplicate_token`` / ``induction``.
    measure:
        ``mul`` (attention mass on mask-eligible sources, TLens-compatible)
        or ``abs`` (1 - L1/2 distance between the pattern row and the mask
        row distribution).
    layers:
        Optional layer selector (indices/addresses, as in head contributions).
    pattern_source:
        Optional callable ``module -> [b, head, dst, src]`` pattern override
        (the SSM pseudo-pattern socket); default reads the ``pattern`` facet.

    Returns
    -------
    HeadScoreResult
        Scores with eligible-row disclosure; empty-eligible layers report
        ``None`` scores, never zero.
    """

    if kind not in _KINDS:
        refuse(
            code="mi_head_score_kind_invalid",
            message=f"Unknown head-score kind {kind!r}.",
            remedy=f"choose among {list(_KINDS)}",
        )
    if measure not in _MEASURES:
        refuse(
            code="mi_head_score_kind_invalid",
            message=f"Unknown head-score measure {measure!r}.",
            remedy=f"choose among {list(_MEASURES)}",
        )
    modules = _attention_modules(trace)
    if layers is not None:
        from ._heads import _select_layers

        modules = _select_layers(modules, layers)
    if not modules:
        refuse(
            code="mi_head_geometry_unavailable",
            message="No attention modules with pattern facets in this trace "
            "(attention-free architectures have no head taxonomy).",
            remedy="pass pattern_source= to supply pseudo-patterns, or trace an attention model",
        )

    addresses: list[str] = []
    all_scores: list[tuple[float | None, ...]] = []
    eligible_counts: list[int] = []
    for module in modules:
        if pattern_source is not None:
            pattern = pattern_source(module)
        else:
            facet = module.facets["pattern"]
            pattern = facet.value if hasattr(facet, "value") else facet
        if not isinstance(pattern, torch.Tensor):
            refuse(
                code="mi_payload_missing",
                message=f"The pattern at {getattr(module, 'address', '?')} is not readable.",
                remedy="save the attention interior (retention_plan(analyses=['head_scores'])); "
                "fused SDPA needs capture=CaptureOptions(save_arg_values=True)",
                analysis="head_scores",
            )
        pattern = pattern.detach().to(torch.float32)
        batch, n_heads, dst_len, src_len = pattern.shape
        tokens = _token_ids(trace, src_len)
        mask = _mask_for(kind, tokens, src_len).to(torch.float32)  # [b, dst, src]
        row_eligible = mask.sum(dim=-1) > 0  # [b, dst]
        n_eligible = int(row_eligible.sum())
        addresses.append(str(getattr(module, "address", "")))
        eligible_counts.append(n_eligible)
        if n_eligible == 0:
            all_scores.append(tuple([None] * n_heads))
            continue
        all_scores.append(_score_layer(pattern, mask, row_eligible, n_heads, measure))

    return HeadScoreResult(
        kind=kind,
        measure=measure,
        layer_addresses=tuple(addresses),
        scores=tuple(all_scores),
        eligible_rows=tuple(eligible_counts),
        provenance={
            "mask_definition": "token-equality (TLens head_detector conventions, credited)",
            "pattern_source": "facet/pattern" if pattern_source is None else "user_callable",
        },
    )
