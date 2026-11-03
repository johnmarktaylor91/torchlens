"""Semantic axis labels, token decoding, and the SDPA hint (B3, F16).

Treescope memo section 4:

- **Semantic axis LABELS everywhere**: ``name:size`` badges replace bare
  ``axis 0/1/2`` markers at ~zero byte cost. Role-driven names apply ONLY
  where a real fact exists; today no shipped per-op axis-role fact does
  (FacetSpec carries transform chains, not axis roles), so wave 1 ships
  the deterministic positional fallback DISCLOSED, with a ``roles=`` seam
  the semantic lanes fill without rework.
- **Tokenizer decode via OUR duck-typed ``token_lookup_fn``** (upstream's
  ``for_tokenizer`` calls HF tokenizers with integers and raises); labels
  are opt-in above a size threshold because they measurably tripled HTML
  for 12 ids.
- **The SDPA hint as a product feature**: transformers 5.x defaults GPT-2
  to fused attention, so captures contain NO softmax/attention pattern;
  the condition is detectable and the card/report says so instead of
  making a silent empty promise.

Every spelling is DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "AxisLabel",
    "TOKEN_LABEL_DEFAULT_LIMIT",
    "axis_labels",
    "decode_token_labels",
    "sdpa_attention_hint",
]

#: Token labels are opt-in above this many ids (measured: labels tripled
#: HTML for 12 ids; memo section 4 tier 2).
TOKEN_LABEL_DEFAULT_LIMIT = 32


@dataclass(frozen=True)
class AxisLabel:
    """One axis badge.

    Attributes
    ----------
    name:
        Badge name -- a proven role when supplied, else the positional
        ``axis N`` fallback.
    size:
        Axis extent.
    provenance:
        ``"role"`` for caller-proven names, ``"positional"`` for the
        disclosed fallback.
    """

    name: str
    size: int
    provenance: str

    @property
    def badge(self) -> str:
        """Render the ``name:size`` badge text."""

        return f"{self.name}:{self.size}"


def axis_labels(
    shape: Sequence[int],
    roles: Sequence[str | None] | None = None,
) -> tuple[AxisLabel, ...]:
    """Build axis badges for one shape.

    Parameters
    ----------
    shape:
        Tensor shape.
    roles:
        Optional per-axis role names from a REAL fact source (semantic
        facets, dataset axis semantics). ``None`` entries fall back to the
        positional name. A wrong-length ``roles`` is ignored entirely
        (labels never guess).

    Returns
    -------
    tuple[AxisLabel, ...]
        One badge per axis; positional entries carry
        ``provenance="positional"`` so renderers can disclose the fallback.
    """

    resolved_roles: Sequence[str | None]
    if roles is not None and len(roles) == len(shape):
        resolved_roles = roles
    else:
        resolved_roles = [None] * len(shape)
    return tuple(
        AxisLabel(
            name=role if role else f"axis {index}",
            size=int(size),
            provenance="role" if role else "positional",
        )
        for index, (size, role) in enumerate(zip(shape, resolved_roles, strict=True))
    )


def decode_token_labels(
    token_ids: Sequence[int],
    token_lookup_fn: Callable[[int], str] | None,
    *,
    limit: int = TOKEN_LABEL_DEFAULT_LIMIT,
) -> tuple[str, ...] | None:
    """Decode token ids to labels through the duck-typed lookup.

    Parameters
    ----------
    token_ids:
        Ids to decode.
    token_lookup_fn:
        Any ``int -> str`` callable (an HF tokenizer's
        ``convert_ids_to_tokens`` bound per-id, a vocab dict's
        ``__getitem__``, ...). ``None`` disables decoding.
    limit:
        Above this many ids decoding is skipped (labels are opt-in above
        the measured threshold); pass a larger explicit limit to opt in.

    Returns
    -------
    tuple[str, ...] | None
        Decoded labels, or ``None`` when disabled/over-limit. A lookup
        error for one id degrades to ``str(id)`` -- decoding never raises.
    """

    if token_lookup_fn is None or len(token_ids) > limit:
        return None
    labels = []
    for token_id in token_ids:
        try:
            labels.append(str(token_lookup_fn(int(token_id))))
        except Exception:  # noqa: BLE001 - decode is cosmetic; never break a card
            labels.append(str(int(token_id)))
    return tuple(labels)


def sdpa_attention_hint(trace: Any) -> str | None:
    """Detect fused-SDPA captures with no materialized attention pattern.

    Returns the teaching hint when the capture contains
    ``scaled_dot_product_attention`` ops and NO softmax ops -- the
    transformers-5.x default that silently removes attention weights from
    the capture -- and ``None`` otherwise.
    """

    try:
        labels = [str(label) for label in (getattr(trace, "layer_labels", None) or ())]
    except Exception:  # noqa: BLE001 - a hint may never break a render
        return None
    # The captured op label spells the fused kernel without separators
    # (``scaleddotproductattention_1_1``); accept the functional spelling too.
    has_sdpa = any(
        "scaleddotproductattention" in label or "scaled_dot_product_attention" in label
        for label in labels
    )
    has_softmax = any("softmax" in label for label in labels)
    if has_sdpa and not has_softmax:
        return (
            "attention weights were never materialized (fused "
            "scaled_dot_product_attention) -- re-capture with "
            "attn_implementation='eager' to inspect attention patterns"
        )
    return None
