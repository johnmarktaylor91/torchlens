"""Attention and normalization-over-axis receptive-field rules."""

from __future__ import annotations

from .._rules import ReceptiveFieldRuleContext, _RuleResult, register_rf_rule


@register_rf_rule("softmax", "log_softmax")
def softmax(context: ReceptiveFieldRuleContext) -> _RuleResult:
    """Make the normalized dimension an exact full dependency."""

    rank = len(context.in_shapes[0]) if context.in_shapes else len(context.out_shape)
    dim = context.arg("dim", context.cfg("dim", -1))
    if not isinstance(dim, int):
        return context.unknown("softmax dimension was not captured")
    return context.full(
        axes=(dim % rank),
        exact=False,
        note="softmax family awaits a focused exactness golden",
    )


@register_rf_rule("scaled_dot_product_attention")
def scaled_dot_product_attention(context: ReceptiveFieldRuleContext) -> _RuleResult:
    """Return an honest whole-domain attention envelope, mask-aware when captured."""

    mask = context.arg("attn_mask", None)
    causal = context.arg("is_causal", False)
    masked = mask is not None or causal is True
    return context.full(
        exact=not masked,
        note="attention mask or causality narrows the full-domain envelope" if masked else None,
    )
