"""CircuitsVis bridge + BertViz tuple adapter (memo D18).

CircuitsVis is the ONE supported bridge: the payload is built from the same
typed records the static renderers read (identical coordinates and values,
different provenance), estimated BEFORE handoff, and refused typed above the
tested threshold. Dormancy and the CDN dependency are disclosed on every
handoff; offline behavior is measured, not promised -- the no-network
guarantee attaches only to torchlens's own emitters.

The BertViz-format tuple adapter is built REGARDLESS as the BERT numeric
parity oracle and documented as a recipe -- not a supported bridge, no
compatibility promise -- promotable to a public exporter later on measured
demand. Static rendering never depends on either.

All spellings are DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

import json
from typing import Any

import torch

from ._errors import refuse
from ._records import AttentionView

__all__ = ["bertviz_tuple", "circuitsvis_attention", "circuitsvis_payload"]

#: Payload ceiling for the bridge handoff in bytes (R0 tuning candidate;
#: a notebook embedding tens of MB of JSON stalls the kernel).
PAYLOAD_MAX_BYTES = 32 * 1024 * 1024

#: The dormancy/CDN disclosure attached to every bridge handoff.
BRIDGE_DISCLOSURE = (
    "circuitsvis is a dormant upstream package whose notebook component loads "
    "assets from a CDN; offline behavior is the package's, not torchlens's -- "
    "the no-network guarantee attaches only to torchlens's own emitters"
)


def circuitsvis_payload(views: list[AttentionView]) -> dict[str, Any]:
    """Build the CircuitsVis attention payload from typed views.

    Parameters
    ----------
    views:
        Per-layer attention views sharing one token axis.

    Returns
    -------
    dict[str, Any]
        ``{"tokens": [...], "attention": [layer][head][dst][src],
        "disclosure": ...}`` -- the same coordinates and values the static
        renderers draw, with bridge provenance attached.
    """

    if not views:
        refuse(
            code="tv_record_invalid",
            message="The bridge payload needs at least one attention view.",
            remedy="pass tviz.attention_views(trace)",
        )
    tokens = views[0].query_tokens.tokens
    for view in views:
        if view.query_tokens.tokens != tokens or view.key_tokens.tokens != tokens:
            refuse(
                code="tv_record_invalid",
                message="CircuitsVis attention expects one shared self-attention token "
                "axis across layers; this set has differing or rectangular axes.",
                remedy="bridge self-attention views with one shared token axis; "
                "rectangular cross-attention stays on the native static renderers",
            )
    attention = [view.pattern.detach().to(torch.float32).tolist() for view in views]
    return {
        "tokens": list(tokens),
        "attention": attention,
        "layers": [view.layer for view in views],
        "provenance": [view.provenance_wording for view in views],
        "disclosure": BRIDGE_DISCLOSURE,
    }


def circuitsvis_attention(views: list[AttentionView], *, max_bytes: int | None = None) -> Any:
    """Hand the payload to circuitsvis's attention component (notebook only).

    Parameters
    ----------
    views:
        Per-layer attention views sharing one token axis.
    max_bytes:
        Payload ceiling override; defaults to the tested threshold.

    Returns
    -------
    Any
        The circuitsvis render handle (its notebook repr embeds the
        component).

    Raises
    ------
    TvizError
        ``tv_bridge_payload_too_large`` above the ceiling (typed, with the
        measured size), or ``tv_bridge_unavailable`` when circuitsvis is
        not installed. The native static path is unaffected either way.
    """

    payload = circuitsvis_payload(views)
    limit = PAYLOAD_MAX_BYTES if max_bytes is None else max_bytes
    size = len(json.dumps(payload).encode())
    if size > limit:
        refuse(
            code="tv_bridge_payload_too_large",
            message=f"The bridge payload is {size / 1e6:.1f} MB "
            f"(ceiling {limit / 1e6:.1f} MB); embedding it would stall the notebook.",
            remedy="crop the views (view.crop_to), select fewer layers/heads, or use "
            "the native static renderers (render_attention / render_attention_atlas)",
            payload_bytes=size,
            limit_bytes=limit,
        )
    try:
        import circuitsvis.attention as cv_attention
    except ImportError:
        refuse(
            code="tv_bridge_unavailable",
            message="circuitsvis is not installed (the one supported bridge; a dormant "
            "upstream, pinned when installed).",
            remedy="pip install circuitsvis -- or use the native static renderers, "
            "which never depend on the bridge",
        )
    tensor = torch.stack([view.pattern for view in views])  # [layer, head, dst, src]
    return cv_attention.attention_heads(
        tokens=payload["tokens"], attention=tensor.reshape(-1, *tensor.shape[-2:])
    )


def bertviz_tuple(views: list[AttentionView], *, batch: bool = True) -> tuple[torch.Tensor, ...]:
    """Return attention in the BertViz/HF ``output_attentions`` tuple format.

    The BERT numeric parity ORACLE (memo D18/R4): one ``[1, n_heads, n_dst,
    n_src]`` tensor per layer, elementwise comparable against
    ``model(..., output_attentions=True).attentions``. Documented as a
    recipe, not a supported bridge; no compatibility promise.

    Parameters
    ----------
    views:
        Per-layer attention views in layer order.
    batch:
        Prepend the singleton batch dim (the HF tuple layout).

    Returns
    -------
    tuple[torch.Tensor, ...]
        One tensor per layer.
    """

    if not views:
        refuse(
            code="tv_record_invalid",
            message="The BertViz tuple adapter needs at least one attention view.",
            remedy="pass tviz.attention_views(trace)",
        )
    tensors = []
    for view in views:
        tensor = view.pattern.detach().to(torch.float32)
        tensors.append(tensor.unsqueeze(0) if batch else tensor)
    return tuple(tensors)
