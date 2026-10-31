"""Per-token loss and per-context entropy strips (inspectus, matched).

Both metrics are computed from ONE logits source in one call, and the source
fingerprint travels on each record -- the position-alignment proof (the
classic defect is the off-by-one between the loss shift and the entropy
axis; composition row 9 pins it). Cells with no defined value (the first
token has no loss; padding has neither) are ``None`` and render N/A, never
zero.

All spellings are DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

from pathlib import Path

import torch

from ._errors import refuse
from ._records import Artifact, TokenMetrics, TokenScoreRow, TokenScores, tensor_fingerprint
from ._strip import render_token_strip

__all__ = ["metric_strip", "render_metric_strip", "token_metrics"]

#: The recorded loss alignment convention (printed on the record).
LOSS_CONVENTION = "loss[i] = -log p(token[i] | logits at position i-1); first token N/A"

#: The recorded entropy convention.
ENTROPY_CONVENTION = "entropy[i] = H(softmax(logits at position i)) over the full vocabulary"


def _validate_metric_inputs(
    logits: torch.Tensor, token_ids: torch.Tensor, padding_mask: torch.Tensor | None
) -> None:
    """Refuse malformed logits/token geometry before any math."""

    if logits.ndim != 2:
        refuse(
            code="tv_record_invalid",
            message=f"token_metrics expects [n_positions, vocab] logits; got rank {logits.ndim}.",
            remedy="select one batch element first (logits[batch_index])",
        )
    if token_ids.ndim != 1 or token_ids.shape[0] != logits.shape[0]:
        refuse(
            code="tv_record_invalid",
            message=f"token_ids shape {tuple(token_ids.shape)} does not align with "
            f"{logits.shape[0]} logit positions.",
            remedy="pass the input ids for exactly the positions the logits cover",
        )
    if padding_mask is not None and padding_mask.shape != token_ids.shape:
        refuse(
            code="tv_record_invalid",
            message="padding_mask must align with token_ids.",
            remedy="pass the [n_positions] attention mask (1 = real token)",
        )


def token_metrics(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    tokens: tuple[str, ...],
    *,
    padding_mask: torch.Tensor | None = None,
) -> tuple[TokenMetrics, TokenMetrics]:
    """Compute the loss and entropy strips from ONE logits source.

    Parameters
    ----------
    logits:
        ``[n_positions, vocab]`` captured logits for one sequence.
    token_ids:
        ``[n_positions]`` input token ids aligned with the logits positions.
    tokens:
        Display strings per position.
    padding_mask:
        Optional ``[n_positions]`` mask (1 = real token); padded cells
        become N/A on BOTH strips.

    Returns
    -------
    tuple[TokenMetrics, TokenMetrics]
        ``(loss_strip, entropy_strip)`` sharing one source fingerprint.
    """

    _validate_metric_inputs(logits, token_ids, padding_mask)
    if len(tokens) != logits.shape[0]:
        refuse(
            code="tv_record_invalid",
            message=f"{len(tokens)} display tokens for {logits.shape[0]} positions.",
            remedy="pass one display string per position",
        )
    work = logits.detach().to(torch.float32)
    fingerprint = tensor_fingerprint(work)
    logsumexp = torch.logsumexp(work, dim=-1)
    # Unreduced shifted cross-entropy: loss[i] scores token[i] under the
    # distribution AT position i-1 (the standard next-token alignment).
    token_logits = work[:-1].gather(1, token_ids[1:].unsqueeze(1)).squeeze(1)
    loss_tail = (logsumexp[:-1] - token_logits).tolist()
    probs = torch.softmax(work, dim=-1)
    entropy_all = (logsumexp - (probs * work).sum(dim=-1)).tolist()

    def padded(index: int) -> bool:
        """Return whether a position is padding."""

        return padding_mask is not None and not bool(padding_mask[index])

    loss_values: list[float | None] = [None]
    loss_values.extend(
        None if padded(index) or padded(index - 1) else float(loss_tail[index - 1])
        for index in range(1, logits.shape[0])
    )
    entropy_values: list[float | None] = [
        None if padded(index) else float(entropy_all[index]) for index in range(logits.shape[0])
    ]
    loss = TokenMetrics(
        tokens=tokens,
        values=tuple(loss_values),
        metric="loss",
        convention=LOSS_CONVENTION,
        source_fingerprint=fingerprint,
    )
    entropy = TokenMetrics(
        tokens=tokens,
        values=tuple(entropy_values),
        metric="entropy",
        convention=ENTROPY_CONVENTION,
        source_fingerprint=fingerprint,
    )
    return loss, entropy


def metric_strip(metrics: TokenMetrics) -> TokenScores:
    """Convert a metric record to a strip record for rendering."""

    return TokenScores(
        tokens=metrics.tokens,
        rows=(TokenScoreRow(label=metrics.metric, scores=metrics.values),),
        domain="magnitude_sequential",
        footer_lines=(metrics.convention, f"source logits: {metrics.source_fingerprint}"),
        provenance=f"per-token {metrics.metric} from captured logits",
    )


def render_metric_strip(
    metrics: TokenMetrics,
    path: Path | str,
    *,
    svg_fonttype: str = "path",
) -> Artifact:
    """Render a loss/entropy strip to paper output.

    Parameters
    ----------
    metrics:
        The metric record from :func:`token_metrics`.
    path:
        Output path (PNG/SVG/PDF by suffix).
    svg_fonttype:
        ``'path'`` (default) or ``'none'``.

    Returns
    -------
    Artifact
        The written file plus rendered disclosure lines.
    """

    return render_token_strip(metric_strip(metrics), path, svg_fonttype=svg_fonttype)
