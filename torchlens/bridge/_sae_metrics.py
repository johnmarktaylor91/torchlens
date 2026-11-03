"""Reconstruction and next-token metrics for the SAE splice experiment.

Private helpers behind ``torchlens.bridge.sae.splice``. Every number is
computed in float64 so a hand computation reproduces it to rounding. A
metric that cannot be computed is ``None`` (or ``nan`` for the
variance-explained floats) with its reason recorded, never a silent 0.0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class VarianceMetrics:
    """Reconstruction fidelity of one splice site.

    Parameters
    ----------
    mse:
        Mean squared error over every element.
    global_fve:
        ``1 - SSE / SST`` with SST centred on the one scalar mean of all
        elements; ``nan`` when SST is zero.
    pooled_fve:
        ``1 - sum_d SSE_d / sum_d SST_d`` with each feature dimension ``d``
        centred on its own mean over tokens; ``nan`` when the pooled SST is
        zero.
    fve_by_dim:
        ``1 - SSE_d / SST_d`` per feature dimension, ``nan`` where ``SST_d``
        is zero.
    zero_variance:
        Whether a headline denominator (global or pooled SST) is zero.
    reasons:
        Metric name to the reason it is ``nan``.
    """

    mse: float
    global_fve: float
    pooled_fve: float
    fve_by_dim: torch.Tensor
    zero_variance: bool
    reasons: dict[str, str] = field(default_factory=dict)


def _token_rows(tensor: torch.Tensor) -> torch.Tensor:
    """Return ``tensor`` as float64 ``[tokens, features]`` (last axis = features)."""

    flat = tensor.detach().to(torch.float64)
    width = flat.shape[-1] if flat.ndim else 1
    return flat.reshape(-1, width)


def variance_metrics(clean: torch.Tensor, reconstruction: torch.Tensor) -> VarianceMetrics:
    """Compute MSE and the three variance-explained readings in float64.

    The pooled reading is the fraction of variance explained that SAELens
    reports in its evals and that the Gemma Scope paper reports (as its
    complement, the fraction of variance unexplained): the residual sum of
    squares over the sum of squared deviations from each feature
    dimension's mean across tokens. The feature axis is the last axis, the
    one an SAE encodes; every leading axis counts as a token axis.

    Parameters
    ----------
    clean:
        Clean site activation.
    reconstruction:
        Reconstruction compared against it, same shape.

    Returns
    -------
    VarianceMetrics
        Fidelity record; zero denominators give ``nan`` with a reason.
    """

    x = _token_rows(clean)
    residual = _token_rows(reconstruction) - x
    sse_by_dim = residual.pow(2).sum(dim=0)
    sst_by_dim = (x - x.mean(dim=0, keepdim=True)).pow(2).sum(dim=0)
    sst_global = float((x - x.mean()).pow(2).sum())
    sse = float(sse_by_dim.sum())
    sst_pooled = float(sst_by_dim.sum())
    reasons: dict[str, str] = {}
    global_fve = 1.0 - sse / sst_global if sst_global > 0.0 else math.nan
    if sst_global == 0.0:
        reasons["fraction_variance_explained"] = (
            "every element of the clean site activation is equal, so its total variance is zero"
        )
    pooled_fve = 1.0 - sse / sst_pooled if sst_pooled > 0.0 else math.nan
    if sst_pooled == 0.0:
        reasons["fraction_variance_explained_pooled"] = (
            f"no feature dimension varies across the {x.shape[0]} token(s) at the site; "
            "the per-dimension variance needs at least two tokens that differ"
        )
    fve_by_dim = torch.where(
        sst_by_dim > 0.0,
        1.0 - sse_by_dim / torch.where(sst_by_dim > 0.0, sst_by_dim, torch.ones_like(sst_by_dim)),
        torch.full_like(sst_by_dim, math.nan),
    )
    return VarianceMetrics(
        mse=float(residual.pow(2).mean()),
        global_fve=global_fve,
        pooled_fve=pooled_fve,
        fve_by_dim=fve_by_dim,
        zero_variance=sst_global == 0.0 or sst_pooled == 0.0,
        reasons=reasons,
    )


def next_token_ce(logits: torch.Tensor, tokens: torch.Tensor, start: int) -> float:
    """Return mean next-token cross-entropy in float64.

    Position ``t`` predicts token ``t + 1``; predictions from positions
    before ``start`` are skipped (``start=1`` drops the prediction made at
    the BOS position, as the Gemma Scope evaluation does).

    Parameters
    ----------
    logits:
        ``tokens.shape + (vocab,)`` logits.
    tokens:
        Integer token ids; the last axis is the sequence axis.
    start:
        First predicting position.

    Returns
    -------
    float
        Mean of ``-log softmax(logits[t])[tokens[t + 1]]`` over every
        predicting position and batch row.
    """

    axis = tokens.ndim - 1
    length = tokens.shape[axis]
    log_probs = torch.log_softmax(logits.detach().to(torch.float64), dim=-1)
    predicting = log_probs.narrow(axis, start, length - 1 - start)
    targets = tokens.narrow(axis, start + 1, length - 1 - start).to(torch.long)
    picked = predicting.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return float(-picked.mean())


def next_token_kl(clean_logits: torch.Tensor, spliced_logits: torch.Tensor, start: int) -> float:
    """Return mean ``KL(P_clean || P_spliced)`` over positions, in float64.

    The SAELens evals convention: KL of the spliced next-token distribution
    from the clean one, summed over the vocabulary, averaged over positions
    from ``start`` on (every position predicts a distribution, the last one
    included) and batch rows.

    Parameters
    ----------
    clean_logits:
        Clean logits, ``(..., sequence, vocab)``.
    spliced_logits:
        Spliced logits, same shape.
    start:
        First position counted.

    Returns
    -------
    float
        Mean KL divergence in nats.
    """

    axis = clean_logits.ndim - 2
    length = clean_logits.shape[axis]
    clean_lp = torch.log_softmax(clean_logits.detach().to(torch.float64), dim=-1)
    spliced_lp = torch.log_softmax(spliced_logits.detach().to(torch.float64), dim=-1)
    clean_lp = clean_lp.narrow(axis, start, length - start)
    spliced_lp = spliced_lp.narrow(axis, start, length - start)
    return float((clean_lp.exp() * (clean_lp - spliced_lp)).sum(dim=-1).mean())


def ce_loss_recovered(ce_clean: float, ce_spliced: float, ce_zero: float) -> float | None:
    """Return ``(CE_zero - CE_spliced) / (CE_zero - CE_clean)``, or ``None``.

    The cross-entropy loss recovered of SAELens's evals (``ce_loss_score``)
    and the Gemma Scope paper, with the site zero-ablated as the baseline.
    ``None`` when zero-ablating the site leaves the loss unchanged, since the
    ratio is then undefined.

    Parameters
    ----------
    ce_clean:
        Clean next-token cross-entropy.
    ce_spliced:
        Cross-entropy with the SAE reconstruction spliced in.
    ce_zero:
        Cross-entropy with the site zero-ablated.

    Returns
    -------
    float | None
        Fraction of the zero-ablation loss gap the reconstruction recovers.
    """

    gap = ce_zero - ce_clean
    if gap == 0.0:
        return None
    return (ce_zero - ce_spliced) / gap
