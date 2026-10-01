"""Cancellation-aware error model for semantic facet RECONSTRUCTION gates.

This module is the SEPARATE tolerance model for reconstruction acceptance
checks (SDPA ``scores``/``pattern``/``z``/``result``, logit-lens style
projections). It deliberately does NOT reuse the replay tolerance table in
``torchlens.utils.tensor_utils``: that table models a FAITHFUL REPLAY of one
op (same kernel family, same inputs, different accumulation order), where the
absolute term may sit at denormal scale because a replay never destroys
precision. A reconstruction is different math: it re-derives a value through
an INDEPENDENT summation (per-head contributions, attention-weighted values),
and wherever that summation cancels -- large addends of opposite sign
producing a small output -- the achievable absolute accuracy is set by the
ADDEND magnitudes, not by the output value. Borrowing the replay table's
denormal-scale absolute term therefore refused numerically CORRECT
reconstructions exactly at cancellation sites (measured: 19 of 36 correct
SDPA reconstructions refused, input-dependently), while a table loose enough
to admit them would have blessed corruption during replay validation. Two
error models, two tables; never edit the replay table to serve this gate.

Error model per element::

    |reconstructed - target| <= rtol(dtype) * |target|          (agreement term)
                               + atol_floor(dtype)              (denormal jitter)
                               + c(n) * eps(accum) * magnitude  (cancellation term)

* ``rtol(dtype)`` is the same ULP-headroom shape the replay model uses: a few
  storage ULPs for fp16/bf16 (both sides accumulate in fp32 and round once to
  storage) and the accumulating 512-ULP row for fp32/fp64 (reduction-order
  drift between a fused kernel and an unfused recomputation).
* ``atol_floor(dtype)`` is the replay-shaped denormal-scale floor: it absorbs
  jitter at the bottom of the representable range only and can never bless
  small-normal corruption on its own.
* ``magnitude`` is the caller-supplied elementwise ACCUMULATED-|ADDEND| bound
  for the summation that produced ``target`` (e.g. ``pattern @ |V|`` for the
  SDPA ``z`` check, ``sum_h |result_h| + |bias|`` for the per-head result
  sum check). Where no cancellation occurred, ``magnitude ~ |target|`` and
  the term is a small rtol-like correction; where cancellation destroyed
  precision, ``magnitude >> |target|`` and the term honestly widens to what
  the arithmetic can actually promise. Omitting ``magnitude`` drops the term
  entirely (strict replay-shaped comparison).
* ``c(n) = ACCUMULATION_ULP_HEADROOM * (1 + log2(n))`` models blocked/tree
  reduction over ``n`` addends on BOTH sides of the comparison; ``eps(accum)``
  is the accumulation dtype's epsilon (fp32 for fp16/bf16/fp32 payloads, fp64
  for fp64).

Tripwire properties (pinned by tests): an all-zero, sign-flipped, or permuted
reconstruction fails wherever ``|target|`` meaningfully exceeds the budget;
the budget only widens where the addends genuinely cancelled.

Every spelling here is DOCUMENTED-UNSTABLE pending naming-session
ratification.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "ACCUMULATION_ULP_HEADROOM",
    "RECONSTRUCTION_ULP_HEADROOM",
    "reconstruction_error_budget",
    "within_reconstruction_tolerance",
]

#: Relative/floor headroom per payload dtype, in that dtype's own ULPs.
#: fp16/bf16: storage-rounding dominated (both sides accumulate in fp32 and
#: round once). fp32/fp64: fused-vs-unfused reduction-order drift.
RECONSTRUCTION_ULP_HEADROOM: dict[torch.dtype, float] = {
    torch.float16: 4.0,
    torch.bfloat16: 4.0,
    torch.float32: 512.0,
    torch.float64: 512.0,
}

#: Per-side ULP headroom of the cancellation term, before the log2(n)
#: reduction-depth factor. 16 covers the two independently-ordered
#: summations (fused kernel and recomputation) plus the final storage round.
ACCUMULATION_ULP_HEADROOM = 16.0


def _headroom_for_dtype(dtype: torch.dtype) -> float:
    """Return the ULP headroom for a payload dtype (fp32 row for unknowns)."""

    return RECONSTRUCTION_ULP_HEADROOM.get(dtype, RECONSTRUCTION_ULP_HEADROOM[torch.float32])


def _accumulation_eps(dtype: torch.dtype) -> float:
    """Return the machine epsilon of the dtype the summation accumulates in.

    fp16/bf16/fp32 payloads accumulate in fp32 (fused kernels widen; the
    recomputation is performed in fp32); fp64 accumulates in fp64.
    """

    if dtype == torch.float64:
        return float(torch.finfo(torch.float64).eps)
    return float(torch.finfo(torch.float32).eps)


def reconstruction_error_budget(
    target: torch.Tensor,
    *,
    magnitude: torch.Tensor | None = None,
    reduction_length: int | None = None,
) -> torch.Tensor:
    """Return the elementwise allowed ``|reconstructed - target|`` budget.

    Parameters
    ----------
    target:
        Captured reference tensor the reconstruction is checked against.
    magnitude:
        Elementwise accumulated-|addend| bound of the summation that produced
        ``target`` (must broadcast against it). ``None`` drops the
        cancellation term.
    reduction_length:
        Number of addends in that summation. Defaults to ``1`` when a
        ``magnitude`` is supplied without it.

    Returns
    -------
    torch.Tensor
        Float32 (or float64 for fp64 targets) elementwise budget.
    """

    finfo = torch.finfo(target.dtype if target.is_floating_point() else torch.float32)
    headroom = _headroom_for_dtype(target.dtype)
    rtol = headroom * float(finfo.eps)
    atol_floor = headroom * float(finfo.tiny) * float(finfo.eps)
    compute_dtype = torch.float64 if target.dtype == torch.float64 else torch.float32
    budget = rtol * target.detach().abs().to(compute_dtype) + atol_floor
    if magnitude is not None:
        n = max(1, int(reduction_length) if reduction_length is not None else 1)
        depth_factor = 1.0 + math.log2(n)
        cancellation = (
            ACCUMULATION_ULP_HEADROOM
            * depth_factor
            * _accumulation_eps(target.dtype)
            * magnitude.detach().abs().to(compute_dtype)
        )
        budget = budget + cancellation
    return budget


def within_reconstruction_tolerance(
    reconstructed: torch.Tensor,
    target: torch.Tensor,
    *,
    magnitude: torch.Tensor | None = None,
    reduction_length: int | None = None,
) -> bool:
    """Return whether a reconstruction matches its captured target.

    Parameters
    ----------
    reconstructed:
        Recomputed tensor (any float dtype; compared in wide precision).
    target:
        Captured reference tensor.
    magnitude:
        Elementwise accumulated-|addend| bound (see module docstring).
    reduction_length:
        Number of addends behind ``magnitude``.

    Returns
    -------
    bool
        ``True`` when every element sits inside the error budget. Shape
        mismatch is ``False``, never an exception: the gate's callers treat a
        failed check as a refusal, not a crash.
    """

    if tuple(reconstructed.shape) != tuple(target.shape):
        return False
    budget = reconstruction_error_budget(
        target, magnitude=magnitude, reduction_length=reduction_length
    )
    compute_dtype = budget.dtype
    diff = (reconstructed.detach().to(compute_dtype) - target.detach().to(compute_dtype)).abs()
    finite = torch.isfinite(target)
    if not bool(torch.isfinite(reconstructed)[finite].all()):
        return False
    if bool((~finite).any()):
        # Non-finite targets must be reproduced exactly (same NaN/inf layout).
        recon_raw = reconstructed.detach()
        target_raw = target.detach().to(recon_raw.dtype)
        nan_match = torch.isnan(recon_raw) == torch.isnan(target_raw)
        inf_match = torch.isinf(recon_raw) == torch.isinf(target_raw)
        sign_match = recon_raw.sign() == target_raw.sign()
        if not bool((nan_match & inf_match)[~finite].all()):
            return False
        if not bool(sign_match[torch.isinf(target_raw)].all()):
            return False
    return bool((diff[finite] <= budget[finite]).all())
