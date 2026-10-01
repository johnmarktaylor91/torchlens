"""Opt-in input tensor diagnostics (tvscope B3, memo D3/D4 -- demoted arm).

Tensor heuristics are NOT verification: measured on seven legitimate real
pipelines they false-positived on five (including plain float16/bfloat16
casts, post-normalization noise, and per-image standardization) and caught
zero of two wrong-preprocessing controls. They ship here as OPT-IN
diagnostics whose findings may CONTRADICT or FAIL-TO-CONTRADICT a declared
configuration and may NEVER return a match -- the outcome vocabulary has no
match/verified member, enforced in the return type.

Every spelling is DOCUMENTED-UNSTABLE pending the naming sprint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ._audit import EVIDENCE_CONTRADICTION, EVIDENCE_INFO, EVIDENCE_PREMISED
from ._records import DeclaredPreprocessing

__tl_layer__ = "L5"

#: Closed outcome vocabulary. Deliberately WITHOUT any match/verified member
#: (memo D4): a tensor arm allowed to say "match" would say it on genuinely
#: wrong pipelines (the measured miss table).
OUTCOME_CONTRADICTION = "contradiction"
OUTCOME_NO_CONTRADICTION = "no_contradiction"
OUTCOME_NOT_APPLICABLE = "not_applicable"

#: The default check set: intrinsic-failure checks only. The confinement arm
#: is opt-in-only (memo B3: "confinement arm opt-in-only ... first to cut").
DEFAULT_CHECKS: tuple[str, ...] = ("nonfinite", "dtype_regime", "layout")

#: Both measured blind spots of the confinement arm, printed on every
#: confinement finding (memo B3).
_CONFINEMENT_BLIND_SPOTS = (
    "blind spots: (1) misses applied-std-below-declared (3 of 6 wrong constant "
    "combinations in the measured table); (2) false-positives on legitimate "
    "pipelines (plain float16/bfloat16 casts, post-normalization noise, "
    "per-image standardization -- 5 of 7 in the measured suite)"
)


@dataclass(frozen=True)
class DiagnosticFinding:
    """One opt-in tensor check's outcome.

    Attributes
    ----------
    check:
        Check name (``"nonfinite"`` / ``"dtype_regime"`` / ``"layout"`` /
        ``"confinement"``).
    outcome:
        ``"contradiction"`` / ``"no_contradiction"`` / ``"not_applicable"``
        -- never a match (closed vocabulary, memo D4).
    evidence_class:
        Evidence ranking token: intrinsic failures are ``contradiction``-
        grade; convention-dependent checks are ``premised``.
    detail:
        Human-readable evidence, printing measured values and (for the
        confinement arm) both blind spots.
    """

    check: str
    outcome: str
    evidence_class: str
    detail: str


@dataclass(frozen=True)
class InputDiagnostics:
    """Opt-in tensor findings (tvscope memo D4, record three of three).

    This record deliberately has NO ``verified`` attribute and no
    match-shaped state: findings may contradict a declared configuration or
    fail to contradict it, nothing more.

    Attributes
    ----------
    findings:
        One row per requested check.
    """

    findings: tuple[DiagnosticFinding, ...]

    @property
    def contradictions(self) -> tuple[DiagnosticFinding, ...]:
        """Every finding whose outcome is a contradiction."""

        return tuple(f for f in self.findings if f.outcome == OUTCOME_CONTRADICTION)

    def to_json(self) -> dict[str, Any]:
        """Serialize to a JSON-portable block."""

        return {
            "schema": "tl_input_diagnostics_v1",
            "findings": [
                {
                    "check": f.check,
                    "outcome": f.outcome,
                    "evidence_class": f.evidence_class,
                    "detail": f.detail,
                }
                for f in self.findings
            ],
        }


def _check_nonfinite(batch: torch.Tensor) -> DiagnosticFinding:
    """Intrinsic check: NaN/Inf in the input batch."""

    if batch.is_floating_point():
        n_nonfinite = int((~torch.isfinite(batch)).sum().item())
    else:
        n_nonfinite = 0
    if n_nonfinite:
        return DiagnosticFinding(
            check="nonfinite",
            outcome=OUTCOME_CONTRADICTION,
            evidence_class=EVIDENCE_CONTRADICTION,
            detail=(
                f"input batch contains {n_nonfinite} non-finite values; no "
                "finite-input pipeline produces these"
            ),
        )
    return DiagnosticFinding(
        check="nonfinite",
        outcome=OUTCOME_NO_CONTRADICTION,
        evidence_class=EVIDENCE_INFO,
        detail="all input values finite",
    )


def _check_dtype_regime(
    batch: torch.Tensor, declared: DeclaredPreprocessing | None
) -> DiagnosticFinding:
    """Intrinsic check: integer batch under a declared float normalization."""

    declares_norm = declared is not None and (declared.mean is not None or declared.std is not None)
    if not batch.is_floating_point() and declares_norm:
        return DiagnosticFinding(
            check="dtype_regime",
            outcome=OUTCOME_CONTRADICTION,
            evidence_class=EVIDENCE_CONTRADICTION,
            detail=(
                f"input dtype is {batch.dtype} but the declared configuration "
                "normalizes with float mean/std; an un-rescaled integer batch "
                "cannot have passed the declared normalization"
            ),
        )
    if declared is None:
        return DiagnosticFinding(
            check="dtype_regime",
            outcome=OUTCOME_NOT_APPLICABLE,
            evidence_class=EVIDENCE_INFO,
            detail="no declared configuration to check the dtype against",
        )
    return DiagnosticFinding(
        check="dtype_regime",
        outcome=OUTCOME_NO_CONTRADICTION,
        evidence_class=EVIDENCE_INFO,
        detail=f"input dtype {batch.dtype} is consistent with the declaration",
    )


def _check_layout(batch: torch.Tensor, declared: DeclaredPreprocessing | None) -> DiagnosticFinding:
    """Premised check: channels-last layout under an image declaration.

    Premised on the NCHW convention for image batches: a 4-D batch whose
    LAST axis is 1/3 while axis 1 is not channel-sized contradicts a
    declared image pipeline. Premised, not intrinsic -- exotic layouts exist.
    """

    is_image = declared is None or declared.modality == "image"
    if batch.ndim == 4 and is_image:
        last, second = int(batch.shape[-1]), int(batch.shape[1])
        if last in (1, 3) and second not in (1, 3):
            return DiagnosticFinding(
                check="layout",
                outcome=OUTCOME_CONTRADICTION,
                evidence_class=EVIDENCE_PREMISED,
                detail=(
                    f"batch shape {tuple(batch.shape)} looks channels-last "
                    "(NHWC) while torch image models consume NCHW; premised "
                    "on the NCHW convention"
                ),
            )
    return DiagnosticFinding(
        check="layout",
        outcome=OUTCOME_NO_CONTRADICTION,
        evidence_class=EVIDENCE_INFO,
        detail="no channels-last signature detected",
    )


def _check_confinement(
    batch: torch.Tensor, declared: DeclaredPreprocessing | None
) -> DiagnosticFinding:
    """Opt-in confinement arm: inverse-normalized values confined to range.

    Measured to be the weakest arm (both blind spots printed on every
    finding); dtype-aware epsilon widens the bound for reduced-precision
    inputs so a plain fp16/bf16 cast does not fire it.
    """

    if (
        declared is None
        or declared.mean is None
        or declared.std is None
        or declared.value_range is None
        or not batch.is_floating_point()
        or batch.ndim != 4
        or not isinstance(declared.mean, tuple)
        or int(batch.shape[1]) != len(declared.mean)
    ):
        return DiagnosticFinding(
            check="confinement",
            outcome=OUTCOME_NOT_APPLICABLE,
            evidence_class=EVIDENCE_INFO,
            detail=(
                "confinement needs declared mean/std/value_range and a "
                f"channel-matched float batch; {_CONFINEMENT_BLIND_SPOTS}"
            ),
        )
    mean = torch.tensor(declared.mean, dtype=torch.float32).view(1, -1, 1, 1)
    std_declared = (
        declared.std
        if isinstance(declared.std, tuple)
        else (float(declared.std),) * len(declared.mean)
    )
    std = torch.tensor(std_declared, dtype=torch.float32).view(1, -1, 1, 1)
    low, high = (float(v) for v in declared.value_range)
    eps = 1e-3 if batch.dtype is torch.float32 else 6e-2
    restored = batch.float() * std + mean
    below = float(restored.min().item()) < low - eps
    above = float(restored.max().item()) > high + eps
    if below or above:
        return DiagnosticFinding(
            check="confinement",
            outcome=OUTCOME_CONTRADICTION,
            evidence_class=EVIDENCE_PREMISED,
            detail=(
                "inverse-normalized values leave the declared range "
                f"[{low}, {high}] (restored min {float(restored.min()):.4f}, "
                f"max {float(restored.max()):.4f}, epsilon {eps}); "
                f"{_CONFINEMENT_BLIND_SPOTS}"
            ),
        )
    return DiagnosticFinding(
        check="confinement",
        outcome=OUTCOME_NO_CONTRADICTION,
        evidence_class=EVIDENCE_INFO,
        detail=(
            "inverse-normalized values are confined to the declared range; "
            f"NOT evidence of correctness -- {_CONFINEMENT_BLIND_SPOTS}"
        ),
    )


def diagnose(
    batch: torch.Tensor,
    *,
    declared: DeclaredPreprocessing | None = None,
    checks: tuple[str, ...] = DEFAULT_CHECKS,
) -> InputDiagnostics:
    """Run opt-in tensor diagnostics against a declared configuration (B3).

    Parameters
    ----------
    batch:
        The exact tensor batch fed (or about to be fed) to the model.
    declared:
        The declared configuration the findings may contradict; omit for
        intrinsic-only checks.
    checks:
        Which checks to run. Default is the intrinsic set
        (``nonfinite`` / ``dtype_regime`` / ``layout``); the measured-weak
        ``"confinement"`` arm must be requested explicitly.

    Returns
    -------
    InputDiagnostics
        Findings that contradict or fail to contradict -- never a match, by
        type (memo D4).

    Raises
    ------
    torchlens.InvalidArgumentError
        On an unknown check name (closed vocabulary, teaching the valid set).
    """

    runners = {
        "nonfinite": lambda: _check_nonfinite(batch),
        "dtype_regime": lambda: _check_dtype_regime(batch, declared),
        "layout": lambda: _check_layout(batch, declared),
        "confinement": lambda: _check_confinement(batch, declared),
    }
    unknown = [name for name in checks if name not in runners]
    if unknown:
        from torchlens._errors import InvalidArgumentError

        raise InvalidArgumentError(
            f"unknown diagnostic checks {unknown!r}.",
            code="preprocessing_diagnostic_unknown_check",
            remedy=f"pick from {sorted(runners)!r}; the confinement arm is opt-in-only",
            unknown_checks=unknown,
        )
    return InputDiagnostics(findings=tuple(runners[name]() for name in checks))
